#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy
import threading

from std_msgs.msg import String
from std_srvs.srv import Trigger, TriggerResponse
from my_robot_odom.srv import GotoWaypoint


class SystemSequenceManager:
    """
    상위 통합 제어 노드

    현재 구현 범위:
      1) waypoint index 0 주행
      2) 패널 기반 주행
      3) 로봇팔 panel 버튼 누르기
      4) 마커 기반 도킹
      5) 로봇팔 button 버튼 누르기 + 엘리베이터 내부 맵 전환 병렬 수행
      6) 두 작업이 모두 성공하면 마커 기반 후진

    사용하는 서비스:
      - /waypoint_navigator/goto
      - /waypoint_navigator/marker_start_1
      - /arm_mission/panel
      - /waypoint_navigator/marker_start_2
      - /arm_mission/button
      - /map_manager/switch_map
      - /map_manager/set_initial_pose
      - /waypoint_navigator/marker_start_3

    기다리는 event:
      - /waypoint_navigator/event :
          NAV_REACHED:0
          PANEL_DONE:3
          MARKER_FWD_DONE:3
          MAP_SWITCH_DONE
          LOCALIZATION_DONE
          MARKER_BACK_DONE:3
          NAV_FAILED
          STOPPED

      - /arm_mission/event :
          SEXY_PANEL
          SEXY_BUTTON
    """

    ST_IDLE = "IDLE"
    ST_NAV_TO_PANEL = "NAV_TO_PANEL"
    ST_PANEL_APPROACH = "PANEL_APPROACH"
    ST_ARM_PANEL = "ARM_PANEL"
    ST_MARKER_DOCKING = "MARKER_DOCKING"

    # 추가된 상태
    ST_BUTTON_AND_MAP_SWITCH = "BUTTON_AND_MAP_SWITCH"

    ST_ARM_BUTTON = "ARM_BUTTON"
    ST_MARKER_BACK = "MARKER_BACK"
    ST_DONE = "DONE"
    ST_ERROR = "ERROR"

    def __init__(self):
        # =====================================================
        # Params
        # =====================================================
        self.start_index = int(rospy.get_param("~start_index", 0))

        self.panel_done_event = rospy.get_param("~panel_done_event", "PANEL_DONE:3")
        self.arm_panel_done_event = rospy.get_param("~arm_panel_done_event", "SEXY_PANEL")
        self.marker_fwd_done_event = rospy.get_param("~marker_fwd_done_event", "MARKER_FWD_DONE:3")
        self.arm_button_done_event = rospy.get_param("~arm_button_done_event", "SEXY_BUTTON")
        self.marker_back_done_event = rospy.get_param("~marker_back_done_event", "MARKER_BACK_DONE:3")

        # 추가된 이벤트
        self.map_switch_done_event = rospy.get_param("~map_switch_done_event", "MAP_SWITCH_DONE")
        self.localization_done_event = rospy.get_param("~localization_done_event", "LOCALIZATION_DONE")

        self.nav_event_topic = rospy.get_param("~nav_event_topic", "/waypoint_navigator/event")
        self.arm_event_topic = rospy.get_param("~arm_event_topic", "/arm_mission/event")

        self.goto_srv_name = rospy.get_param("~goto_srv_name", "/waypoint_navigator/goto")

        self.marker_start_1_srv_name = rospy.get_param(
            "~marker_start_1_srv_name",
            "/waypoint_navigator/marker_start_1"
        )

        self.marker_start_2_srv_name = rospy.get_param(
            "~marker_start_2_srv_name",
            "/waypoint_navigator/marker_start_2"
        )

        self.marker_start_3_srv_name = rospy.get_param(
            "~marker_start_3_srv_name",
            "/waypoint_navigator/marker_start_3"
        )

        self.arm_panel_srv_name = rospy.get_param("~arm_panel_srv_name", "/arm_mission/panel")
        self.arm_button_srv_name = rospy.get_param("~arm_button_srv_name", "/arm_mission/button")

        # 추가된 맵 전환 관련 서비스
        self.map_switch_srv_name = rospy.get_param(
            "~map_switch_srv_name",
            "/map_manager/switch_map"
        )

        self.set_initial_pose_srv_name = rospy.get_param(
            "~set_initial_pose_srv_name",
            "/map_manager/set_initial_pose"
        )

        self.wait_service_timeout = float(rospy.get_param("~wait_service_timeout", 10.0))
        self.event_timeout = float(rospy.get_param("~event_timeout", 180.0))

        self.auto_start = bool(rospy.get_param("~auto_start", True))
        self.start_delay = float(rospy.get_param("~start_delay", 1.0))

        # =====================================================
        # Internal State
        # =====================================================
        self.state = self.ST_IDLE
        self.last_nav_event = ""
        self.last_arm_event = ""

        self.running = False
        self.finished = False
        self.failed = False

        self.lock = threading.Lock()
        self.event_cv = threading.Condition(self.lock)

        # =====================================================
        # Subscribers
        # =====================================================
        rospy.Subscriber(self.nav_event_topic, String, self._nav_event_cb, queue_size=10)
        rospy.Subscriber(self.arm_event_topic, String, self._arm_event_cb, queue_size=10)

        # =====================================================
        # Services provided by this manager
        # =====================================================
        self.srv_start = rospy.Service("~start", Trigger, self._srv_start)
        self.srv_stop = rospy.Service("~stop", Trigger, self._srv_stop)

        # =====================================================
        # Service Proxies
        # =====================================================
        self.goto_srv = None
        self.marker_start_1_srv = None
        self.marker_start_2_srv = None
        self.marker_start_3_srv = None
        self.arm_panel_srv = None
        self.arm_button_srv = None

        # 추가된 서비스 프록시
        self.map_switch_srv = None
        self.set_initial_pose_srv = None

        rospy.loginfo("[sequence] ready")
        rospy.loginfo("[sequence] nav_event_topic=%s", self.nav_event_topic)
        rospy.loginfo("[sequence] arm_event_topic=%s", self.arm_event_topic)
        rospy.loginfo("[sequence] goto_srv=%s", self.goto_srv_name)
        rospy.loginfo("[sequence] marker_start_1_srv=%s", self.marker_start_1_srv_name)
        rospy.loginfo("[sequence] marker_start_2_srv=%s", self.marker_start_2_srv_name)
        rospy.loginfo("[sequence] marker_start_3_srv=%s", self.marker_start_3_srv_name)
        rospy.loginfo("[sequence] arm_panel_srv=%s", self.arm_panel_srv_name)
        rospy.loginfo("[sequence] arm_button_srv=%s", self.arm_button_srv_name)

        # 추가된 로그
        rospy.loginfo("[sequence] map_switch_srv=%s", self.map_switch_srv_name)
        rospy.loginfo("[sequence] set_initial_pose_srv=%s", self.set_initial_pose_srv_name)

        rospy.loginfo("[sequence] panel_done_event=%s", self.panel_done_event)
        rospy.loginfo("[sequence] arm_panel_done_event=%s", self.arm_panel_done_event)
        rospy.loginfo("[sequence] marker_fwd_done_event=%s", self.marker_fwd_done_event)
        rospy.loginfo("[sequence] arm_button_done_event=%s", self.arm_button_done_event)

        # 추가된 로그
        rospy.loginfo("[sequence] map_switch_done_event=%s", self.map_switch_done_event)
        rospy.loginfo("[sequence] localization_done_event=%s", self.localization_done_event)

        rospy.loginfo("[sequence] marker_back_done_event=%s", self.marker_back_done_event)

        if self.auto_start:
            threading.Thread(target=self._delayed_auto_start, daemon=True).start()

    # =====================================================
    # Event callbacks
    # =====================================================
    def _nav_event_cb(self, msg):
        event = str(msg.data)

        with self.event_cv:
            self.last_nav_event = event
            rospy.loginfo("[sequence] nav event received: %s", event)
            self.event_cv.notify_all()

    def _arm_event_cb(self, msg):
        event = str(msg.data)

        with self.event_cv:
            self.last_arm_event = event
            rospy.loginfo("[sequence] arm event received: %s", event)
            self.event_cv.notify_all()

    # =====================================================
    # Service connection helpers
    # =====================================================
    def _connect_services(self):
        try:
            rospy.loginfo("[sequence] waiting service: %s", self.goto_srv_name)
            rospy.wait_for_service(self.goto_srv_name, timeout=self.wait_service_timeout)
            self.goto_srv = rospy.ServiceProxy(self.goto_srv_name, GotoWaypoint)

            rospy.loginfo("[sequence] waiting service: %s", self.marker_start_1_srv_name)
            rospy.wait_for_service(self.marker_start_1_srv_name, timeout=self.wait_service_timeout)
            self.marker_start_1_srv = rospy.ServiceProxy(self.marker_start_1_srv_name, Trigger)

            rospy.loginfo("[sequence] waiting service: %s", self.arm_panel_srv_name)
            rospy.wait_for_service(self.arm_panel_srv_name, timeout=self.wait_service_timeout)
            self.arm_panel_srv = rospy.ServiceProxy(self.arm_panel_srv_name, Trigger)

            rospy.loginfo("[sequence] waiting service: %s", self.marker_start_2_srv_name)
            rospy.wait_for_service(self.marker_start_2_srv_name, timeout=self.wait_service_timeout)
            self.marker_start_2_srv = rospy.ServiceProxy(self.marker_start_2_srv_name, Trigger)

            rospy.loginfo("[sequence] waiting service: %s", self.arm_button_srv_name)
            rospy.wait_for_service(self.arm_button_srv_name, timeout=self.wait_service_timeout)
            self.arm_button_srv = rospy.ServiceProxy(self.arm_button_srv_name, Trigger)

            # =================================================
            # 추가: 맵 전환 서비스 연결
            # =================================================
            rospy.loginfo("[sequence] waiting service: %s", self.map_switch_srv_name)
            rospy.wait_for_service(self.map_switch_srv_name, timeout=self.wait_service_timeout)
            self.map_switch_srv = rospy.ServiceProxy(self.map_switch_srv_name, Trigger)

            rospy.loginfo("[sequence] waiting service: %s", self.set_initial_pose_srv_name)
            rospy.wait_for_service(self.set_initial_pose_srv_name, timeout=self.wait_service_timeout)
            self.set_initial_pose_srv = rospy.ServiceProxy(self.set_initial_pose_srv_name, Trigger)

            rospy.loginfo("[sequence] waiting service: %s", self.marker_start_3_srv_name)
            rospy.wait_for_service(self.marker_start_3_srv_name, timeout=self.wait_service_timeout)
            self.marker_start_3_srv = rospy.ServiceProxy(self.marker_start_3_srv_name, Trigger)

            rospy.loginfo("[sequence] all services connected")
            return True

        except Exception as e:
            rospy.logerr("[sequence] service connection failed: %s", e)
            return False

    # =====================================================
    # Wait helpers
    # =====================================================
    def _wait_nav_event(self, target_event, timeout=None):
        if timeout is None:
            timeout = self.event_timeout

        rospy.loginfo("[sequence] waiting nav event: %s", target_event)
        start_time = rospy.Time.now()

        with self.event_cv:
            while not rospy.is_shutdown() and self.running:
                if self.last_nav_event == target_event:
                    rospy.loginfo("[sequence] nav event matched: %s", target_event)
                    return True

                if self.last_nav_event.startswith("NAV_FAILED"):
                    rospy.logerr("[sequence] navigation failed event received: %s", self.last_nav_event)
                    return False

                if self.last_nav_event == "STOPPED":
                    rospy.logerr("[sequence] navigation stopped event received")
                    return False

                elapsed = (rospy.Time.now() - start_time).to_sec()
                if elapsed > timeout:
                    rospy.logerr("[sequence] timeout waiting nav event: %s", target_event)
                    return False

                self.event_cv.wait(timeout=0.2)

        return False

    def _wait_arm_event(self, target_event, timeout=None):
        if timeout is None:
            timeout = self.event_timeout

        rospy.loginfo("[sequence] waiting arm event: %s", target_event)
        start_time = rospy.Time.now()

        with self.event_cv:
            while not rospy.is_shutdown() and self.running:
                if self.last_arm_event == target_event:
                    rospy.loginfo("[sequence] arm event matched: %s", target_event)
                    return True

                elapsed = (rospy.Time.now() - start_time).to_sec()
                if elapsed > timeout:
                    rospy.logerr("[sequence] timeout waiting arm event: %s", target_event)
                    return False

                self.event_cv.wait(timeout=0.2)

        return False

    # =====================================================
    # Command helpers
    # =====================================================
    def _call_goto(self, index):
        rospy.loginfo("[sequence] call goto index=%d", index)

        try:
            resp = self.goto_srv(index)

            if not resp.success:
                rospy.logerr("[sequence] goto failed to start: %s", resp.message)
                return False

            rospy.loginfo("[sequence] goto started: %s", resp.message)
            return True

        except Exception as e:
            rospy.logerr("[sequence] goto service call error: %s", e)
            return False

    def _call_marker_start_1(self):
        rospy.loginfo("[sequence] call marker_start_1")

        try:
            resp = self.marker_start_1_srv()

            if not resp.success:
                rospy.logerr("[sequence] marker_start_1 failed to start: %s", resp.message)
                return False

            rospy.loginfo("[sequence] marker_start_1 started: %s", resp.message)
            return True

        except Exception as e:
            rospy.logerr("[sequence] marker_start_1 service call error: %s", e)
            return False

    def _call_arm_panel(self):
        rospy.loginfo("[sequence] call arm panel mission")

        try:
            resp = self.arm_panel_srv()

            if not resp.success:
                rospy.logerr("[sequence] arm panel mission failed to start: %s", resp.message)
                return False

            rospy.loginfo("[sequence] arm panel mission started: %s", resp.message)
            return True

        except Exception as e:
            rospy.logerr("[sequence] arm panel service call error: %s", e)
            return False

    def _call_marker_start_2(self):
        rospy.loginfo("[sequence] call marker_start_2")

        try:
            resp = self.marker_start_2_srv()

            if not resp.success:
                rospy.logerr("[sequence] marker_start_2 failed to start: %s", resp.message)
                return False

            rospy.loginfo("[sequence] marker_start_2 started: %s", resp.message)
            return True

        except Exception as e:
            rospy.logerr("[sequence] marker_start_2 service call error: %s", e)
            return False

    def _call_arm_button(self):
        rospy.loginfo("[sequence] call arm button mission")

        try:
            resp = self.arm_button_srv()

            if not resp.success:
                rospy.logerr("[sequence] arm button mission failed to start: %s", resp.message)
                return False

            rospy.loginfo("[sequence] arm button mission started: %s", resp.message)
            return True

        except Exception as e:
            rospy.logerr("[sequence] arm button service call error: %s", e)
            return False

    # =====================================================
    # 추가: 맵 전환 및 초기 위치 재설정 호출 함수
    # =====================================================
    def _call_map_switch(self):
        rospy.loginfo("[sequence] call map switch")

        try:
            resp = self.map_switch_srv()

            if not resp.success:
                rospy.logerr("[sequence] map switch failed to start: %s", resp.message)
                return False

            rospy.loginfo("[sequence] map switch started: %s", resp.message)
            return True

        except Exception as e:
            rospy.logerr("[sequence] map switch service call error: %s", e)
            return False

    def _call_set_initial_pose(self):
        rospy.loginfo("[sequence] call set initial pose")

        try:
            resp = self.set_initial_pose_srv()

            if not resp.success:
                rospy.logerr("[sequence] set initial pose failed to start: %s", resp.message)
                return False

            rospy.loginfo("[sequence] set initial pose started: %s", resp.message)
            return True

        except Exception as e:
            rospy.logerr("[sequence] set initial pose service call error: %s", e)
            return False

    def _call_marker_start_3(self):
        rospy.loginfo("[sequence] call marker_start_3")

        try:
            resp = self.marker_start_3_srv()

            if not resp.success:
                rospy.logerr("[sequence] marker_start_3 failed to start: %s", resp.message)
                return False

            rospy.loginfo("[sequence] marker_start_3 started: %s", resp.message)
            return True

        except Exception as e:
            rospy.logerr("[sequence] marker_start_3 service call error: %s", e)
            return False

    # =====================================================
    # 추가: 버튼 누르기 + 맵 전환 병렬 실행
    # =====================================================
    def _run_button_and_map_switch_parallel(self):
        """
        엘리베이터 탑승 완료 후 다음 두 작업을 병렬로 실행한다.

        A. 로봇팔 button 버튼 누르기
           - /arm_mission/button 호출
           - /arm_mission/event 에서 SEXY_BUTTON 확인

        B. 엘리베이터 내부 맵 전환 및 초기 위치 재설정
           - /map_manager/switch_map 호출
           - /waypoint_navigator/event 에서 MAP_SWITCH_DONE 확인
           - /map_manager/set_initial_pose 호출
           - /waypoint_navigator/event 에서 LOCALIZATION_DONE 확인

        다음 단계 진행 조건:
           A와 B가 모두 성공해야 True 반환
        """

        results = {
            "button": False,
            "map": False
        }

        def button_task():
            rospy.loginfo("[sequence] parallel task A start: arm button mission")

            if not self._call_arm_button():
                rospy.logerr("[sequence] parallel task A failed: arm button service failed")
                results["button"] = False
                return

            if not self._wait_arm_event(self.arm_button_done_event):
                rospy.logerr(
                    "[sequence] parallel task A failed while waiting %s",
                    self.arm_button_done_event
                )
                results["button"] = False
                return

            results["button"] = True
            rospy.loginfo("[sequence] parallel task A success: arm button mission done")

        def map_task():
            rospy.loginfo("[sequence] parallel task B start: map switch + localization")

            if not self._call_map_switch():
                rospy.logerr("[sequence] parallel task B failed: map switch service failed")
                results["map"] = False
                return

            if not self._wait_nav_event(self.map_switch_done_event):
                rospy.logerr(
                    "[sequence] parallel task B failed while waiting %s",
                    self.map_switch_done_event
                )
                results["map"] = False
                return

            if not self._call_set_initial_pose():
                rospy.logerr("[sequence] parallel task B failed: set initial pose service failed")
                results["map"] = False
                return

            if not self._wait_nav_event(self.localization_done_event):
                rospy.logerr(
                    "[sequence] parallel task B failed while waiting %s",
                    self.localization_done_event
                )
                results["map"] = False
                return

            results["map"] = True
            rospy.loginfo("[sequence] parallel task B success: map switch + localization done")

        button_thread = threading.Thread(target=button_task)
        map_thread = threading.Thread(target=map_task)

        button_thread.start()
        map_thread.start()

        button_thread.join()
        map_thread.join()

        rospy.loginfo(
            "[sequence] parallel result: button=%s, map=%s",
            results["button"],
            results["map"]
        )

        if results["button"] and results["map"]:
            rospy.loginfo("[sequence] both parallel tasks succeeded")
            return True

        rospy.logerr("[sequence] one or more parallel tasks failed")
        return False

    # =====================================================
    # Main sequence
    # =====================================================
    def run_sequence(self):
        with self.lock:
            if self.running:
                rospy.logwarn("[sequence] already running")
                return False

            self.running = True
            self.finished = False
            self.failed = False
            self.state = self.ST_IDLE
            self.last_nav_event = ""
            self.last_arm_event = ""

        rospy.loginfo("[sequence] sequence start")

        if not self._connect_services():
            return self._finish_failed("service connection failed")

        # -------------------------------------------------
        # 1. waypoint index 0 주행
        # -------------------------------------------------
        self.state = self.ST_NAV_TO_PANEL
        rospy.loginfo("[sequence] STEP 1: waypoint navigation to index %d", self.start_index)

        if not self._call_goto(self.start_index):
            return self._finish_failed("goto service failed")

        nav_reached_event = "NAV_REACHED:%d" % self.start_index

        if not self._wait_nav_event(nav_reached_event):
            return self._finish_failed("failed while waiting %s" % nav_reached_event)

        # -------------------------------------------------
        # 2. 패널 기반 주행
        # -------------------------------------------------
        self.state = self.ST_PANEL_APPROACH
        rospy.loginfo("[sequence] STEP 2: panel approach")

        if not self._call_marker_start_1():
            return self._finish_failed("marker_start_1 service failed")

        if not self._wait_nav_event(self.panel_done_event):
            return self._finish_failed("failed while waiting %s" % self.panel_done_event)

        # -------------------------------------------------
        # 3. 로봇팔 panel 버튼 누르기
        # -------------------------------------------------
        self.state = self.ST_ARM_PANEL
        rospy.loginfo("[sequence] STEP 3: arm panel mission")

        if not self._call_arm_panel():
            return self._finish_failed("arm panel service failed")

        if not self._wait_arm_event(self.arm_panel_done_event):
            return self._finish_failed("failed while waiting %s" % self.arm_panel_done_event)

        # -------------------------------------------------
        # 4. 마커 기반 도킹
        # -------------------------------------------------
        self.state = self.ST_MARKER_DOCKING
        rospy.loginfo("[sequence] STEP 4: marker docking")

        if not self._call_marker_start_2():
            return self._finish_failed("marker_start_2 service failed")

        if not self._wait_nav_event(self.marker_fwd_done_event):
            return self._finish_failed("failed while waiting %s" % self.marker_fwd_done_event)

        # -------------------------------------------------
        # 5. 버튼 누르기 + 맵 전환 병렬 실행
        # -------------------------------------------------
        self.state = self.ST_BUTTON_AND_MAP_SWITCH
        rospy.loginfo("[sequence] STEP 5: arm button + map switch parallel")

        if not self._run_button_and_map_switch_parallel():
            return self._finish_failed("button mission or map switch failed")

        # -------------------------------------------------
        # 6. 마커 기반 후진
        # -------------------------------------------------
        self.state = self.ST_MARKER_BACK
        rospy.loginfo("[sequence] STEP 6: marker backward")

        if not self._call_marker_start_3():
            return self._finish_failed("marker_start_3 service failed")

        if not self._wait_nav_event(self.marker_back_done_event):
            return self._finish_failed("failed while waiting %s" % self.marker_back_done_event)

        return self._finish_success()

    # =====================================================
    # Finish helpers
    # =====================================================
    def _finish_success(self):
        with self.lock:
            self.state = self.ST_DONE
            self.running = False
            self.finished = True
            self.failed = False

        rospy.loginfo("[sequence] sequence completed successfully up to STEP 6")
        return True

    def _finish_failed(self, reason):
        with self.lock:
            self.state = self.ST_ERROR
            self.running = False
            self.finished = True
            self.failed = True

        rospy.logerr("[sequence] sequence failed: %s", reason)
        return False

    # =====================================================
    # External services
    # =====================================================
    def _srv_start(self, _req):
        with self.lock:
            if self.running:
                return TriggerResponse(success=False, message="sequence already running")

        threading.Thread(target=self.run_sequence, daemon=True).start()
        return TriggerResponse(success=True, message="sequence started")

    def _srv_stop(self, _req):
        with self.lock:
            self.running = False
            self.state = self.ST_IDLE

        rospy.logwarn("[sequence] stop requested")
        return TriggerResponse(success=True, message="sequence stop requested")

    def _delayed_auto_start(self):
        if self.start_delay > 0.0:
            rospy.sleep(self.start_delay)

        if rospy.is_shutdown():
            return

        rospy.loginfo("[sequence] auto_start enabled")
        self.run_sequence()


if __name__ == "__main__":
    rospy.init_node("system_sequence_manager")

    try:
        SystemSequenceManager()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
