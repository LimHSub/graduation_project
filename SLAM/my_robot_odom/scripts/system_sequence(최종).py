#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy
import threading

from std_msgs.msg import String
from std_srvs.srv import Trigger, TriggerResponse
from geometry_msgs.msg import Twist
from my_robot_odom.srv import GotoWaypoint


class SystemSequenceManager:
    """
    상위 통합 제어 노드

    현재 구현 범위:
      0) 시작 조건 처리
         - auto_start=True  : NAV_START를 기다리지 않고 자동 시작
         - auto_start=False : /loading_mission/event 에서 NAV_START를 받아야 시작
         - 실제 시퀀스 시작 직후 /cmd_vel_nav로 3초 직진

      1) waypoint index 0 주행
      2) 패널 기반 주행
      3) 로봇팔 panel 버튼 누르기
      4) 마커 기반 도킹
      5) 로봇팔 button 버튼 누르기 + 맵 전환 병렬 수행
      6) 마커 기반 후진
      7) waypoint index 1 주행
      8) loading_mission finish 처리

    사용하는 서비스:
      - /waypoint_navigator/goto
      - /waypoint_navigator/marker_start_1
      - /arm_mission/panel
      - /waypoint_navigator/marker_start_2
      - /arm_mission/button
      - /waypoint_navigator/switch_next_map
      - /waypoint_navigator/marker_start_3
      - /loading_mission/finish

    기다리는 event:
      - /loading_mission/event :
          NAV_START
          FINISHED

      - /waypoint_navigator/event :
          NAV_REACHED:0
          PANEL_DONE:3
          MARKER_FWD_DONE:3
          MAP_SWITCHED:B
          MARKER_BACK_DONE:3
          NAV_REACHED:1
          NAV_FAILED
          STOPPED

      - /arm_mission/event :
          SEXY_PANEL
          SEXY_BUTTON

    수정 사항:
      - waypoint 주행 단계에서는 NAV_FAILED가 들어와도 바로 실패 처리하지 않음.
      - 이후 NAV_REACHED가 들어오면 다음 시퀀스로 진행함.
      - 단, 마커 주행 단계에서는 기존처럼 NAV_FAILED를 실패 처리함.
      - auto_start 여부와 관계없이 실제 시퀀스 시작 시 3초 직진을 수행함.
      - 마지막 waypoint index 1 도착 후 /loading_mission/finish 호출 및 FINISHED 대기.
    """

    ST_IDLE = "IDLE"

    ST_WAIT_LOADING_START = "WAIT_LOADING_START"
    ST_START_FORWARD = "START_FORWARD"

    ST_NAV_TO_PANEL = "NAV_TO_PANEL"
    ST_PANEL_APPROACH = "PANEL_APPROACH"
    ST_ARM_PANEL = "ARM_PANEL"
    ST_MARKER_DOCKING = "MARKER_DOCKING"

    ST_BUTTON_AND_MAP_SWITCH = "BUTTON_AND_MAP_SWITCH"

    ST_ARM_BUTTON = "ARM_BUTTON"
    ST_MARKER_BACK = "MARKER_BACK"
    ST_NAV_TO_NEXT = "NAV_TO_NEXT"

    ST_LOADING_FINISH = "LOADING_FINISH"

    ST_DONE = "DONE"
    ST_ERROR = "ERROR"

    def __init__(self):
        # =====================================================
        # Params
        # =====================================================
        self.start_index = int(rospy.get_param("~start_index", 0))
        self.next_index = int(rospy.get_param("~next_index", 1))

        self.panel_done_event = rospy.get_param("~panel_done_event", "PANEL_DONE:3")
        self.arm_panel_done_event = rospy.get_param("~arm_panel_done_event", "SEXY_PANEL")
        self.marker_fwd_done_event = rospy.get_param("~marker_fwd_done_event", "MARKER_FWD_DONE:3")
        self.arm_button_done_event = rospy.get_param("~arm_button_done_event", "SEXY_BUTTON")
        self.map_switched_event = rospy.get_param("~map_switched_event", "MAP_SWITCHED:B")
        self.marker_back_done_event = rospy.get_param("~marker_back_done_event", "MARKER_BACK_DONE:3")

        # loading mission event
        self.loading_start_event = rospy.get_param("~loading_start_event", "NAV_START")
        self.loading_finished_event = rospy.get_param("~loading_finished_event", "FINISHED")

        self.nav_event_topic = rospy.get_param("~nav_event_topic", "/waypoint_navigator/event")
        self.arm_event_topic = rospy.get_param("~arm_event_topic", "/arm_mission/event")
        self.loading_event_topic = rospy.get_param("~loading_event_topic", "/loading_mission/event")

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

        self.switch_next_map_srv_name = rospy.get_param(
            "~switch_next_map_srv_name",
            "/waypoint_navigator/switch_next_map"
        )

        self.arm_panel_srv_name = rospy.get_param("~arm_panel_srv_name", "/arm_mission/panel")
        self.arm_button_srv_name = rospy.get_param("~arm_button_srv_name", "/arm_mission/button")

        self.loading_finish_srv_name = rospy.get_param(
            "~loading_finish_srv_name",
            "/loading_mission/finish"
        )

        self.wait_service_timeout = float(rospy.get_param("~wait_service_timeout", 10.0))
        self.event_timeout = float(rospy.get_param("~event_timeout", 300.0))

        self.auto_start = bool(rospy.get_param("~auto_start", True))
        self.start_delay = float(rospy.get_param("~start_delay", 1.0))

        # 시작 직진 설정
        self.start_forward_enabled = bool(rospy.get_param("~start_forward_enabled", True))
        self.start_forward_cmd_topic = rospy.get_param("~start_forward_cmd_topic", "/cmd_vel_nav")
        self.start_forward_duration = float(rospy.get_param("~start_forward_duration", 3.0))
        self.start_forward_linear_x = float(rospy.get_param("~start_forward_linear_x", 0.25))
        self.start_forward_rate = float(rospy.get_param("~start_forward_rate", 20.0))

        # =====================================================
        # Internal State
        # =====================================================
        self.state = self.ST_IDLE
        self.last_nav_event = ""
        self.last_arm_event = ""
        self.last_loading_event = ""

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
        rospy.Subscriber(self.loading_event_topic, String, self._loading_event_cb, queue_size=10)

        # =====================================================
        # Publishers
        # =====================================================
        self.start_forward_pub = rospy.Publisher(
            self.start_forward_cmd_topic,
            Twist,
            queue_size=1
        )

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
        self.switch_next_map_srv = None
        self.arm_panel_srv = None
        self.arm_button_srv = None
        self.loading_finish_srv = None

        rospy.loginfo("[sequence] ready")
        rospy.loginfo("[sequence] nav_event_topic=%s", self.nav_event_topic)
        rospy.loginfo("[sequence] arm_event_topic=%s", self.arm_event_topic)
        rospy.loginfo("[sequence] loading_event_topic=%s", self.loading_event_topic)

        rospy.loginfo("[sequence] goto_srv=%s", self.goto_srv_name)
        rospy.loginfo("[sequence] marker_start_1_srv=%s", self.marker_start_1_srv_name)
        rospy.loginfo("[sequence] marker_start_2_srv=%s", self.marker_start_2_srv_name)
        rospy.loginfo("[sequence] marker_start_3_srv=%s", self.marker_start_3_srv_name)
        rospy.loginfo("[sequence] switch_next_map_srv=%s", self.switch_next_map_srv_name)
        rospy.loginfo("[sequence] arm_panel_srv=%s", self.arm_panel_srv_name)
        rospy.loginfo("[sequence] arm_button_srv=%s", self.arm_button_srv_name)
        rospy.loginfo("[sequence] loading_finish_srv=%s", self.loading_finish_srv_name)

        rospy.loginfo("[sequence] start_index=%d", self.start_index)
        rospy.loginfo("[sequence] next_index=%d", self.next_index)

        rospy.loginfo("[sequence] panel_done_event=%s", self.panel_done_event)
        rospy.loginfo("[sequence] arm_panel_done_event=%s", self.arm_panel_done_event)
        rospy.loginfo("[sequence] marker_fwd_done_event=%s", self.marker_fwd_done_event)
        rospy.loginfo("[sequence] arm_button_done_event=%s", self.arm_button_done_event)
        rospy.loginfo("[sequence] map_switched_event=%s", self.map_switched_event)
        rospy.loginfo("[sequence] marker_back_done_event=%s", self.marker_back_done_event)

        rospy.loginfo("[sequence] loading_start_event=%s", self.loading_start_event)
        rospy.loginfo("[sequence] loading_finished_event=%s", self.loading_finished_event)

        rospy.loginfo("[sequence] auto_start=%s", str(self.auto_start))
        rospy.loginfo("[sequence] start_forward_enabled=%s", str(self.start_forward_enabled))
        rospy.loginfo("[sequence] start_forward_cmd_topic=%s", self.start_forward_cmd_topic)
        rospy.loginfo("[sequence] start_forward_duration=%.2f", self.start_forward_duration)
        rospy.loginfo("[sequence] start_forward_linear_x=%.3f", self.start_forward_linear_x)

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

    def _loading_event_cb(self, msg):
        event = str(msg.data)

        with self.event_cv:
            self.last_loading_event = event
            rospy.loginfo("[sequence] loading event received: %s", event)
            self.event_cv.notify_all()

            # auto_start=False일 때는 NAV_START를 받으면 자동으로 run_sequence 시작
            if (not self.auto_start) and (not self.running) and (not self.finished):
                if event == self.loading_start_event:
                    rospy.loginfo("[sequence] loading start event received -> start sequence")
                    threading.Thread(target=self.run_sequence, daemon=True).start()

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

            rospy.loginfo("[sequence] waiting service: %s", self.switch_next_map_srv_name)
            rospy.wait_for_service(self.switch_next_map_srv_name, timeout=self.wait_service_timeout)
            self.switch_next_map_srv = rospy.ServiceProxy(self.switch_next_map_srv_name, Trigger)

            rospy.loginfo("[sequence] waiting service: %s", self.marker_start_3_srv_name)
            rospy.wait_for_service(self.marker_start_3_srv_name, timeout=self.wait_service_timeout)
            self.marker_start_3_srv = rospy.ServiceProxy(self.marker_start_3_srv_name, Trigger)

            rospy.loginfo("[sequence] waiting service: %s", self.loading_finish_srv_name)
            rospy.wait_for_service(self.loading_finish_srv_name, timeout=self.wait_service_timeout)
            self.loading_finish_srv = rospy.ServiceProxy(self.loading_finish_srv_name, Trigger)

            rospy.loginfo("[sequence] all services connected")
            return True

        except Exception as e:
            rospy.logerr("[sequence] service connection failed: %s", e)
            return False

    # =====================================================
    # Wait helpers
    # =====================================================
    def _wait_nav_event(self, target_event, timeout=None, ignore_nav_failed=False):
        """
        /waypoint_navigator/event 대기 함수

        ignore_nav_failed=False:
          - 기존 동작 유지
          - NAV_FAILED 계열 이벤트가 들어오면 즉시 실패 처리

        ignore_nav_failed=True:
          - NAV_FAILED 계열 이벤트가 들어와도 실패 처리하지 않음
          - 이후 target_event가 들어오면 성공 처리
          - waypoint 주행 단계에서만 사용
        """
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
                    if ignore_nav_failed:
                        rospy.logwarn(
                            "[sequence] navigation failed event received but ignored: %s",
                            self.last_nav_event
                        )
                    else:
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

    def _wait_loading_event(self, target_event, timeout=None):
        if timeout is None:
            timeout = self.event_timeout

        rospy.loginfo("[sequence] waiting loading event: %s", target_event)
        start_time = rospy.Time.now()

        with self.event_cv:
            while not rospy.is_shutdown() and self.running:
                if self.last_loading_event == target_event:
                    rospy.loginfo("[sequence] loading event matched: %s", target_event)
                    return True

                elapsed = (rospy.Time.now() - start_time).to_sec()
                if elapsed > timeout:
                    rospy.logerr("[sequence] timeout waiting loading event: %s", target_event)
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

    def _call_switch_next_map(self):
        rospy.loginfo("[sequence] call switch_next_map")

        try:
            resp = self.switch_next_map_srv()

            if not resp.success:
                rospy.logerr("[sequence] switch_next_map failed to start: %s", resp.message)
                return False

            rospy.loginfo("[sequence] switch_next_map completed: %s", resp.message)
            return True

        except Exception as e:
            rospy.logerr("[sequence] switch_next_map service call error: %s", e)
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

    def _call_loading_finish(self):
        rospy.loginfo("[sequence] call loading_mission finish")

        try:
            resp = self.loading_finish_srv()

            if not resp.success:
                rospy.logerr("[sequence] loading_mission finish failed to start: %s", resp.message)
                return False

            rospy.loginfo("[sequence] loading_mission finish started: %s", resp.message)
            return True

        except Exception as e:
            rospy.logerr("[sequence] loading_mission finish service call error: %s", e)
            return False

    def _drive_forward_for_start(self):
        if not self.start_forward_enabled:
            rospy.loginfo("[sequence] start forward disabled")
            return True

        if self.start_forward_duration <= 0.0:
            rospy.loginfo("[sequence] start forward duration <= 0.0, skip")
            return True

        rospy.loginfo(
            "[sequence] start forward: topic=%s duration=%.2f linear_x=%.3f",
            self.start_forward_cmd_topic,
            self.start_forward_duration,
            self.start_forward_linear_x
        )

        try:
            rate_hz = self.start_forward_rate
            if rate_hz <= 0.0:
                rate_hz = 20.0

            rate = rospy.Rate(rate_hz)
            start_time = rospy.Time.now()

            cmd = Twist()
            cmd.linear.x = self.start_forward_linear_x
            cmd.angular.z = 0.0

            while not rospy.is_shutdown() and self.running:
                elapsed = (rospy.Time.now() - start_time).to_sec()

                if elapsed >= self.start_forward_duration:
                    break

                self.start_forward_pub.publish(cmd)
                rate.sleep()

            stop_cmd = Twist()
            self.start_forward_pub.publish(stop_cmd)
            rospy.sleep(0.1)
            self.start_forward_pub.publish(stop_cmd)

            rospy.loginfo("[sequence] start forward completed")
            return True

        except Exception as e:
            rospy.logerr("[sequence] start forward error: %s", e)

            try:
                self.start_forward_pub.publish(Twist())
            except Exception:
                pass

            return False

    # =====================================================
    # Parallel task
    # =====================================================
    def _run_button_and_map_switch_parallel(self):
        """
        엘리베이터 탑승 완료 후 다음 두 작업을 병렬로 실행한다.

        A. 로봇팔 button 버튼 누르기
           - /arm_mission/button 호출
           - /arm_mission/event 에서 SEXY_BUTTON 확인

        B. 맵 전환
           - /waypoint_navigator/switch_next_map 호출
           - /waypoint_navigator/event 에서 MAP_SWITCHED:B 확인

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
            rospy.loginfo("[sequence] parallel task B start: switch_next_map")

            if not self._call_switch_next_map():
                rospy.logerr("[sequence] parallel task B failed: switch_next_map service failed")
                results["map"] = False
                return

            if not self._wait_nav_event(self.map_switched_event):
                rospy.logerr(
                    "[sequence] parallel task B failed while waiting %s",
                    self.map_switched_event
                )
                results["map"] = False
                return

            results["map"] = True
            rospy.loginfo("[sequence] parallel task B success: map switched")

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

            # auto_start=False에서 NAV_START callback으로 시작된 경우,
            # 이미 last_loading_event에 NAV_START가 들어와 있을 수 있으므로
            # 여기서는 last_loading_event를 초기화하지 않는다.
            if self.auto_start:
                self.last_loading_event = ""

        rospy.loginfo("[sequence] sequence start")

        if not self._connect_services():
            return self._finish_failed("service connection failed")

        # -------------------------------------------------
        # 0. loading_mission NAV_START 대기
        # -------------------------------------------------
        # 방식 B:
        #   auto_start=True  -> NAV_START 기다리지 않고 바로 시작
        #   auto_start=False -> NAV_START를 받아야 시작
        #
        # auto_start=False인 경우에는 _loading_event_cb에서 NAV_START 수신 후
        # run_sequence를 실행시키므로, 이미 NAV_START가 들어온 상태다.
        # 그래도 안전하게 event 값을 확인하고 넘어간다.
        if not self.auto_start:
            self.state = self.ST_WAIT_LOADING_START
            rospy.loginfo("[sequence] auto_start disabled -> require loading event: %s", self.loading_start_event)

            if self.last_loading_event != self.loading_start_event:
                if not self._wait_loading_event(self.loading_start_event):
                    return self._finish_failed("failed while waiting %s" % self.loading_start_event)

        # -------------------------------------------------
        # 0-1. 시작 직진
        # -------------------------------------------------
        # auto_start=True/False와 관계없이 실제 시퀀스 시작 시 3초 직진 수행
        self.state = self.ST_START_FORWARD
        rospy.loginfo("[sequence] STEP 0: start forward before waypoint navigation")

        if not self._drive_forward_for_start():
            return self._finish_failed("start forward failed")

        # -------------------------------------------------
        # 1. waypoint index 0 주행
        # -------------------------------------------------
        self.state = self.ST_NAV_TO_PANEL
        rospy.loginfo("[sequence] STEP 1: waypoint navigation to index %d", self.start_index)

        if not self._call_goto(self.start_index):
            return self._finish_failed("goto service failed")

        nav_reached_event = "NAV_REACHED:%d" % self.start_index

        # waypoint 주행 중에는 NAV_FAILED가 먼저 들어와도 바로 실패 처리하지 않음.
        # 이후 NAV_REACHED가 들어오면 정상적으로 다음 단계 진행.
        if not self._wait_nav_event(nav_reached_event, ignore_nav_failed=True):
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
        rospy.loginfo("[sequence] STEP 5: arm button + switch_next_map parallel")

        if not self._run_button_and_map_switch_parallel():
            return self._finish_failed("button mission or switch_next_map failed")

        # -------------------------------------------------
        # 6. 마커 기반 후진
        # -------------------------------------------------
        self.state = self.ST_MARKER_BACK
        rospy.loginfo("[sequence] STEP 6: marker backward")

        if not self._call_marker_start_3():
            return self._finish_failed("marker_start_3 service failed")

        if not self._wait_nav_event(self.marker_back_done_event):
            return self._finish_failed("failed while waiting %s" % self.marker_back_done_event)

        # -------------------------------------------------
        # 7. 다음 waypoint index 1 주행
        # -------------------------------------------------
        self.state = self.ST_NAV_TO_NEXT
        rospy.loginfo("[sequence] STEP 7: waypoint navigation to next index %d", self.next_index)

        if not self._call_goto(self.next_index):
            return self._finish_failed("next goto service failed")

        next_nav_reached_event = "NAV_REACHED:%d" % self.next_index

        # waypoint 주행 중에는 NAV_FAILED가 먼저 들어와도 바로 실패 처리하지 않음.
        # 이후 NAV_REACHED가 들어오면 정상적으로 다음 단계 진행.
        if not self._wait_nav_event(next_nav_reached_event, ignore_nav_failed=True):
            return self._finish_failed("failed while waiting %s" % next_nav_reached_event)

        # -------------------------------------------------
        # 8. loading_mission finish
        # -------------------------------------------------
        self.state = self.ST_LOADING_FINISH
        rospy.loginfo("[sequence] STEP 8: loading mission finish")

        if not self._call_loading_finish():
            return self._finish_failed("loading_mission finish service failed")

        if not self._wait_loading_event(self.loading_finished_event):
            return self._finish_failed("failed while waiting %s" % self.loading_finished_event)

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

        rospy.loginfo("[sequence] sequence completed successfully up to STEP 8")
        return True

    def _finish_failed(self, reason):
        with self.lock:
            self.state = self.ST_ERROR
            self.running = False
            self.finished = True
            self.failed = True

        try:
            self.start_forward_pub.publish(Twist())
        except Exception:
            pass

        rospy.logerr("[sequence] sequence failed: %s", reason)
        return False

    # =====================================================
    # External services
    # =====================================================
    def _srv_start(self, _req):
        with self.lock:
            if self.running:
                return TriggerResponse(success=False, message="sequence already running")

            # 수동 start는 기존 기능 유지.
            # auto_start=False 상태에서도 이 서비스를 직접 호출하면 NAV_START 없이 시작 가능.
            # 앱/서버 기반 운송 시작은 /loading_mission/event NAV_START를 사용.
            self.finished = False
            self.failed = False

        threading.Thread(target=self.run_sequence, daemon=True).start()
        return TriggerResponse(success=True, message="sequence started")

    def _srv_stop(self, _req):
        with self.lock:
            self.running = False
            self.state = self.ST_IDLE

        try:
            self.start_forward_pub.publish(Twist())
        except Exception:
            pass

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
