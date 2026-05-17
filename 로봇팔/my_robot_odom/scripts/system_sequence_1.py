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

    사용하는 서비스:
      - /waypoint_navigator/goto
      - /waypoint_navigator/marker_start_1
      - /arm_mission/panel

    기다리는 event:
      - /waypoint_navigator/event : NAV_REACHED:0, PANEL_DONE:1, NAV_FAILED:0, STOPPED
      - /arm_mission/event        : SEXY_PANEL
    """

    ST_IDLE = "IDLE"
    ST_NAV_TO_PANEL = "NAV_TO_PANEL"
    ST_PANEL_APPROACH = "PANEL_APPROACH"
    ST_ARM_PANEL = "ARM_PANEL"
    ST_DONE = "DONE"
    ST_ERROR = "ERROR"

    def __init__(self):
        # =====================================================
        # Params
        # =====================================================
        self.start_index = int(rospy.get_param("~start_index", 0))
        self.panel_done_event = rospy.get_param("~panel_done_event", "PANEL_DONE:3")
        self.arm_panel_done_event = rospy.get_param("~arm_panel_done_event", "SEXY_PANEL")

        self.nav_event_topic = rospy.get_param("~nav_event_topic", "/waypoint_navigator/event")
        self.arm_event_topic = rospy.get_param("~arm_event_topic", "/arm_mission/event")

        self.goto_srv_name = rospy.get_param("~goto_srv_name", "/waypoint_navigator/goto")
        self.marker_start_1_srv_name = rospy.get_param("~marker_start_1_srv_name", "/waypoint_navigator/marker_start_1")
        self.arm_panel_srv_name = rospy.get_param("~arm_panel_srv_name", "/arm_mission/panel")

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
        self.arm_panel_srv = None

        rospy.loginfo("[sequence] ready")
        rospy.loginfo("[sequence] nav_event_topic=%s", self.nav_event_topic)
        rospy.loginfo("[sequence] arm_event_topic=%s", self.arm_event_topic)
        rospy.loginfo("[sequence] goto_srv=%s", self.goto_srv_name)
        rospy.loginfo("[sequence] marker_start_1_srv=%s", self.marker_start_1_srv_name)
        rospy.loginfo("[sequence] arm_panel_srv=%s", self.arm_panel_srv_name)

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

        rospy.loginfo("[sequence] sequence completed successfully up to STEP 3")
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
