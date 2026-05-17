#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy
import actionlib
import math
import yaml
import threading
import serial

from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal
from geometry_msgs.msg import PoseWithCovarianceStamped
from actionlib_msgs.msg import GoalStatus
from tf.transformations import quaternion_from_euler

from std_msgs.msg import Bool, Int32, String
from std_srvs.srv import Trigger, TriggerResponse, SetBool, SetBoolResponse, Empty
from my_robot_odom.srv import SetMode, GotoWaypoint, GotoWaypointResponse


UART_PORT = "/dev/ttyACM1"
UART_BAUD = 9600

MODE_MANUAL   = 0
MODE_WAYPOINT = 1
MODE_CAMERA   = 2


class HybridNavigator:
    """
    목표 시나리오
      1) goto(index) 로 waypoint 자율주행
      2) 도착 후 WAIT_PANEL_CMD
      3) marker_start_1 -> /camera_mission_panel = +mission (패널 기반 전진/정렬)
      4) 완료 후 WAIT_MARKER_FWD_CMD
      5) marker_start_2 -> /camera_mission_marker = +mission (마커 기반 전진)
      6) 완료 후 WAIT_MARKER_BACK_CMD
      7) marker_start_3 -> /camera_mission_marker = -mission (마커 기반 후진)
      8) 완료 후 WAIT_MAP_SWITCH_CMD
      9) switch_next_map
     10) 다음 goto(index)

    추가 기능
      - 기존 순차 구조 유지
      - rosservice call /waypoint_navigator/marker_start_1
      - rosservice call /waypoint_navigator/marker_start_2
      - rosservice call /waypoint_navigator/marker_start_3
      - rosservice call /waypoint_navigator/switch_next_map
        를 따로 호출해도 수동 실행 가능

    추가 디버그 기능
      - /vision_debug_mode 토픽으로 현재 보여줄 디버그 화면 모드 전달
      - 값 예시: "off", "panel", "marker"

    추가 costmap clear 기능
      - goto/start/재시도 진입 직전에 /move_base/clear_costmaps 호출 가능
      - 기본 move_base 서비스 특성상 global/local 둘 다 clear 됨

    추가 event 기능
      - /waypoint_navigator/event 토픽으로 완료 이벤트 발행
      - 예:
        NAV_REACHED:0
        NAV_FAILED:0
        PANEL_DONE:1
        MARKER_FWD_DONE:1
        MARKER_BACK_DONE:1
        MAP_SWITCHED:B
        STOPPED
    """

    ST_IDLE                   = "IDLE"
    ST_NAVIGATING             = "NAVIGATING"

    ST_WAIT_PANEL_CMD         = "WAIT_PANEL_CMD"
    ST_PANEL_RUNNING          = "PANEL_RUNNING"

    ST_WAIT_MARKER_FWD_CMD    = "WAIT_MARKER_FWD_CMD"
    ST_MARKER_FWD_RUNNING     = "MARKER_FWD_RUNNING"

    ST_WAIT_MARKER_BACK_CMD   = "WAIT_MARKER_BACK_CMD"
    ST_MARKER_BACK_RUNNING    = "MARKER_BACK_RUNNING"

    ST_WAIT_MAP_SWITCH_CMD    = "WAIT_MAP_SWITCH_CMD"
    ST_MAP_SWITCHING          = "MAP_SWITCHING"
    ST_STOPPED                = "STOPPED"

    def __init__(self):
        # =========================
        # Params
        # =========================
        self.yaml_path = rospy.get_param("~yaml_path", "/config/waypoints.yaml")
        self.map_frame = rospy.get_param("~map_frame", "map")
        self.goal_timeout = float(rospy.get_param("~goal_timeout", 120.0))
        self.retry_max = int(rospy.get_param("~retry_max", 2))
        self.wait_for_amcl = bool(rospy.get_param("~wait_for_amcl", True))

        self.amcl_cov_xy = float(rospy.get_param("~amcl_cov_xy", 0.5))
        self.amcl_cov_yaw = float(rospy.get_param("~amcl_cov_yaw", 0.2))

        self.switch_map_srv = rospy.get_param("~switch_map_srv", "/map_mux/switch_map")
        self.switch_settle_sec = float(rospy.get_param("~switch_settle_sec", 0.7))

        # nav stack 재기동 후 준비 확인용
        self.move_base_wait_timeout = float(rospy.get_param("~move_base_wait_timeout", 20.0))
        self.amcl_wait_timeout_after_switch = float(rospy.get_param("~amcl_wait_timeout_after_switch", 12.0))
        self.amcl_pose_topic = rospy.get_param("~amcl_pose_topic", "/amcl_pose")
        self.move_base_action_name = rospy.get_param("~move_base_action_name", "move_base")

        self.enable_uart = bool(rospy.get_param("~enable_uart", True))
        self.uart_port = rospy.get_param("~uart_port", UART_PORT)
        self.uart_baud = int(rospy.get_param("~uart_baud", UART_BAUD))

        # mission topic 분리
        self.panel_mission_topic = rospy.get_param("~panel_mission_topic", "/camera_mission_panel")
        self.marker_mission_topic = rospy.get_param("~marker_mission_topic", "/camera_mission_marker")

        # done topic 분리
        self.use_panel_mission_done = bool(rospy.get_param("~use_panel_mission_done", True))
        self.use_marker_mission_done = bool(rospy.get_param("~use_marker_mission_done", True))
        self.use_panel_docking_done = bool(rospy.get_param("~use_panel_docking_done", True))
        self.use_marker_docking_done = bool(rospy.get_param("~use_marker_docking_done", True))

        self.panel_mission_done_topic = rospy.get_param("~panel_mission_done_topic", "/panel_mission_done")
        self.marker_mission_done_topic = rospy.get_param("~marker_mission_done_topic", "/marker_mission_done")
        self.panel_docking_done_topic = rospy.get_param("~panel_docking_done_topic", "/panel_docking_done")
        self.marker_docking_done_topic = rospy.get_param("~marker_docking_done_topic", "/marker_docking_done")

        # legacy compatibility
        self.use_legacy_camera_mission_done = bool(rospy.get_param("~use_legacy_camera_mission_done", False))
        self.use_legacy_docking_done = bool(rospy.get_param("~use_legacy_docking_done", False))
        self.legacy_camera_mission_done_topic = rospy.get_param("~legacy_camera_mission_done_topic", "/camera_mission_done")
        self.legacy_docking_done_topic = rospy.get_param("~legacy_docking_done_topic", "/docking_done")

        self.auto_start_after_map_switch = bool(rospy.get_param("~auto_start_after_map_switch", False))
        self.start_with_map_b = bool(rospy.get_param("~start_with_map_b", False))
        self.current_map_is_b = self.start_with_map_b

        self.default_start_index = int(rospy.get_param("~default_start_index", 0))

        # ===== debug mode topic 추가 =====
        self.vision_debug_mode_topic = rospy.get_param("~vision_debug_mode_topic", "/vision_debug_mode")

        # ===== event topic 추가 =====
        self.event_topic = rospy.get_param("~event_topic", "/waypoint_navigator/event")

        # ===== costmap clear params 추가 =====
        self.clear_costmaps_before_goto = bool(rospy.get_param("~clear_costmaps_before_goto", True))
        self.clear_costmaps_srv = rospy.get_param("~clear_costmaps_srv", "/move_base/clear_costmaps")
        self.clear_costmaps_wait = float(rospy.get_param("~clear_costmaps_wait", 0.3))

        # =========================
        # Internal State
        # =========================
        self.state = self.ST_IDLE
        self.running = False
        self.has_amcl = (not self.wait_for_amcl)

        self.wps = self._load_yaml(self.yaml_path)
        self.current_index = self.default_start_index
        self.last_reached_idx = None
        self.active_goal_name = None
        self.active_goal = None

        self.pending_marker_mission = None
        self.last_goal_result_status = None
        self.stop_requested = False

        self.lock = threading.Lock()

        # UART
        self.ser = None
        if self.enable_uart:
            try:
                self.ser = serial.Serial(self.uart_port, self.uart_baud, timeout=1)
                rospy.loginfo("[hybrid_nav] UART connected: %s @ %d", self.uart_port, self.uart_baud)
            except Exception as e:
                self.ser = None
                rospy.logwarn("[hybrid_nav] UART open failed: %s", e)

        # =========================
        # ROS Interfaces
        # =========================
        self.panel_mission_pub = rospy.Publisher(self.panel_mission_topic, Int32, queue_size=1, latch=True)
        self.marker_mission_pub = rospy.Publisher(self.marker_mission_topic, Int32, queue_size=1, latch=True)

        # ===== debug mode publisher 추가 =====
        self.vision_debug_mode_pub = rospy.Publisher(
            self.vision_debug_mode_topic, String, queue_size=1, latch=True
        )

        # ===== event publisher 추가 =====
        self.event_pub = rospy.Publisher(
            self.event_topic, String, queue_size=10, latch=True
        )

        self.ac = actionlib.SimpleActionClient(self.move_base_action_name, MoveBaseAction)
        rospy.loginfo("[hybrid_nav] waiting move_base...")
        self.ac.wait_for_server()
        rospy.loginfo("[hybrid_nav] move_base connected")

        if self.wait_for_amcl:
            rospy.Subscriber(self.amcl_pose_topic, PoseWithCovarianceStamped, self._amcl_cb, queue_size=1)

        self.set_mode = None
        self._ensure_set_mode_service()

        self.switch_map = None
        self._ensure_switch_service()

        # ===== clear_costmaps service proxy 추가 =====
        self.clear_costmaps = None
        self._ensure_clear_costmaps_service()

        # done subscribers
        rospy.Subscriber(self.panel_mission_done_topic, Int32, self._on_panel_mission_done, queue_size=1)
        rospy.Subscriber(self.marker_mission_done_topic, Int32, self._on_marker_mission_done, queue_size=1)
        rospy.Subscriber(self.panel_docking_done_topic, Bool, self._on_panel_docking_done, queue_size=1)
        rospy.Subscriber(self.marker_docking_done_topic, Bool, self._on_marker_docking_done, queue_size=1)

        # legacy compatibility
        if self.use_legacy_camera_mission_done:
            rospy.Subscriber(self.legacy_camera_mission_done_topic, Int32, self._on_legacy_mission_done, queue_size=1)

        if self.use_legacy_docking_done:
            rospy.Subscriber(self.legacy_docking_done_topic, Bool, self._on_legacy_docking_done, queue_size=1)

        # services
        self.srv_goto = rospy.Service("~goto", GotoWaypoint, self._srv_goto)
        self.srv_start = rospy.Service("~start", Trigger, self._srv_start)
        self.srv_stop = rospy.Service("~stop", Trigger, self._srv_stop)

        self.srv_marker_start = rospy.Service("~marker_start", Trigger, self._srv_marker_start_1)  # 호환용
        self.srv_marker_start_1 = rospy.Service("~marker_start_1", Trigger, self._srv_marker_start_1)
        self.srv_marker_start_2 = rospy.Service("~marker_start_2", Trigger, self._srv_marker_start_2)
        self.srv_marker_start_3 = rospy.Service("~marker_start_3", Trigger, self._srv_marker_start_3)

        self.srv_switch_next_map = rospy.Service("~switch_next_map", Trigger, self._srv_switch_next_map)
        self.srv_switch_map_bool = rospy.Service("~switch_map_bool", SetBool, self._srv_switch_map_bool)

        if self.enable_uart and self.ser is not None:
            threading.Thread(target=self._uart_loop, daemon=True).start()

        # 초기값
        self._publish_vision_debug_mode("off")

        rospy.loginfo("[hybrid_nav] ready. state=%s waypoints=%d", self.state, len(self.wps))
        rospy.loginfo("[hybrid_nav] panel_mission_topic=%s", self.panel_mission_topic)
        rospy.loginfo("[hybrid_nav] marker_mission_topic=%s", self.marker_mission_topic)
        rospy.loginfo("[hybrid_nav] vision_debug_mode_topic=%s", self.vision_debug_mode_topic)
        rospy.loginfo("[hybrid_nav] event_topic=%s", self.event_topic)
        rospy.loginfo("[hybrid_nav] clear_costmaps_before_goto=%s", str(self.clear_costmaps_before_goto))
        rospy.loginfo("[hybrid_nav] clear_costmaps_srv=%s", self.clear_costmaps_srv)

    # =========================================================
    # Debug mode helpers
    # =========================================================
    def _publish_vision_debug_mode(self, mode_str):
        try:
            self.vision_debug_mode_pub.publish(String(data=str(mode_str)))
            rospy.loginfo("[hybrid_nav] vision_debug_mode=%s", str(mode_str))
        except Exception as e:
            rospy.logwarn("[hybrid_nav] failed to publish vision_debug_mode: %s", e)

    # =========================================================
    # Event helpers
    # =========================================================
    def _publish_event(self, event_str):
        """
        상위 시퀀스 제어 노드가 기다릴 완료 이벤트를 발행한다.

        예:
          NAV_REACHED:0
          NAV_FAILED:0
          PANEL_DONE:1
          MARKER_FWD_DONE:1
          MARKER_BACK_DONE:1
          MAP_SWITCHED:B
          STOPPED
        """
        try:
            event_str = str(event_str)
            self.event_pub.publish(String(data=event_str))
            rospy.loginfo("[hybrid_nav][EVENT] %s", event_str)
        except Exception as e:
            rospy.logwarn("[hybrid_nav] failed to publish event: %s", e)

    # =========================================================
    # YAML
    # =========================================================
    def _load_yaml(self, path):
        with open(path, "r") as f:
            d = yaml.safe_load(f) or {}
        return d.get("waypoints", []) or []

    def _get_wp(self, index):
        if index < 0 or index >= len(self.wps):
            return None
        return self.wps[index]

    def _wp_name(self, index):
        wp = self._get_wp(index)
        if wp is None:
            return "invalid"
        return wp.get("name", "wp%d" % (index + 1))

    def _wp_mission(self, index):
        wp = self._get_wp(index)
        if wp is None:
            return None
        return int(wp.get("mission", index + 1))

    def _resolve_manual_marker_mission(self):
        """
        수동 marker_start 호출 시 사용할 mission 번호를 결정.
        우선순위:
          1) pending_marker_mission
          2) last_reached_idx의 mission
          3) current_index의 mission
        """
        if self.pending_marker_mission is not None:
            return int(self.pending_marker_mission)

        if self.last_reached_idx is not None:
            mission = self._wp_mission(self.last_reached_idx)
            if mission is not None:
                return int(mission)

        mission = self._wp_mission(self.current_index)
        if mission is not None:
            return int(mission)

        return None

    # =========================================================
    # AMCL
    # =========================================================
    def _amcl_cb(self, msg):
        cov = msg.pose.covariance
        if cov[0] < self.amcl_cov_xy and cov[7] < self.amcl_cov_xy and cov[35] < self.amcl_cov_yaw:
            self.has_amcl = True

    def _wait_amcl_if_needed(self, timeout_sec=10.0):
        if not self.wait_for_amcl:
            return True

        t0 = rospy.Time.now()
        while not rospy.is_shutdown():
            if self.has_amcl:
                rospy.loginfo("[hybrid_nav] AMCL ready")
                return True

            if (rospy.Time.now() - t0).to_sec() > timeout_sec:
                rospy.logwarn("[hybrid_nav] AMCL not stable within %.1fs (continue anyway)", timeout_sec)
                return False

            rospy.loginfo_throttle(1.0, "[hybrid_nav] waiting AMCL initial pose...")
            rospy.sleep(0.1)

        return False

    def _reset_amcl_wait_state(self):
        self.has_amcl = (not self.wait_for_amcl)

    # =========================================================
    # Service Proxy
    # =========================================================
    def _ensure_set_mode_service(self):
        if self.set_mode is not None:
            return True
        try:
            rospy.wait_for_service("/odom_imu/set_mode", timeout=3.0)
            self.set_mode = rospy.ServiceProxy("/odom_imu/set_mode", SetMode)
            rospy.loginfo("[hybrid_nav] /odom_imu/set_mode connected")
            return True
        except Exception as e:
            rospy.logwarn("[hybrid_nav] /odom_imu/set_mode not available: %s", e)
            self.set_mode = None
            return False

    def _ensure_switch_service(self):
        if self.switch_map is not None:
            return True
        try:
            rospy.wait_for_service(self.switch_map_srv, timeout=3.0)
            self.switch_map = rospy.ServiceProxy(self.switch_map_srv, SetBool)
            rospy.loginfo("[hybrid_nav] switch_map connected: %s", self.switch_map_srv)
            return True
        except Exception as e:
            rospy.logwarn("[hybrid_nav] switch_map not available: %s", e)
            self.switch_map = None
            return False

    def _ensure_clear_costmaps_service(self):
        if self.clear_costmaps is not None:
            return True
        try:
            rospy.wait_for_service(self.clear_costmaps_srv, timeout=3.0)
            self.clear_costmaps = rospy.ServiceProxy(self.clear_costmaps_srv, Empty)
            rospy.loginfo("[hybrid_nav] clear_costmaps connected: %s", self.clear_costmaps_srv)
            return True
        except Exception as e:
            rospy.logwarn("[hybrid_nav] clear_costmaps not available: %s", e)
            self.clear_costmaps = None
            return False

    # =========================================================
    # UART
    # =========================================================
    def _uart_write(self, text):
        if self.ser is None:
            return
        try:
            self.ser.write((text + "\n").encode("ascii"))
            rospy.loginfo("[hybrid_nav][UART] sent: %s", text)
        except Exception as e:
            rospy.logwarn("[hybrid_nav][UART] write error: %s", e)

    def _uart_loop(self):
        rospy.loginfo("[hybrid_nav] UART listening on %s", self.uart_port)
        while not rospy.is_shutdown():
            try:
                line = self.ser.readline().decode(errors="ignore").strip()
                if not line:
                    continue

                rospy.loginfo("[hybrid_nav][UART] recv: %s", line)

                if line.startswith("WAYPOINT:"):
                    idx = int(line.split(":")[1]) - 1
                    fake_req = type("req", (), {"index": idx})
                    self._srv_goto(fake_req)

                elif line == "START":
                    self._srv_start(None)
                elif line == "STOP":
                    self._srv_stop(None)
                elif line == "MARKER_START":
                    self._srv_marker_start_1(None)
                elif line == "MARKER_START_1":
                    self._srv_marker_start_1(None)
                elif line == "MARKER_START_2":
                    self._srv_marker_start_2(None)
                elif line == "MARKER_START_3":
                    self._srv_marker_start_3(None)
                elif line == "SWITCH_MAP":
                    self._srv_switch_next_map(None)

            except Exception as e:
                rospy.logwarn("[hybrid_nav][UART] error: %s", e)

    # =========================================================
    # Mode Control
    # =========================================================
    def _request_mode(self, mode_val):
        if not self._ensure_set_mode_service():
            rospy.logwarn("[hybrid_nav] set_mode unavailable; skip")
            return False

        try:
            resp = self.set_mode(mode_val)
            if resp.success:
                rospy.loginfo("[hybrid_nav] set_mode(%d) OK", mode_val)
                return True
            else:
                rospy.logwarn("[hybrid_nav] set_mode(%d) FAIL: %s", mode_val, resp.message)
                return False
        except Exception as e:
            rospy.logwarn("[hybrid_nav] set_mode(%d) error: %s", mode_val, e)
            return False

    # =========================================================
    # move_base / nav stack wait
    # =========================================================
    def _wait_move_base_ready(self, timeout_sec):
        rospy.loginfo("[hybrid_nav] waiting move_base server after map switch...")
        new_ac = actionlib.SimpleActionClient(self.move_base_action_name, MoveBaseAction)
        ok = new_ac.wait_for_server(rospy.Duration(timeout_sec))

        if ok:
            self.ac = new_ac
            rospy.loginfo("[hybrid_nav] move_base reconnected")
            return True

        rospy.logwarn("[hybrid_nav] move_base not ready within %.1fs", timeout_sec)
        return False

    def _wait_nav_stack_ready_after_switch(self):
        mb_ok = self._wait_move_base_ready(self.move_base_wait_timeout)

        self._reset_amcl_wait_state()
        amcl_ok = self._wait_amcl_if_needed(timeout_sec=self.amcl_wait_timeout_after_switch)

        if self.switch_settle_sec > 0.0:
            rospy.sleep(max(0.0, self.switch_settle_sec))

        return mb_ok, amcl_ok

    # =========================================================
    # Costmap Clear
    # =========================================================
    def _clear_costmaps(self):
        if not self.clear_costmaps_before_goto:
            return True

        if not self._ensure_clear_costmaps_service():
            rospy.logwarn("[hybrid_nav] clear_costmaps service unavailable")
            return False

        try:
            self.clear_costmaps()
            rospy.loginfo("[hybrid_nav] clear_costmaps called")
            if self.clear_costmaps_wait > 0.0:
                rospy.sleep(self.clear_costmaps_wait)
            return True
        except Exception as e:
            rospy.logwarn("[hybrid_nav] clear_costmaps call failed: %s", e)
            return False

    # =========================================================
    # Goal Helpers
    # =========================================================
    def _build_goal(self, wp):
        x = float(wp["x"])
        y = float(wp["y"])

        if "yaw" in wp:
            yaw = float(wp["yaw"])
        else:
            yaw = math.radians(float(wp.get("yaw_deg", 0.0)))

        qx, qy, qz, qw = quaternion_from_euler(0.0, 0.0, yaw)

        goal = MoveBaseGoal()
        goal.target_pose.header.frame_id = self.map_frame
        goal.target_pose.header.stamp = rospy.Time.now()
        goal.target_pose.pose.position.x = x
        goal.target_pose.pose.position.y = y
        goal.target_pose.pose.orientation.x = qx
        goal.target_pose.pose.orientation.y = qy
        goal.target_pose.pose.orientation.z = qz
        goal.target_pose.pose.orientation.w = qw
        return goal

    def _send_goal_async(self, index):
        wp = self._get_wp(index)
        if wp is None:
            rospy.logwarn("[hybrid_nav] invalid waypoint index=%d", index)
            return False

        self.current_index = index
        self.active_goal_name = wp.get("name", "wp%d" % (index + 1))
        self.pending_marker_mission = None
        self.last_goal_result_status = None

        goal = self._build_goal(wp)

        rospy.loginfo("[hybrid_nav] send goal idx=%d name=%s x=%.3f y=%.3f",
                      index, self.active_goal_name, float(wp["x"]), float(wp["y"]))

        self.state = self.ST_NAVIGATING
        self.running = True
        self.stop_requested = False

        # 자율주행 중에는 vision debug 끔
        self._publish_vision_debug_mode("off")

        self.ac.send_goal(goal,
                          done_cb=self._goal_done_cb,
                          active_cb=self._goal_active_cb,
                          feedback_cb=self._goal_feedback_cb)
        self.active_goal = goal
        return True

    def _goal_active_cb(self):
        rospy.loginfo("[hybrid_nav] goal active")

    def _goal_feedback_cb(self, _feedback):
        pass

    def _goal_done_cb(self, status, _result):
        with self.lock:
            self.last_goal_result_status = status
            self.active_goal = None

            if self.stop_requested:
                rospy.loginfo("[hybrid_nav] goal done after stop request")
                self.state = self.ST_STOPPED
                self.running = False
                self._publish_vision_debug_mode("off")
                self._publish_event("STOPPED")
                return

            if status == GoalStatus.SUCCEEDED:
                self.last_reached_idx = self.current_index
                mission = self._wp_mission(self.current_index)
                self.pending_marker_mission = mission
                self.state = self.ST_WAIT_PANEL_CMD

                rospy.loginfo("[hybrid_nav] goal success -> WAIT_PANEL_CMD")
                rospy.loginfo("[hybrid_nav] pending marker mission=%s for waypoint=%s",
                              str(mission), self._wp_name(self.current_index))
                self._uart_write("NAV_REACHED:%d" % (self.current_index + 1))

                # ===== event 추가: waypoint 도착 완료 =====
                # ROS service index와 맞추기 위해 0-based index로 발행
                self._publish_event("NAV_REACHED:%d" % self.current_index)

                # 아직 panel 시작 전이므로 off 유지
                self._publish_vision_debug_mode("off")
            else:
                rospy.logwarn("[hybrid_nav] goal failed status=%d", status)
                self.state = self.ST_IDLE
                self.running = False
                self._uart_write("NAV_FAILED:%d" % (self.current_index + 1))

                # ===== event 추가: waypoint 실패 =====
                # ROS service index와 맞추기 위해 0-based index로 발행
                self._publish_event("NAV_FAILED:%d" % self.current_index)

                self._publish_vision_debug_mode("off")

    def _cancel_goal(self):
        try:
            self.ac.cancel_goal()
        except Exception:
            pass
        self.active_goal = None

    # =========================================================
    # Navigation Thread
    # =========================================================
    def _run_nav_with_retry(self, index):
        self.has_amcl = (not self.wait_for_amcl)
        self._wait_amcl_if_needed(timeout_sec=10.0)
        self._request_mode(MODE_WAYPOINT)
        self._publish_vision_debug_mode("off")

        # ===== goto/start/재시도 진입 전에 costmap clear =====
        if self.clear_costmaps_before_goto:
            self._clear_costmaps()

        attempt = 0
        while not rospy.is_shutdown() and not self.stop_requested:
            self._send_goal_async(index)

            t0 = rospy.Time.now()
            while not rospy.is_shutdown() and not self.stop_requested:
                if self.state != self.ST_NAVIGATING:
                    break

                if (rospy.Time.now() - t0).to_sec() > self.goal_timeout:
                    rospy.logwarn("[hybrid_nav] goal timeout -> cancel")
                    self._cancel_goal()
                    self.last_goal_result_status = GoalStatus.ABORTED
                    self.state = self.ST_IDLE

                    # ===== event 추가: timeout도 실패 이벤트로 발행 =====
                    self._publish_event("NAV_FAILED:%d" % index)

                    self._publish_vision_debug_mode("off")
                    break

                rospy.sleep(0.1)

            if self.stop_requested:
                rospy.loginfo("[hybrid_nav] navigation interrupted by stop")
                self._publish_vision_debug_mode("off")
                self._publish_event("STOPPED")
                return

            if self.last_goal_result_status == GoalStatus.SUCCEEDED:
                return

            attempt += 1
            if attempt > self.retry_max:
                rospy.logwarn("[hybrid_nav] retry exceeded for idx=%d", index)

                # ===== event 추가: 재시도 초과 실패 =====
                self._publish_event("NAV_FAILED:%d" % index)

                self._publish_vision_debug_mode("off")
                return

            rospy.logwarn("[hybrid_nav] retry %d/%d for idx=%d", attempt, self.retry_max, index)

            # ===== 재시도 직전에도 한 번 더 clear =====
            if self.clear_costmaps_before_goto:
                self._clear_costmaps()

            rospy.sleep(0.3)

    # =========================================================
    # Mission publish helpers
    # =========================================================
    def _publish_panel_idle(self):
        self.panel_mission_pub.publish(Int32(data=0))
        rospy.loginfo("[hybrid_nav] publish panel idle: 0")

    def _publish_marker_idle(self):
        self.marker_mission_pub.publish(Int32(data=0))
        rospy.loginfo("[hybrid_nav] publish marker idle: 0")

    def _publish_panel_mission(self, mission):
        self.panel_mission_pub.publish(Int32(data=int(mission)))
        rospy.loginfo("[hybrid_nav] publish panel mission: %d", int(mission))

    def _publish_marker_mission(self, mission):
        self.marker_mission_pub.publish(Int32(data=int(mission)))
        rospy.loginfo("[hybrid_nav] publish marker mission: %d", int(mission))

    # =========================================================
    # Panel / Marker / Map Switch
    # =========================================================
    def _start_panel_mission(self):
        if self.state in [self.ST_PANEL_RUNNING, self.ST_MARKER_FWD_RUNNING,
                          self.ST_MARKER_BACK_RUNNING, self.ST_MAP_SWITCHING]:
            return False, "panel/marker or map switching already running"

        if self.state == self.ST_WAIT_PANEL_CMD:
            if self.pending_marker_mission is None:
                return False, "no pending panel mission"

            mission = int(self.pending_marker_mission)

            self._request_mode(MODE_CAMERA)
            self._publish_marker_idle()
            self._publish_panel_mission(mission)
            self.state = self.ST_PANEL_RUNNING

            # panel 디버그 화면 on
            self._publish_vision_debug_mode("panel")

            rospy.loginfo("[hybrid_nav] marker_start_1 -> panel mission=%d (panel forward)", mission)
            self._uart_write("PANEL_START:%d" % mission)
            return True, "marker_start_1(panel) started: %d" % mission

        mission = self._resolve_manual_marker_mission()
        if mission is None:
            return False, "cannot resolve panel mission for manual start"

        self.pending_marker_mission = int(mission)
        self._request_mode(MODE_CAMERA)
        self._publish_marker_idle()
        self._publish_panel_mission(mission)
        self.state = self.ST_PANEL_RUNNING

        # panel 디버그 화면 on
        self._publish_vision_debug_mode("panel")

        rospy.loginfo("[hybrid_nav] marker_start_1 manual -> panel mission=%d (panel forward)", mission)
        self._uart_write("PANEL_START:%d" % mission)
        return True, "marker_start_1(panel) manual started: %d" % mission

    def _start_marker_mission_fwd(self):
        if self.state in [self.ST_PANEL_RUNNING, self.ST_MARKER_FWD_RUNNING,
                          self.ST_MARKER_BACK_RUNNING, self.ST_MAP_SWITCHING]:
            return False, "panel/marker or map switching already running"

        if self.state == self.ST_WAIT_MARKER_FWD_CMD:
            if self.pending_marker_mission is None:
                return False, "no pending marker mission"

            mission = int(self.pending_marker_mission)

            self._request_mode(MODE_CAMERA)
            self._publish_panel_idle()
            self._publish_marker_mission(mission)
            self.state = self.ST_MARKER_FWD_RUNNING

            # marker 디버그 화면 on
            self._publish_vision_debug_mode("marker")

            rospy.loginfo("[hybrid_nav] marker_start_2 -> mission=%d (marker forward)", mission)
            self._uart_write("MARKER_START_2:%d" % mission)
            return True, "marker_start_2 started: %d" % mission

        mission = self._resolve_manual_marker_mission()
        if mission is None:
            return False, "cannot resolve marker mission for manual start"

        self.pending_marker_mission = int(mission)
        self._request_mode(MODE_CAMERA)
        self._publish_panel_idle()
        self._publish_marker_mission(mission)
        self.state = self.ST_MARKER_FWD_RUNNING

        # marker 디버그 화면 on
        self._publish_vision_debug_mode("marker")

        rospy.loginfo("[hybrid_nav] marker_start_2 manual -> mission=%d (marker forward)", mission)
        self._uart_write("MARKER_START_2:%d" % mission)
        return True, "marker_start_2 manual started: %d" % mission

    def _start_marker_mission_back(self):
        if self.state in [self.ST_PANEL_RUNNING, self.ST_MARKER_FWD_RUNNING,
                          self.ST_MARKER_BACK_RUNNING, self.ST_MAP_SWITCHING]:
            return False, "panel/marker or map switching already running"

        if self.state == self.ST_WAIT_MARKER_BACK_CMD:
            if self.pending_marker_mission is None:
                return False, "no pending marker mission"

            mission = -int(self.pending_marker_mission)

            self._request_mode(MODE_CAMERA)
            self._publish_panel_idle()
            self._publish_marker_mission(mission)
            self.state = self.ST_MARKER_BACK_RUNNING

            # marker 디버그 화면 on
            self._publish_vision_debug_mode("marker")

            rospy.loginfo("[hybrid_nav] marker_start_3 -> mission=%d (marker backward)", mission)
            self._uart_write("MARKER_START_3:%d" % abs(mission))
            return True, "marker_start_3 started: %d" % abs(mission)

        base_mission = self._resolve_manual_marker_mission()
        if base_mission is None:
            return False, "cannot resolve marker mission for manual start"

        self.pending_marker_mission = int(base_mission)
        mission = -int(base_mission)

        self._request_mode(MODE_CAMERA)
        self._publish_panel_idle()
        self._publish_marker_mission(mission)
        self.state = self.ST_MARKER_BACK_RUNNING

        # marker 디버그 화면 on
        self._publish_vision_debug_mode("marker")

        rospy.loginfo("[hybrid_nav] marker_start_3 manual -> mission=%d (marker backward)", mission)
        self._uart_write("MARKER_START_3:%d" % abs(mission))
        return True, "marker_start_3 manual started: %d" % abs(mission)

    def _handle_panel_finish(self, mission_value=None, source="unknown"):
        with self.lock:
            if self.state != self.ST_PANEL_RUNNING:
                return

            rospy.loginfo("[hybrid_nav] panel finished from %s mission=%s",
                          source, str(mission_value))
            self._publish_panel_idle()
            self.state = self.ST_WAIT_MARKER_FWD_CMD
            self._request_mode(MODE_WAYPOINT)

            # ===== event 추가: 패널 기반 주행 완료 =====
            if mission_value is not None:
                self._publish_event("PANEL_DONE:%d" % int(mission_value))
            elif self.pending_marker_mission is not None:
                self._publish_event("PANEL_DONE:%d" % int(self.pending_marker_mission))
            else:
                self._publish_event("PANEL_DONE")

            # panel 끝났으니 일단 off
            self._publish_vision_debug_mode("off")

    def _handle_marker_finish(self, mission_value=None, source="unknown"):
        with self.lock:
            if self.state == self.ST_MARKER_FWD_RUNNING:
                rospy.loginfo("[hybrid_nav] marker forward finished from %s mission=%s",
                              source, str(mission_value))
                self._publish_marker_idle()
                self.state = self.ST_WAIT_MARKER_BACK_CMD
                self._request_mode(MODE_WAYPOINT)

                # ===== event 추가: 마커 기반 전진/도킹 완료 =====
                if mission_value is not None:
                    self._publish_event("MARKER_FWD_DONE:%d" % int(mission_value))
                elif self.pending_marker_mission is not None:
                    self._publish_event("MARKER_FWD_DONE:%d" % int(self.pending_marker_mission))
                else:
                    self._publish_event("MARKER_FWD_DONE")

                # marker fwd 끝났으니 일단 off
                self._publish_vision_debug_mode("off")
                return

            if self.state == self.ST_MARKER_BACK_RUNNING:
                rospy.loginfo("[hybrid_nav] marker backward finished from %s mission=%s",
                              source, str(mission_value))

                self._publish_marker_idle()

                if mission_value is not None:
                    self._uart_write("R:%d" % int(mission_value))

                if self.last_reached_idx is not None:
                    self._uart_write("REACHED:%d" % self.last_reached_idx)

                self.state = self.ST_WAIT_MAP_SWITCH_CMD
                self._request_mode(MODE_WAYPOINT)

                # ===== event 추가: 마커 기반 후진 완료 =====
                # 후진 완료는 mission_value가 음수로 들어올 수 있으므로 abs 처리
                if mission_value is not None:
                    self._publish_event("MARKER_BACK_DONE:%d" % abs(int(mission_value)))
                elif self.pending_marker_mission is not None:
                    self._publish_event("MARKER_BACK_DONE:%d" % abs(int(self.pending_marker_mission)))
                else:
                    self._publish_event("MARKER_BACK_DONE")

                # marker back 끝났으니 off
                self._publish_vision_debug_mode("off")
                return

    def _switch_map_internal(self, target_is_b):
        if not self._ensure_switch_service():
            return False, "switch_map service unavailable"

        self.state = self.ST_MAP_SWITCHING

        # 맵 전환 중에는 off
        self._publish_vision_debug_mode("off")

        try:
            resp = self.switch_map(bool(target_is_b))
            if not resp.success:
                self.state = self.ST_IDLE
                self._publish_vision_debug_mode("off")
                return False, resp.message

            self.current_map_is_b = bool(target_is_b)
            rospy.loginfo("[hybrid_nav] switch_map(%s) ok: %s",
                          "B" if target_is_b else "A", resp.message)

            mb_ok, amcl_ok = self._wait_nav_stack_ready_after_switch()

            self.state = self.ST_IDLE
            self._uart_write("MAP_SWITCHED:%s" % ("B" if target_is_b else "A"))

            # ===== event 추가: 맵 전환 완료 =====
            self._publish_event("MAP_SWITCHED:%s" % ("B" if target_is_b else "A"))

            rospy.loginfo("[hybrid_nav] map switch ready: move_base_ok=%s amcl_ok=%s",
                          str(mb_ok), str(amcl_ok))

            self._publish_vision_debug_mode("off")

            if self.auto_start_after_map_switch:
                next_index = self.current_index + 1
                if next_index < len(self.wps):
                    rospy.loginfo("[hybrid_nav] auto start next waypoint idx=%d", next_index)
                    threading.Thread(target=self._run_nav_with_retry, args=(next_index,), daemon=True).start()

            return True, "{} | move_base_ok={} amcl_ok={}".format(resp.message, mb_ok, amcl_ok)

        except Exception as e:
            self.state = self.ST_IDLE
            self._publish_vision_debug_mode("off")
            return False, str(e)

    # =========================================================
    # Done Callbacks
    # =========================================================
    def _on_panel_mission_done(self, msg):
        if not self.use_panel_mission_done:
            return

        mission = int(msg.data)
        rospy.loginfo("[hybrid_nav] panel mission done = %d", mission)
        self._handle_panel_finish(mission_value=mission, source="panel_mission_done")

    def _on_marker_mission_done(self, msg):
        if not self.use_marker_mission_done:
            return

        mission = int(msg.data)
        rospy.loginfo("[hybrid_nav] marker mission done = %d", mission)
        self._handle_marker_finish(mission_value=mission, source="marker_mission_done")

    def _on_panel_docking_done(self, msg):
        if not self.use_panel_docking_done:
            return

        if not msg.data:
            return

        mission = self.pending_marker_mission if self.pending_marker_mission is not None else -1
        rospy.loginfo("[hybrid_nav] panel docking done = True")
        self._handle_panel_finish(mission_value=mission, source="panel_docking_done")

    def _on_marker_docking_done(self, msg):
        if not self.use_marker_docking_done:
            return

        if not msg.data:
            return

        mission = self.pending_marker_mission if self.pending_marker_mission is not None else -1
        rospy.loginfo("[hybrid_nav] marker docking done = True")
        self._handle_marker_finish(mission_value=mission, source="marker_docking_done")

    # legacy compatibility callbacks
    def _on_legacy_mission_done(self, msg):
        mission = int(msg.data)
        rospy.loginfo("[hybrid_nav] legacy /camera_mission_done = %d", mission)

        if self.state == self.ST_PANEL_RUNNING:
            self._handle_panel_finish(mission_value=mission, source="legacy_camera_mission_done")
        elif self.state in [self.ST_MARKER_FWD_RUNNING, self.ST_MARKER_BACK_RUNNING]:
            self._handle_marker_finish(mission_value=mission, source="legacy_camera_mission_done")

    def _on_legacy_docking_done(self, msg):
        if not msg.data:
            return

        mission = self.pending_marker_mission if self.pending_marker_mission is not None else -1
        rospy.loginfo("[hybrid_nav] legacy /docking_done = True")

        if self.state == self.ST_PANEL_RUNNING:
            self._handle_panel_finish(mission_value=mission, source="legacy_docking_done")
        elif self.state in [self.ST_MARKER_FWD_RUNNING, self.ST_MARKER_BACK_RUNNING]:
            self._handle_marker_finish(mission_value=mission, source="legacy_docking_done")

    # =========================================================
    # Services
    # =========================================================
    def _srv_goto(self, req):
        index = int(req.index)
        if index < 0 or index >= len(self.wps):
            return GotoWaypointResponse(success=False, message="Invalid index")

        with self.lock:
            self.stop_requested = False
            self.running = True
            self.pending_marker_mission = None

        if self.active_goal is not None:
            rospy.loginfo("[hybrid_nav] cancel current goal before goto")
            self._cancel_goal()

        # goto 시작 시 off
        self._publish_vision_debug_mode("off")

        threading.Thread(target=self._run_nav_with_retry, args=(index,), daemon=True).start()
        return GotoWaypointResponse(success=True, message="Sent waypoint #%d" % index)

    def _srv_start(self, _req):
        idx = self.default_start_index

        with self.lock:
            if self.state == self.ST_NAVIGATING:
                return TriggerResponse(success=False, message="already navigating")

        # start도 자율주행 시작이므로 off
        self._publish_vision_debug_mode("off")

        threading.Thread(target=self._run_nav_with_retry, args=(idx,), daemon=True).start()
        return TriggerResponse(success=True,
                               message="started navigation to idx=%d name=%s" % (idx, self._wp_name(idx)))

    def _srv_stop(self, _req):
        with self.lock:
            self.stop_requested = True
            self.running = False
            self.state = self.ST_STOPPED

        self._cancel_goal()
        self._publish_panel_idle()
        self._publish_marker_idle()
        self._request_mode(MODE_WAYPOINT)

        # stop 시 off
        self._publish_vision_debug_mode("off")

        self._uart_write("STOPPED")

        # ===== event 추가: 정지 완료 =====
        self._publish_event("STOPPED")

        return TriggerResponse(success=True, message="stopped")

    def _srv_marker_start_1(self, _req):
        with self.lock:
            ok, msg = self._start_panel_mission()
        return TriggerResponse(success=ok, message=msg)

    def _srv_marker_start_2(self, _req):
        with self.lock:
            ok, msg = self._start_marker_mission_fwd()
        return TriggerResponse(success=ok, message=msg)

    def _srv_marker_start_3(self, _req):
        with self.lock:
            ok, msg = self._start_marker_mission_back()
        return TriggerResponse(success=ok, message=msg)

    def _srv_switch_next_map(self, _req):
        with self.lock:
            target_is_b = (not self.current_map_is_b)

        ok, msg = self._switch_map_internal(target_is_b)
        return TriggerResponse(success=ok, message=msg)

    def _srv_switch_map_bool(self, req):
        ok, msg = self._switch_map_internal(bool(req.data))
        return SetBoolResponse(success=ok, message=msg)


if __name__ == "__main__":
    rospy.init_node("waypoint_navigator_hybrid")
    try:
        HybridNavigator()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
