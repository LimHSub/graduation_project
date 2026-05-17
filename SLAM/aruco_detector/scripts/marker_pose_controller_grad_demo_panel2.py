#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
import time

import cv2
import numpy as np
import rospy
from cv_bridge import CvBridge
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Image, CameraInfo
from std_msgs.msg import Bool, Int16MultiArray, Int32, String
from ultralytics import YOLO


class PanelPoseController:
    """
    기존 marker_pose_controller.py 구조를 최대한 유지하면서,
    ArUco 대신 YOLO + Depth 카메라 기반으로 left_panel 객체를 검출하여
    중앙 정렬 후 30cm까지 접근하는 컨트롤러.

    유지 기능:
      - /cmd_vel_doc publish
      - /doc_cmd_char publish
      - /doc_pwm_cmd publish
      - /docking_done, /camera_mission_done publish
      - /camera_mission 수신 시 동작 시작
      - timer 기반 제어

    변경 기능:
      - YOLO(pt)로 left_panel 검출
      - depth + camera_info 로 패널 중심의 x, y, z 계산
      - 패널 중심이 화면 중앙에 오도록 정렬
      - z <= 0.30m 이면 정지 후 완료
      - 화면에 x, y, z, yaw 오버레이 표시

    좌표계:
      - x : 카메라 기준 좌/우(m), 우측 +
      - y : 카메라 기준 상/하(m), 아래 +
      - z : 카메라 기준 전방 거리(m)
      - yaw : 패널 중심이 카메라 정면으로부터 벗어난 방향각(deg)
              = atan2(x, z)

    주의:
      - 실제 물체 자세(yaw orientation) 추정이 아니라 중심 방향각임.
      - aligned depth 이미지와 camera_info 필요.
    """

    MODE_IDLE = 0
    MODE_APPROACH = 1
    MODE_RETREAT = 2  # 기존 인터페이스 유지용. 이번 구현은 APPROACH 중심.

    def __init__(self):
        rospy.init_node('panel_pose_controller')

        # =========================
        # Topics / Mission Interface
        # =========================
        self.cmd_doc_topic = rospy.get_param("~cmd_doc_topic", "/cmd_vel_doc")
        self.char_cmd_topic = rospy.get_param("~char_cmd_topic", "/doc_cmd_char")
        self.doc_pwm_topic = rospy.get_param("~doc_pwm_topic", "/doc_pwm_cmd")
        self.topic_mission = rospy.get_param("~topic_mission", "/camera_mission")

        # =========================
        # Camera / YOLO Params
        # =========================
        self.rgb_topic = rospy.get_param("~rgb_topic", "/camera/color/image_raw")
        self.depth_topic = rospy.get_param("~depth_topic", "/camera/aligned_depth_to_color/image_raw")
        self.camera_info_topic = rospy.get_param("~camera_info_topic", "/camera/color/camera_info")
        self.debug_image_topic = rospy.get_param("~debug_image_topic", "~debug_image")

        self.model_path = rospy.get_param("~model_path", "/home/inwoong/panel_best.pt")
        self.target_class_name = rospy.get_param("~target_class_name", "left_panel")
        self.conf_threshold = float(rospy.get_param("~conf_threshold", 0.55))
        self.input_size = int(rospy.get_param("~input_size", 640))

        # =========================
        # Control Params
        # =========================
        self.control_hz = float(rospy.get_param("~control_hz", 10.0))
        self.target_stop_z = float(rospy.get_param("~target_stop_z", 0.30))
        self.align_x_threshold = float(rospy.get_param("~align_x_threshold", 0.03))
        self.marker_timeout_sec = float(rospy.get_param("~marker_timeout_sec", 0.7))
        self.stop_burst_count = int(rospy.get_param("~stop_burst_count", 3))
        self.left_if_pos = rospy.get_param("~left_if_pos", True)

        # Twist fallback parameters
        self.forward_speed = float(rospy.get_param("~forward_speed", 0.10))
        self.backward_speed = float(rospy.get_param("~backward_speed", 0.08))
        self.turn_speed = float(rospy.get_param("~turn_speed", 0.8))

        # PWM parameters
        self.use_direct_pwm = bool(rospy.get_param("~use_direct_pwm", True))
        self.pwm_forward_l = int(rospy.get_param("~pwm_forward_l", 30))
        self.pwm_forward_r = int(rospy.get_param("~pwm_forward_r", 30))
        self.pwm_turn_left_l = int(rospy.get_param("~pwm_turn_left_l", 0))
        self.pwm_turn_left_r = int(rospy.get_param("~pwm_turn_left_r", 10))
        self.pwm_turn_right_l = int(rospy.get_param("~pwm_turn_right_l", 10))
        self.pwm_turn_right_r = int(rospy.get_param("~pwm_turn_right_r", 0))
        self.pwm_backward_l = int(rospy.get_param("~pwm_backward_l", -30))
        self.pwm_backward_r = int(rospy.get_param("~pwm_backward_r", -30))

        # depth sampling parameters
        self.depth_window = int(rospy.get_param("~depth_window", 5))
        self.max_valid_depth_m = float(rospy.get_param("~max_valid_depth_m", 5.0))
        self.min_valid_depth_m = float(rospy.get_param("~min_valid_depth_m", 0.05))

        # =========================
        # ROS Interfaces
        # =========================
        self.cmd_pub = rospy.Publisher(self.cmd_doc_topic, Twist, queue_size=10)
        self.char_pub = rospy.Publisher(self.char_cmd_topic, String, queue_size=10)
        self.pwm_pub = rospy.Publisher(self.doc_pwm_topic, Int16MultiArray, queue_size=10)
        self.done_pub = rospy.Publisher("/docking_done", Bool, queue_size=1)
        self.mdone_pub = rospy.Publisher("/camera_mission_done", Int32, queue_size=1, latch=True)
        self.mode_pub = rospy.Publisher("~mode", String, queue_size=1, latch=True)
        self.debug_pub = rospy.Publisher(self.debug_image_topic, Image, queue_size=1)

        rospy.Subscriber(self.topic_mission, Int32, self.mission_callback, queue_size=1)
        rospy.Subscriber(self.rgb_topic, Image, self.rgb_callback, queue_size=1, buff_size=2 ** 24)
        rospy.Subscriber(self.depth_topic, Image, self.depth_callback, queue_size=1, buff_size=2 ** 24)
        rospy.Subscriber(self.camera_info_topic, CameraInfo, self.camera_info_callback, queue_size=1)

        # =========================
        # Internal States
        # =========================
        self.bridge = CvBridge()
        self.model = YOLO(self.model_path)
        self.class_names = self.model.names
        self.target_class_id = self._resolve_target_class_id(self.target_class_name)

        self.rgb_image = None
        self.depth_image = None
        self.depth_encoding = None
        self.fx = None
        self.fy = None
        self.cx = None
        self.cy = None

        self.last_rgb_time = 0.0
        self.last_depth_time = 0.0
        self.last_cam_info_time = 0.0

        self.current_pose_x = 0.0
        self.current_pose_y = 0.0
        self.current_distance = float('inf')
        self.current_yaw = 0.0
        self.current_conf = 0.0
        self.current_bbox = None
        self.target_visible = False
        self.last_target_time = 0.0

        self.mode = self.MODE_IDLE
        self.expected_marker_id = None
        self.last_completed_marker_id = None

        rospy.Timer(rospy.Duration(1.0 / self.control_hz), self._timer_cb)

        rospy.loginfo(
            "[panel_pose_controller] ready | model=%s target=%s(class_id=%s) stop_z=%.3f align_x=%.3f use_direct_pwm=%s",
            self.model_path,
            self.target_class_name,
            str(self.target_class_id),
            self.target_stop_z,
            self.align_x_threshold,
            str(self.use_direct_pwm)
        )
        rospy.spin()

    # =========================================================
    # ROS Callbacks
    # =========================================================
    def mission_callback(self, msg):
        val = int(msg.data)

        if val == 0:
            self.mode = self.MODE_IDLE
            self.expected_marker_id = None
            self.publish_stop("mission=0 -> idle")
            self.mode_pub.publish("IDLE")
            return

        # 기존 인터페이스 유지
        if val > 0:
            self.mode = self.MODE_APPROACH
            self.expected_marker_id = val
            self.mode_pub.publish("APPROACH")
            rospy.loginfo("[panel_ctrl] APPROACH start mission=%d", self.expected_marker_id)
        else:
            self.mode = self.MODE_RETREAT
            self.expected_marker_id = abs(val)
            self.mode_pub.publish("RETREAT")
            rospy.loginfo("[panel_ctrl] RETREAT start mission=%d", self.expected_marker_id)

        self.publish_stop("new mission start")

    def rgb_callback(self, msg):
        try:
            self.rgb_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            self.last_rgb_time = time.time()
            self.process_detection()
        except Exception as e:
            rospy.logwarn_throttle(1.0, "[panel_ctrl] rgb callback error: %s", str(e))

    def depth_callback(self, msg):
        try:
            if msg.encoding in ['16UC1', 'mono16']:
                self.depth_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
                self.depth_encoding = '16UC1'
            else:
                self.depth_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
                self.depth_encoding = msg.encoding
            self.last_depth_time = time.time()
        except Exception as e:
            rospy.logwarn_throttle(1.0, "[panel_ctrl] depth callback error: %s", str(e))

    def camera_info_callback(self, msg):
        self.fx = msg.K[0]
        self.fy = msg.K[4]
        self.cx = msg.K[2]
        self.cy = msg.K[5]
        self.last_cam_info_time = time.time()

    def _timer_cb(self, _event):
        self.run_control()

    # =========================================================
    # Vision / Geometry
    # =========================================================
    def _resolve_target_class_id(self, target_name):
        for cid, cname in self.class_names.items():
            if str(cname) == str(target_name):
                return int(cid)
        raise RuntimeError("target class '%s' not found in model classes=%s" % (target_name, self.class_names))

    def process_detection(self):
        if self.rgb_image is None:
            return

        frame = self.rgb_image.copy()
        results = self.model.predict(source=frame, imgsz=self.input_size, conf=self.conf_threshold, verbose=False)
        if not results:
            self.target_visible = False
            self._publish_debug(frame, None, None)
            return

        result = results[0]
        best = None
        best_area = -1.0

        if result.boxes is not None:
            for box in result.boxes:
                cls_id = int(box.cls[0].item())
                conf = float(box.conf[0].item())
                if cls_id != self.target_class_id:
                    continue

                xyxy = box.xyxy[0].cpu().numpy().astype(int)
                x1, y1, x2, y2 = xyxy.tolist()
                area = max(0, x2 - x1) * max(0, y2 - y1)
                if area > best_area:
                    best_area = area
                    best = (x1, y1, x2, y2, conf)

        if best is None:
            self.target_visible = False
            self._publish_debug(frame, None, None)
            return

        x1, y1, x2, y2, conf = best
        u = int((x1 + x2) * 0.5)
        v = int((y1 + y2) * 0.5)
        xyz = self._get_xyz_from_depth(u, v)

        self.current_bbox = (x1, y1, x2, y2)
        self.current_conf = conf

        if xyz is None:
            self.target_visible = False
            self._publish_debug(frame, (x1, y1, x2, y2, conf), None)
            return

        x, y, z = xyz
        yaw_deg = math.degrees(math.atan2(x, z))

        self.current_pose_x = x
        self.current_pose_y = y
        self.current_distance = z
        self.current_yaw = yaw_deg
        self.target_visible = True
        self.last_target_time = time.time()

        self._publish_debug(frame, (x1, y1, x2, y2, conf), (x, y, z, yaw_deg, u, v))

    def _get_xyz_from_depth(self, u, v):
        if self.depth_image is None or self.fx is None:
            return None

        h, w = self.depth_image.shape[:2]
        if u < 0 or u >= w or v < 0 or v >= h:
            return None

        half = max(1, self.depth_window // 2)
        x0 = max(0, u - half)
        x1 = min(w, u + half + 1)
        y0 = max(0, v - half)
        y1 = min(h, v + half + 1)
        patch = self.depth_image[y0:y1, x0:x1]

        if patch.size == 0:
            return None

        if self.depth_encoding == '16UC1':
            patch_m = patch.astype(np.float32) * 0.001
        else:
            patch_m = patch.astype(np.float32)

        valid = patch_m[np.isfinite(patch_m)]
        valid = valid[(valid > self.min_valid_depth_m) & (valid < self.max_valid_depth_m)]
        if valid.size == 0:
            return None

        z = float(np.median(valid))
        x = (float(u) - self.cx) * z / self.fx
        y = (float(v) - self.cy) * z / self.fy
        return x, y, z

    def _publish_debug(self, frame, bbox, xyz_info):
        h, w = frame.shape[:2]
        center_u = w // 2
        center_v = h // 2
        cv2.line(frame, (center_u, 0), (center_u, h), (255, 255, 0), 1)
        cv2.line(frame, (0, center_v), (w, center_v), (255, 255, 0), 1)

        if bbox is not None:
            x1, y1, x2, y2, conf = bbox
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame,
                        "%s %.2f" % (self.target_class_name, conf),
                        (x1, max(20, y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            u = int((x1 + x2) * 0.5)
            v = int((y1 + y2) * 0.5)
            cv2.circle(frame, (u, v), 5, (0, 0, 255), -1)

        if xyz_info is not None:
            x, y, z, yaw_deg, u, v = xyz_info
            info_lines = [
                "x: %.3f m" % x,
                "y: %.3f m" % y,
                "z: %.3f m" % z,
                "yaw: %.2f deg" % yaw_deg,
            ]
            for i, text in enumerate(info_lines):
                cv2.putText(frame, text, (20, 35 + i * 28),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 255), 2)
            cv2.circle(frame, (u, v), 6, (255, 0, 255), -1)

        try:
            self.debug_pub.publish(self.bridge.cv2_to_imgmsg(frame, encoding='bgr8'))
        except Exception as e:
            rospy.logwarn_throttle(1.0, "[panel_ctrl] debug publish error: %s", str(e))

    # =========================================================
    # Utils
    # =========================================================
    def sensor_ready_recently(self):
        now = time.time()
        return ((now - self.last_rgb_time) < self.marker_timeout_sec and
                (now - self.last_depth_time) < self.marker_timeout_sec and
                (now - self.last_cam_info_time) < 3.0)

    def target_visible_recently(self):
        now = time.time()
        return self.target_visible and ((now - self.last_target_time) < self.marker_timeout_sec)

    def publish_twist(self, lin, ang, reason=""):
        twist = Twist()
        twist.linear.x = lin
        twist.angular.z = ang
        self.cmd_pub.publish(twist)
        rospy.loginfo("[panel_ctrl] cmd_vel lin=%.3f ang=%.3f | %s", lin, ang, reason)

    def publish_pwm(self, left, right, reason=""):
        msg = Int16MultiArray()
        msg.data = [int(left), int(right)]
        self.pwm_pub.publish(msg)
        rospy.loginfo("[panel_ctrl] pwm [%d, %d] | %s", int(left), int(right), reason)

    def publish_stop(self, reason="stop"):
        twist = Twist()
        stop_pwm = Int16MultiArray()
        stop_pwm.data = [0, 0]

        for _ in range(self.stop_burst_count):
            self.cmd_pub.publish(twist)
            self.pwm_pub.publish(stop_pwm)

        self.char_pub.publish(String(data='x'))
        rospy.loginfo("[panel_ctrl] STOP | %s", reason)

    def finish_mission(self, reason="done"):
        mission_id = self.expected_marker_id if self.expected_marker_id is not None else -1

        self.publish_stop(reason)
        self.done_pub.publish(Bool(data=True))
        if mission_id > 0:
            self.mdone_pub.publish(Int32(data=mission_id))

        self.last_completed_marker_id = mission_id
        rospy.loginfo("[panel_ctrl] mission finished id=%s | %s", str(mission_id), reason)

        self.mode = self.MODE_IDLE
        self.expected_marker_id = None
        self.mode_pub.publish("IDLE")

    def _turn_ang(self, err_positive, turn_speed=None):
        ts = self.turn_speed if turn_speed is None else float(turn_speed)
        left = (err_positive and self.left_if_pos) or ((not err_positive) and (not self.left_if_pos))
        return +ts if left else -ts

    def _turn_pwm(self, err_positive):
        left = (err_positive and self.left_if_pos) or ((not err_positive) and (not self.left_if_pos))
        if left:
            return self.pwm_turn_left_l, self.pwm_turn_left_r
        return self.pwm_turn_right_l, self.pwm_turn_right_r

    def drive_to_panel_center(self, x, z):
        err_x = x

        if z <= self.target_stop_z:
            self.finish_mission("target reached z=%.3f <= %.3f" % (z, self.target_stop_z))
            return

        if abs(err_x) <= self.align_x_threshold:
            if self.use_direct_pwm:
                self.publish_pwm(
                    self.pwm_forward_l,
                    self.pwm_forward_r,
                    "forward z=%.3f x=%.3f yaw=%.2f" % (z, x, self.current_yaw)
                )
            else:
                self.publish_twist(
                    self.forward_speed,
                    0.0,
                    "forward z=%.3f x=%.3f yaw=%.2f" % (z, x, self.current_yaw)
                )
        else:
            if self.use_direct_pwm:
                l, r = self._turn_pwm(err_x > 0.0)
                self.publish_pwm(
                    l,
                    r,
                    "align turn z=%.3f x=%.3f yaw=%.2f threshold=%.3f" %
                    (z, x, self.current_yaw, self.align_x_threshold)
                )
            else:
                ang = self._turn_ang(err_x > 0.0)
                self.publish_twist(
                    0.0,
                    ang,
                    "align turn z=%.3f x=%.3f yaw=%.2f threshold=%.3f" %
                    (z, x, self.current_yaw, self.align_x_threshold)
                )

    # =========================================================
    # Main Control
    # =========================================================
    def run_control(self):
        if self.mode == self.MODE_IDLE:
            return

        if self.mode == self.MODE_RETREAT:
            self.publish_stop("RETREAT mode is not used in panel tracking version")
            return

        if not self.sensor_ready_recently():
            self.publish_stop("camera/depth info stale")
            return

        if not self.target_visible_recently():
            self.publish_stop("left_panel not detected recently")
            return

        self.drive_to_panel_center(self.current_pose_x, self.current_distance)


if __name__ == '__main__':
    try:
        PanelPoseController()
    except rospy.ROSInterruptException:
        pass
