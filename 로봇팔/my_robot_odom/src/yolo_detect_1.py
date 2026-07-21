#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import copy
import cv2
import math
import time
import statistics
import logging
import rospy
import numpy as np
import moveit_commander
import tf2_ros
import tf2_geometry_msgs

from ultralytics import YOLO
from cv_bridge import CvBridge
from geometry_msgs.msg import PointStamped
from sensor_msgs.msg import Image, CameraInfo
from std_msgs.msg import String, Float64MultiArray
from moveit_msgs.srv import GetPositionFK, GetPositionFKRequest
from tf.transformations import euler_from_quaternion


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


class YoloObjectTFDetector:
    def __init__(self):
        rospy.init_node("yolo_object_tf_detector", anonymous=False)

        # YOLO models
        self.detect_model_path = rospy.get_param("~detect_model_path", "/home/park/best1.pt")
        self.pressed_model_path = rospy.get_param("~pressed_model_path", "/home/park/best2.pt")

        self.target_class = rospy.get_param("~target_class", "")
        self.detect_conf_thresh = float(rospy.get_param("~detect_conf_thresh", 0.5))
        self.detect_iou_thresh = float(rospy.get_param("~detect_iou_thresh", 0.45))
        self.classify_imgsz = int(rospy.get_param("~classify_imgsz", 224))

        # Camera / TF
        self.color_topic = rospy.get_param("~color_topic", "/camera/color/image_raw")
        self.depth_topic = rospy.get_param("~depth_topic", "/camera/aligned_depth_to_color/image_raw")
        self.camera_info_topic = rospy.get_param("~camera_info_topic", "/camera/color/camera_info")
        self.camera_frame = rospy.get_param("~camera_frame", "camera_color_optical_frame")
        self.base_frame = rospy.get_param("~base_frame", "base_link")
        self.view_image = bool(rospy.get_param("~view_image", True))
        self.loop_hz = float(rospy.get_param("~loop_hz", 30.0))

        self.use_depth_patch = True
        self.depth_patch_radius = 2
        self.depth_min = 0.05
        self.depth_max = 3.0
        self.target_timeout = 5.0

        # ----------------------------------------------------------
        # F3 geometry / state validation
        # Button layout used only as relative geometry, not robot target coords:
        #     F6
        #     F2  F3  F4  F5
        #     F1  B1
        # ----------------------------------------------------------
        self.button_gap = float(rospy.get_param("~button_gap", 0.06))
        self.button_distance_tol = float(rospy.get_param("~button_distance_tol", 0.035))
        self.button_middle_ratio_tol = float(rospy.get_param("~button_middle_ratio_tol", 0.18))
        self.button_line_tol = float(rospy.get_param("~button_line_tol", 0.030))
        self.f3_off_min_conf = float(rospy.get_param("~f3_off_min_conf", 0.50))
        self.f3_on_min_conf = float(rospy.get_param("~f3_on_min_conf", 0.50))
        self.f3_stable_required = int(rospy.get_param("~f3_stable_required", 3))
        self.f3_stable_window = float(rospy.get_param("~f3_stable_window", 1.0))
        self.f3_stable_pos_tol = float(rospy.get_param("~f3_stable_pos_tol", 0.020))
        self.f3_stable_min_conf = float(rospy.get_param("~f3_stable_min_conf", 0.50))
        self.target_history = {}
        self.target_history_max = int(rospy.get_param("~target_history_max", 30))

        self.button_grid = {
            "F6": (0.0, 1.0),
            "F2": (0.0, 0.0),
            "F3": (1.0, 0.0),
            "F4": (2.0, 0.0),
            "F5": (3.0, 0.0),
            "F1": (0.0, -1.0),
            "B1": (1.0, -1.0),
        }

        # Re-detection motion. While moving, YOLO inference is paused and stored targets are cleared.
        self.detection_paused = False
        self.f3_reobserve_attempted = False
        self.reobserve_settle_wait = float(rospy.get_param("~reobserve_settle_wait", 0.8))

        # MoveIt
        self.enable_moveit = True
        self.move_group_name = "arm"
        self.ee_link = "brk9_1"
        self.planning_time = float(rospy.get_param("~planning_time", 5.0))
        self.num_planning_attempts = int(rospy.get_param("~num_planning_attempts", 12))
        self.max_vel_scale = float(rospy.get_param("~max_vel_scale", 0.2))
        self.max_acc_scale = float(rospy.get_param("~max_acc_scale", 0.2))

        # push
        self.pose_push_distance = float(rospy.get_param("~pose_push_distance", 0.04))
        self.pose_back_distance = float(rospy.get_param("~pose_back_distance", 0.04))
        self.pose_push_pos_tol = 0.010
        self.pose_push_ori_tol = 0.35
        self.pose_push_joint_tol = 0.05
        self.pose_push_vel_scale = 0.01
        self.pose_push_acc_scale = 0.01
        self.pose_push_planning_time = 2.0
        self.pose_push_num_planning_attempts = 10

        self.target_offset_x = float(rospy.get_param("~target_offset_x", 0.005))
        self.target_offset_y = float(rospy.get_param("~target_offset_y", -0.02))
        self.target_offset_z = float(rospy.get_param("~target_offset_z", 0.0))

        # current monitor
        self.current_topic = "/arm/joint_current_raw"
        self.latest_joint_current_raw = None
        self.current_msg_time = None
        self.current_idx_q2 = 1
        self.current_idx_q3 = 2

        self.current_baseline_duration = 0.5
        self.current_baseline_dt = 0.05
        self.current_poll_dt = 0.02
        self.contact_q3_delta_th = 15.0
        self.contact_q2_abs_delta_th = 40.0
        self.contact_consecutive_required = 2
        self.pose_push_timeout = 6.0

        self.pre_push_joint_map = None
        self.start_pose_joint_map = {
            "Revolute1": 0.0,
            "Revolute2": 0.098,
            "Revolute3": -1.318,
            "Revolute4": 0.0,
            "Revolute5": 1.243,
        }

        # F3/F2/F4 인식이 부족할 때 사용하는 재인식 자세
        # 요청 조건: 2번 조인트 +0.08, 5번 조인트 -0.08
        self.reobserve_pose_joint_map = dict(self.start_pose_joint_map)
        self.reobserve_pose_joint_map["Revolute2"] = self.start_pose_joint_map["Revolute2"] + float(rospy.get_param("~reobserve_q2_delta", 0.08))
        self.reobserve_pose_joint_map["Revolute5"] = self.start_pose_joint_map["Revolute5"] + float(rospy.get_param("~reobserve_q5_delta", -0.08))

        # F3 버튼 임무 종료 후 복귀 자세
        # move_start_pose_panel.sh에 있던 joint 값을 코드 내부에서 직접 사용한다.
        self.finish_pose_joint_map = {
            "Revolute1": 1.573,
            "Revolute2": 0.098,
            "Revolute3": -1.318,
            "Revolute4": 0.0,
            "Revolute5": 1.58,
        }

        # fk compare / stabilization
        self.settle_initial_wait = 0.6	# FK 비교/보정 전 joint 안정화 대기
        self.settle_num_samples = 5		# 샘플링 수
        self.settle_sample_interval = 0.03	# 샘플링 사이 대기 시간
        self.use_median_joint_sampling = True	

        self.level_roll_ref = -0.088
        self.comp_q2_span = 0.30
        self.comp_q3_span = 0.30
        self.comp_step = 0.03
        self.comp_pos_weight = 40.0
        self.comp_roll_weight = 1.0
        self.comp_z_drop_weight = 60.0
        self.comp_q5_gain = 0.60
        self.comp_adaptive_stages = [
            {"name": "strict",  "q2_span_scale": 1.00, "q3_span_scale": 1.00, "q5_gain": 0.60, "z_drop_limit": 0.010},
            {"name": "medium",  "q2_span_scale": 1.15, "q3_span_scale": 1.15, "q5_gain": 0.70, "z_drop_limit": 0.015},
            {"name": "relaxed", "q2_span_scale": 1.30, "q3_span_scale": 1.30, "q5_gain": 0.80, "z_drop_limit": 0.020},
        ]
        self.joint_limits = {
            "Revolute1": (-3.141593,  3.141593),
            "Revolute2": (-1.832596,  1.832596),
            "Revolute3": (-1.483530,  1.989675),
            "Revolute4": (-3.141593,  3.141593),
            "Revolute5": (-1.989675,  1.989675),
        }

        self.command_topic = rospy.get_param("~command_topic", "/move_target_label")
        self.detected_targets = {}
        self.pending_target_label = None

        # 상위 sequence manager가 기다리는 로봇팔 완료 이벤트
        # robot_arm_grad_demo.py에서 /arm_mission/button 호출 후 F3 버튼 임무가 끝나면 SEXY_BUTTON publish
        self.arm_mission_event_topic = rospy.get_param("~arm_mission_event_topic", "/arm_mission/event")
        self.arm_mission_done_event = rospy.get_param("~arm_mission_done_event", "SEXY_BUTTON")
        self.pub_arm_mission_event = rospy.Publisher(
            self.arm_mission_event_topic,
            String,
            queue_size=10,
            latch=True
        )

        # F3 버튼 상태 전환 확인 설정
        # push 이후 F3가 ON이 된 것을 확인한 뒤, 다시 OFF로 바뀌면 SEXY_BUTTON을 publish
        self.button_state_watch_label = rospy.get_param("~button_state_watch_label", "F3")
        self.button_state_after_push_timeout = float(rospy.get_param("~button_state_after_push_timeout", 180.0))
        self.button_state_poll_dt = float(rospy.get_param("~button_state_poll_dt", 0.15))
        self.button_state_required_count = int(rospy.get_param("~button_state_required_count", 2))
        self.button_state_min_conf = float(rospy.get_param("~button_state_min_conf", 0.50))

        # 버튼을 누르기 전에는 ON이 연속 3회 이상 확인될 때만 이미 눌린 상태로 인정한다.
        self.pre_button_on_required_count = int(rospy.get_param("~pre_button_on_required_count", 3))
        self.pre_button_state_timeout = float(rospy.get_param("~pre_button_state_timeout", 2.0))

        # 버튼을 누르고 시작 자세로 복귀한 뒤 ON이 아니면 다시 누르는 최대 횟수
        self.button_push_max_attempts = int(rospy.get_param("~button_push_max_attempts", 3))
        self.button_on_confirm_timeout = float(rospy.get_param("~button_on_confirm_timeout", 3.0))

        rospy.loginfo("[yolo_detect_1] prestarted button node ready. command_topic=%s view_image=%s",
                      self.command_topic, str(self.view_image))
        rospy.loginfo("[yolo_detect_1] offsets x=%.3f y=%.3f z=%.3f push=%.3f",
                      self.target_offset_x, self.target_offset_y, self.target_offset_z, self.pose_push_distance)

        # check model files
        if not os.path.exists(self.detect_model_path):
            raise FileNotFoundError(self.detect_model_path)
        if not os.path.exists(self.pressed_model_path):
            raise FileNotFoundError(self.pressed_model_path)

        logging.getLogger("ultralytics").setLevel(logging.ERROR)
        rospy.loginfo("Loading detect model: %s", self.detect_model_path)
        self.detect_model = YOLO(self.detect_model_path)
        rospy.loginfo("Loading pressed model: %s", self.pressed_model_path)
        self.pressed_model = YOLO(self.pressed_model_path)

        # TF
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        # ROS image state
        self.bridge = CvBridge()
        self.latest_color_image = None
        self.latest_depth_image = None
        self.latest_color_stamp = None
        self.latest_depth_stamp = None

        # camera intrinsics from CameraInfo
        self.fx = None
        self.fy = None
        self.cx = None
        self.cy = None

        # subs
        self.cmd_sub = rospy.Subscriber(self.command_topic, String, self.command_callback, queue_size=10)
        self.current_sub = rospy.Subscriber(self.current_topic, Float64MultiArray, self.current_callback, queue_size=20)
        self.color_sub = rospy.Subscriber(self.color_topic, Image, self.color_callback, queue_size=1, buff_size=2**24)
        self.depth_sub = rospy.Subscriber(self.depth_topic, Image, self.depth_callback, queue_size=1, buff_size=2**24)
        self.info_sub = rospy.Subscriber(self.camera_info_topic, CameraInfo, self.camera_info_callback, queue_size=1)

        # MoveIt
        self.arm_group = None
        self.fk_srv = None
        if self.enable_moveit:
            try:
                moveit_commander.roscpp_initialize(sys.argv)
                self.arm_group = moveit_commander.MoveGroupCommander(self.move_group_name)
                self.arm_group.set_pose_reference_frame(self.base_frame)
                self.arm_group.set_planning_time(self.planning_time)
                self.arm_group.set_num_planning_attempts(self.num_planning_attempts)
                self.arm_group.set_max_velocity_scaling_factor(self.max_vel_scale)
                self.arm_group.set_max_acceleration_scaling_factor(self.max_acc_scale)
                self.arm_group.set_end_effector_link(self.ee_link)
                rospy.wait_for_service('/compute_fk', timeout=5.0)
                self.fk_srv = rospy.ServiceProxy('/compute_fk', GetPositionFK)
            except Exception as e:
                rospy.logerr("Failed to initialize MoveIt/FK: %s", str(e))
                self.enable_moveit = False
                self.arm_group = None
                self.fk_srv = None

        rospy.on_shutdown(self.shutdown_hook)

    def color_callback(self, msg):
        try:
            self.latest_color_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
            self.latest_color_stamp = msg.header.stamp
        except Exception as e:
            rospy.logerr_throttle(1.0, "Color callback failed: %s", str(e))

    def depth_callback(self, msg):
        try:
            self.latest_depth_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
            self.latest_depth_stamp = msg.header.stamp
        except Exception as e:
            rospy.logerr_throttle(1.0, "Depth callback failed: %s", str(e))

    def camera_info_callback(self, msg):
        if len(msg.K) >= 9:
            self.fx = float(msg.K[0])
            self.fy = float(msg.K[4])
            self.cx = float(msg.K[2])
            self.cy = float(msg.K[5])

    def camera_ready(self):
        return (
            self.latest_color_image is not None and
            self.latest_depth_image is not None and
            self.fx is not None and
            self.fy is not None and
            self.cx is not None and
            self.cy is not None
        )

    def norm3(self, a, b):
        return math.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2 + (a[2]-b[2])**2)

    def pose_to_xyz(self, pose):
        return [pose.position.x, pose.position.y, pose.position.z]

    def compute_xyz_error(self, target_xyz, current_xyz):
        dx = target_xyz[0] - current_xyz[0]
        dy = target_xyz[1] - current_xyz[1]
        dz = target_xyz[2] - current_xyz[2]
        return {"dx": dx, "dy": dy, "dz": dz, "dist": math.sqrt(dx*dx + dy*dy + dz*dz)}

    def format_joint_values(self, joint_names, joint_values):
        return ", ".join(["%s=%.4f" % (n, v) for n, v in zip(joint_names, joint_values)])

    def current_callback(self, msg):
        try:
            self.latest_joint_current_raw = list(msg.data)
            self.current_msg_time = rospy.Time.now()
        except Exception:
            self.latest_joint_current_raw = None

    def command_callback(self, msg):
        label = msg.data.strip().upper()
        if label:
            self.pending_target_label = label
            if label == "F3":
                self.f3_reobserve_attempted = False
            rospy.loginfo("Received target label command: %s", label)

    def get_depth_robust(self, depth_image, u, v, r=2):
        if depth_image is None:
            return 0.0

        h, w = depth_image.shape[:2]
        vals = []
        for yy in range(max(0, v-r), min(h, v+r+1)):
            for xx in range(max(0, u-r), min(w, u+r+1)):
                raw = depth_image[yy, xx]
                if np.issubdtype(depth_image.dtype, np.integer):
                    d = float(raw) / 1000.0  # mm -> m
                else:
                    d = float(raw)
                if d > 0:
                    vals.append(d)

        if not vals:
            return 0.0

        z = float(np.median(vals))
        if z < self.depth_min or z > self.depth_max:
            return 0.0
        return z

    def pixel_to_camera_xyz(self, u, v, z):
        x = (float(u) - self.cx) * float(z) / self.fx
        y = (float(v) - self.cy) * float(z) / self.fy
        return x, y, float(z)

    def transform_point_to_base(self, x, y, z, stamp):
        p = PointStamped()
        p.header.stamp = stamp
        p.header.frame_id = self.camera_frame
        p.point.x = float(x)
        p.point.y = float(y)
        p.point.z = float(z)
        try:
            return self.tf_buffer.transform(p, self.base_frame, rospy.Duration(0.5))
        except Exception as e:
            rospy.logwarn_throttle(1.0, "Transform failed: %s", str(e))
            return None

    def classify_pressed_state(self, bgr, x1, y1, x2, y2):
        h, w = bgr.shape[:2]
        x1 = clamp(int(x1), 0, w-1)
        x2 = clamp(int(x2), 0, w-1)
        y1 = clamp(int(y1), 0, h-1)
        y2 = clamp(int(y2), 0, h-1)
        if x2 <= x1 or y2 <= y1:
            return "unknown", 0.0
        crop = bgr[y1:y2, x1:x2]
        if crop.size == 0:
            return "unknown", 0.0
        try:
            cls_result = self.pressed_model.predict(source=crop, imgsz=self.classify_imgsz, verbose=False)[0]
            if getattr(cls_result, "probs", None) is None:
                return "unknown", 0.0
            names = cls_result.names
            idx = int(cls_result.probs.top1)
            return str(names[idx]).lower(), float(cls_result.probs.top1conf)
        except Exception as e:
            rospy.logwarn_throttle(1.0, "Pressed classification failed: %s", str(e))
            return "unknown", 0.0

    def update_detected_target(self, label, bx, by, bz, conf, stamp, pressed_label="unknown", pressed_conf=0.0):
        key = str(label).strip().upper()
        info = {
            "x": float(bx), "y": float(by), "z": float(bz),
            "conf": float(conf), "stamp": stamp,
            "pressed_label": str(pressed_label).lower(), "pressed_conf": float(pressed_conf)
        }
        self.detected_targets[key] = info

        hist = self.target_history.setdefault(key, [])
        hist.append(dict(info))
        if len(hist) > self.target_history_max:
            del hist[:-self.target_history_max]

    def clear_stale_targets(self, now):
        for k in list(self.detected_targets.keys()):
            if (now - self.detected_targets[k]["stamp"]).to_sec() > self.target_timeout:
                del self.detected_targets[k]

        # 연속 프레임 안정성 확인용 history도 오래된 것은 삭제한다.
        keep_sec = max(self.target_timeout, self.f3_stable_window)
        for k in list(self.target_history.keys()):
            self.target_history[k] = [
                info for info in self.target_history[k]
                if (now - info["stamp"]).to_sec() <= keep_sec
            ]
            if not self.target_history[k]:
                del self.target_history[k]

    def clear_detection_buffers(self):
        self.detected_targets.clear()
        self.target_history.clear()

    def target_xyz(self, info):
        return [float(info["x"]), float(info["y"]), float(info["z"])]

    def expected_button_distance(self, label_a, label_b):
        a = str(label_a).strip().upper()
        b = str(label_b).strip().upper()
        if a not in self.button_grid or b not in self.button_grid:
            return None
        ax, ay = self.button_grid[a]
        bx, by = self.button_grid[b]
        grid_dist = math.sqrt((ax - bx) ** 2 + (ay - by) ** 2)
        return grid_dist * self.button_gap

    def check_button_distance(self, label_a, info_a, label_b, info_b, reason="geometry"):
        expected = self.expected_button_distance(label_a, label_b)
        if expected is None:
            return False
        actual = self.norm3(self.target_xyz(info_a), self.target_xyz(info_b))
        ok = abs(actual - expected) <= self.button_distance_tol
        log_msg = "[%s] distance %s: %s-%s actual=%.3f expected=%.3f tol=%.3f" % (
            reason, "ok" if ok else "bad", label_a, label_b, actual, expected, self.button_distance_tol
        )
        if ok:
            rospy.loginfo(log_msg)
        else:
            rospy.logwarn(log_msg)
        return ok

    def check_f3_between_f2_f4(self, f2, f3, f4):
        # 거리 확인: F2-F4 ≈ 12cm, F2-F3/F3-F4 ≈ 6cm
        d24_ok = self.check_button_distance("F2", f2, "F4", f4, reason="f3_verify")
        d23_ok = self.check_button_distance("F2", f2, "F3", f3, reason="f3_verify")
        d34_ok = self.check_button_distance("F3", f3, "F4", f4, reason="f3_verify")
        if not (d24_ok and d23_ok and d34_ok):
            return False

        p2 = np.array(self.target_xyz(f2), dtype=np.float64)
        p3 = np.array(self.target_xyz(f3), dtype=np.float64)
        p4 = np.array(self.target_xyz(f4), dtype=np.float64)
        v24 = p4 - p2
        denom = float(np.dot(v24, v24))
        if denom < 1e-8:
            rospy.logwarn("[f3_verify] F2/F4 vector too small")
            return False

        ratio = float(np.dot(p3 - p2, v24) / denom)
        proj = p2 + ratio * v24
        line_err = float(np.linalg.norm(p3 - proj))

        ratio_ok = abs(ratio - 0.5) <= self.button_middle_ratio_tol
        line_ok = line_err <= self.button_line_tol

        if ratio_ok and line_ok:
            rospy.loginfo("[f3_verify] order/line ok: F2 -> F3 -> F4 ratio=%.3f line_err=%.3f", ratio, line_err)
            return True

        rospy.logwarn("[f3_verify] order/line bad: F2 -> F3 -> F4 ratio=%.3f line_err=%.3f", ratio, line_err)
        return False

    def validate_panel_geometry_snapshot(self):
        labels = [k for k in self.detected_targets.keys() if k in self.button_grid]
        suspicious = []
        for i in range(len(labels)):
            for j in range(i + 1, len(labels)):
                a, b = labels[i], labels[j]
                expected = self.expected_button_distance(a, b)
                if expected is None or expected <= 0.0:
                    continue
                actual = self.norm3(self.target_xyz(self.detected_targets[a]), self.target_xyz(self.detected_targets[b]))
                if abs(actual - expected) > self.button_distance_tol:
                    suspicious.append((a, b, actual, expected))

        if suspicious:
            for a, b, actual, expected in suspicious[:5]:
                rospy.logwarn_throttle(
                    1.0,
                    "[panel_geometry] suspicious pair %s-%s actual=%.3f expected=%.3f",
                    a, b, actual, expected
                )
            return False
        return True

    def get_button_state(self, info):
        state = str(info.get("pressed_label", "unknown")).lower()
        conf = float(info.get("pressed_conf", 0.0))
        return state, conf

    def is_f3_off(self, f3_info):
        state, conf = self.get_button_state(f3_info)
        ok = (state == "off" and conf >= self.f3_off_min_conf)
        rospy.loginfo("[f3_state] OFF check: state=%s conf=%.2f ok=%s", state, conf, str(ok))
        return ok

    def is_f3_on(self, f3_info):
        state, conf = self.get_button_state(f3_info)
        ok = (state == "on" and conf >= self.f3_on_min_conf)
        rospy.loginfo("[f3_state] ON check: state=%s conf=%.2f ok=%s", state, conf, str(ok))
        return ok

    def is_f3_stable_recently(self):
        now = rospy.Time.now()
        hist = []
        for info in self.target_history.get("F3", []):
            if (now - info["stamp"]).to_sec() <= self.f3_stable_window and float(info.get("conf", 0.0)) >= self.f3_stable_min_conf:
                hist.append(info)

        if len(hist) < self.f3_stable_required:
            rospy.logwarn_throttle(1.0, "[f3_stable] insufficient F3 history: %d/%d", len(hist), self.f3_stable_required)
            return False

        pts = np.array([self.target_xyz(info) for info in hist[-self.f3_stable_required:]], dtype=np.float64)
        center = np.median(pts, axis=0)
        max_dev = float(np.max(np.linalg.norm(pts - center, axis=1)))
        ok = max_dev <= self.f3_stable_pos_tol
        if ok:
            rospy.loginfo("[f3_stable] ok: count=%d max_dev=%.3f", len(hist), max_dev)
        else:
            rospy.logwarn("[f3_stable] bad: count=%d max_dev=%.3f tol=%.3f", len(hist), max_dev, self.f3_stable_pos_tol)
        return ok

    def make_f3_selection(self, info, source, skip_pre_state_check=False):
        selected = dict(info)
        selected["source"] = str(source)
        selected["skip_pre_state_check"] = bool(skip_pre_state_check)
        return selected

    def estimate_f3_from_f2_f4(self, f2=None, f4=None, source="estimated_from_current_f2_f4"):
        # F3가 직접 검출되지 않았을 때, F2와 F4의 base_link 좌표 중간값으로 F3 위치를 추정한다.
        if f2 is None:
            f2 = self.detected_targets.get("F2")
        if f4 is None:
            f4 = self.detected_targets.get("F4")
        if f2 is None or f4 is None:
            return None

        estimated = {
            "x": (f2["x"] + f4["x"]) / 2.0,
            "y": (f2["y"] + f4["y"]) / 2.0,
            "z": (f2["z"] + f4["z"]) / 2.0,
            "conf": min(float(f2.get("conf", 0.0)), float(f4.get("conf", 0.0))),
            "stamp": rospy.Time.now(),
            "pressed_label": "estimated",
            "pressed_conf": 0.0,
        }

        rospy.logwarn(
            "[f3_select] F3 estimated from F2/F4: x=%.3f y=%.3f z=%.3f | source=%s",
            estimated["x"], estimated["y"], estimated["z"], source
        )
        return self.make_f3_selection(estimated, source, skip_pre_state_check=True)

    def select_f3_target(self):
        self.f3_select_fail_reason = "unknown"
        self.validate_panel_geometry_snapshot()

        f2 = self.detected_targets.get("F2")
        f3 = self.detected_targets.get("F3")
        f4 = self.detected_targets.get("F4")

        # 1) F3가 현재 인식된 경우: ON이면 움직이지 않고 OFF 대기, OFF이면 위치 검증 후 사용.
        if f3 is not None:
            if self.is_f3_on(f3):
                rospy.logwarn("[f3_select] F3 is already ON. Skip arm motion and wait for OFF.")
                return self.make_f3_selection(f3, "direct_f3_already_on_wait_off", skip_pre_state_check=True)

            if not self.is_f3_off(f3):
                rospy.logwarn("[f3_select] F3 detected but OFF is not confirmed. Do not fallback. Wait.")
                self.f3_select_fail_reason = "wait_state"
                return None

            if f2 is not None and f4 is not None:
                if self.check_f3_between_f2_f4(f2, f3, f4):
                    return self.make_f3_selection(f3, "direct_f3_verified_f2_f4")

            if f2 is not None:
                if self.check_button_distance("F2", f2, "F3", f3, reason="f3_verify"):
                    return self.make_f3_selection(f3, "direct_f3_verified_f2_only")

            if f4 is not None:
                if self.check_button_distance("F3", f3, "F4", f4, reason="f3_verify"):
                    return self.make_f3_selection(f3, "direct_f3_verified_f4_only")

            if self.is_f3_stable_recently():
                return self.make_f3_selection(f3, "direct_f3_stable_only")

            rospy.logwarn("[f3_select] F3 is OFF but geometry/stability check failed. Re-observe if possible.")
            self.f3_select_fail_reason = "need_reobserve"
            return None

        # 2) F3가 현재 인식되지 않은 경우: F2/F4가 둘 다 정상일 때만 중간 좌표 추정.
        if f2 is not None and f4 is not None:
            if self.check_button_distance("F2", f2, "F4", f4, reason="f3_fallback"):
                return self.estimate_f3_from_f2_f4(f2, f4, source="estimated_from_current_f2_f4")

        rospy.logwarn_throttle(1.0, "[f3_select] F3 not detected and F2/F4 fallback unavailable. Re-observe if possible.")
        self.f3_select_fail_reason = "need_reobserve"
        return None

    def apply_target_offset(self, info):
        source = str(info.get("source", "unknown"))
        ox = self.target_offset_x
        oy = self.target_offset_y
        oz = self.target_offset_z

        # F2/F4 중간 추정 좌표는 버튼 가로 방향 보정으로 밀리면 F2 쪽으로 갈 수 있으므로 x offset은 제외한다.
        if source == "estimated_from_current_f2_f4":
            ox = 0.0

        return info["x"] + ox, info["y"] + oy, info["z"] + oz

    def move_to_reobserve_pose(self):
        if not self.enable_moveit or self.arm_group is None:
            rospy.logwarn("[reobserve] MoveIt is not available; cannot move to reobserve pose")
            return False

        rospy.logwarn(
            "[reobserve] move to re-detection pose. Revolute2=%.3f Revolute5=%.3f. Detection paused while moving.",
            self.reobserve_pose_joint_map["Revolute2"], self.reobserve_pose_joint_map["Revolute5"]
        )
        self.detection_paused = True
        self.clear_detection_buffers()
        ok = self.move_to_joint_map(self.reobserve_pose_joint_map, label="F3_reobserve", prefix="Move to F3 reobserve pose")
        rospy.sleep(self.reobserve_settle_wait)
        self.clear_detection_buffers()
        self.detection_paused = False
        rospy.logwarn("[reobserve] pose move done ok=%s. Detection resumed after buffer clear.", str(ok))
        return bool(ok)

    def print_detected_targets_log(self):
        if not self.detected_targets:
            rospy.loginfo("No detected targets stored.")
            return
        for label, info in sorted(self.detected_targets.items()):
            rospy.loginfo("[TARGET] %s -> x=%.3f y=%.3f z=%.3f conf=%.2f state=%s(%.2f)",
                          label, info["x"], info["y"], info["z"], info["conf"],
                          info["pressed_label"], info["pressed_conf"])

    def detect_button_state_once(self, target_label):
        """
        현재 color image 한 프레임에서 target_label의 ON/OFF 상태만 확인한다.
        push 이후 이벤트 판단용이며, depth/TF/MoveIt 동작은 수행하지 않는다.
        """
        if self.latest_color_image is None:
            return None, 0.0

        try:
            color_image = self.latest_color_image.copy()
            detect_result = self.detect_model.predict(
                source=color_image,
                conf=self.detect_conf_thresh,
                iou=self.detect_iou_thresh,
                verbose=False
            )[0]

            if detect_result is None or detect_result.boxes is None or len(detect_result.boxes) <= 0:
                return None, 0.0

            target_key = str(target_label).strip().upper()
            best_state = None
            best_conf = 0.0

            names = detect_result.names
            for box in detect_result.boxes:
                cls_id = int(box.cls[0].item())
                label = str(names.get(cls_id, str(cls_id))).strip().upper()
                if label != target_key:
                    continue

                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                pressed_label, pressed_conf = self.classify_pressed_state(color_image, x1, y1, x2, y2)

                if pressed_conf > best_conf:
                    best_state = str(pressed_label).lower()
                    best_conf = float(pressed_conf)

            return best_state, best_conf

        except Exception as e:
            rospy.logwarn_throttle(1.0, "detect_button_state_once failed: %s", str(e))
            return None, 0.0

    def confirm_button_state_consecutive(self, label, desired_state, required_count, timeout):
        """
        현재 카메라 영상에서 원하는 버튼 상태가 지정 횟수만큼 연속 검출되는지 확인한다.
        중간에 다른 상태, 미검출, 낮은 신뢰도가 나오면 연속 횟수를 초기화한다.
        """
        target_label = str(label).strip().upper()
        desired_state = str(desired_state).strip().lower()
        required = max(1, int(required_count))
        consecutive = 0
        start_t = time.time()

        rospy.loginfo(
            "[button_state] confirm %s=%s consecutively %d times, timeout=%.1fs",
            target_label, desired_state.upper(), required, float(timeout)
        )

        while not rospy.is_shutdown() and (time.time() - start_t) < float(timeout):
            state, conf = self.detect_button_state_once(target_label)

            if state == desired_state and conf >= self.button_state_min_conf:
                consecutive += 1
                rospy.loginfo(
                    "[button_state] %s %s count=%d/%d conf=%.2f",
                    target_label, desired_state.upper(), consecutive, required, conf
                )
                if consecutive >= required:
                    return True
            else:
                if consecutive > 0:
                    rospy.loginfo(
                        "[button_state] %s consecutive %s reset: state=%s conf=%.2f",
                        target_label, desired_state.upper(), str(state), float(conf)
                    )
                consecutive = 0

            rospy.sleep(self.button_state_poll_dt)

        rospy.logwarn(
            "[button_state] %s consecutive %s confirmation failed: required=%d",
            target_label, desired_state.upper(), required
        )
        return False


    def detect_stable_button_state(self, label="F3", required_count=None, timeout=None):
        """
        현재 버튼 상태를 연속 검출로 확인한다.
        ON이 연속 확인되면 "on", OFF가 연속 확인되면 "off",
        제한 시간 안에 어느 쪽도 확정되지 않으면 "unknown"을 반환한다.
        """
        target_label = str(label).strip().upper()
        required = max(1, int(
            self.button_state_required_count if required_count is None else required_count
        ))
        check_timeout = float(
            self.button_on_confirm_timeout if timeout is None else timeout
        )

        on_count = 0
        off_count = 0
        start_t = time.time()

        rospy.loginfo(
            "[button_state] determine stable %s state, required=%d timeout=%.1fs",
            target_label, required, check_timeout
        )

        while not rospy.is_shutdown() and (time.time() - start_t) < check_timeout:
            state, conf = self.detect_button_state_once(target_label)

            if conf < self.button_state_min_conf:
                on_count = 0
                off_count = 0
                rospy.sleep(self.button_state_poll_dt)
                continue

            if state == "on":
                on_count += 1
                off_count = 0
                if on_count >= required:
                    rospy.loginfo("[button_state] %s stable ON confirmed", target_label)
                    return "on"
            elif state == "off":
                off_count += 1
                on_count = 0
                if off_count >= required:
                    rospy.loginfo("[button_state] %s stable OFF confirmed", target_label)
                    return "off"
            else:
                on_count = 0
                off_count = 0

            rospy.sleep(self.button_state_poll_dt)

        rospy.logwarn("[button_state] %s stable state could not be determined", target_label)
        return "unknown"

    def wait_for_button_off_after_on_confirmed(self, label="F3"):
        """ON 상태가 이미 확인된 뒤 OFF가 연속 확인될 때까지 기다린다."""
        target_label = str(label).strip().upper()
        timeout = self.button_state_after_push_timeout
        required = max(1, self.button_state_required_count)
        off_count = 0
        start_t = time.time()

        rospy.loginfo(
            "[button_state] %s ON already confirmed. wait OFF, timeout=%.1fs, required=%d",
            target_label, timeout, required
        )

        while not rospy.is_shutdown() and (time.time() - start_t) < timeout:
            state, conf = self.detect_button_state_once(target_label)

            if state == "off" and conf >= self.button_state_min_conf:
                off_count += 1
                if off_count >= required:
                    rospy.loginfo("[button_state] %s OFF confirmed after ON", target_label)
                    return True
            else:
                off_count = 0

            rospy.sleep(self.button_state_poll_dt)

        rospy.logwarn("[button_state] %s OFF wait timeout after ON confirmation", target_label)
        return False

    def wait_for_button_on_to_off_after_push(self, label="F3"):
        """
        push 이후 버튼 상태가 ON으로 바뀐 것을 먼저 확인하고,
        이후 OFF가 연속으로 확인되면 True를 반환한다.
        """
        target_label = str(label).strip().upper()
        timeout = self.button_state_after_push_timeout
        required = max(1, self.button_state_required_count)

        rospy.loginfo(
            "[button_state] wait %s transition: ON -> OFF, timeout=%.1fs, required=%d",
            target_label, timeout, required
        )

        saw_on = False
        on_count = 0
        off_count = 0
        start_t = time.time()

        while not rospy.is_shutdown() and (time.time() - start_t) < timeout:
            state, conf = self.detect_button_state_once(target_label)

            if state is None:
                rospy.loginfo_throttle(0.5, "[button_state] %s not detected yet", target_label)
                rospy.sleep(self.button_state_poll_dt)
                continue

            rospy.loginfo_throttle(
                0.3,
                "[button_state] %s state=%s conf=%.2f saw_on=%s",
                target_label, state, conf, str(saw_on)
            )

            if conf < self.button_state_min_conf:
                rospy.sleep(self.button_state_poll_dt)
                continue

            if not saw_on:
                if state == "on":
                    on_count += 1
                    if on_count >= required:
                        saw_on = True
                        off_count = 0
                        rospy.loginfo("[button_state] %s ON confirmed", target_label)
                else:
                    on_count = 0
            else:
                if state == "off":
                    off_count += 1
                    if off_count >= required:
                        rospy.loginfo("[button_state] %s OFF confirmed after ON", target_label)
                        return True
                else:
                    off_count = 0

            rospy.sleep(self.button_state_poll_dt)

        rospy.logwarn("[button_state] %s ON->OFF transition timeout", target_label)
        return False

    def wait_for_button_off_without_motion(self, label="F3"):
        """
        F3가 이미 ON인 상태로 인식된 경우, 로봇팔을 움직이지 않고 OFF가 될 때까지 기다린다.
        """
        target_label = str(label).strip().upper()
        timeout = self.button_state_after_push_timeout
        required = max(1, self.button_state_required_count)

        rospy.loginfo(
            "[button_state] %s already ON. wait OFF without arm motion, timeout=%.1fs, required=%d",
            target_label, timeout, required
        )

        off_count = 0
        start_t = time.time()
        while not rospy.is_shutdown() and (time.time() - start_t) < timeout:
            state, conf = self.detect_button_state_once(target_label)

            if state is None:
                rospy.loginfo_throttle(0.5, "[button_state] %s not detected while waiting OFF", target_label)
                rospy.sleep(self.button_state_poll_dt)
                continue

            rospy.loginfo_throttle(
                0.3,
                "[button_state] %s state=%s conf=%.2f waiting_off_without_motion",
                target_label, state, conf
            )

            if conf < self.button_state_min_conf:
                rospy.sleep(self.button_state_poll_dt)
                continue

            if state == "off":
                off_count += 1
                if off_count >= required:
                    rospy.loginfo("[button_state] %s OFF confirmed without arm motion", target_label)
                    return True
            else:
                off_count = 0

            rospy.sleep(self.button_state_poll_dt)

        rospy.logwarn("[button_state] %s OFF wait timeout without arm motion", target_label)
        return False

    def get_current_q2_q3_raw(self):
        if self.latest_joint_current_raw is None:
            return None, None
        data = self.latest_joint_current_raw
        if len(data) <= max(self.current_idx_q2, self.current_idx_q3):
            return None, None
        return float(data[self.current_idx_q2]), float(data[self.current_idx_q3])

    def measure_push_current_baseline(self, label="target"):
        samples_q2, samples_q3 = [], []
        start_t = time.time()
        while time.time() - start_t < self.current_baseline_duration and not rospy.is_shutdown():
            i2, i3 = self.get_current_q2_q3_raw()
            if i2 is not None and i3 is not None:
                samples_q2.append(i2)
                samples_q3.append(i3)
            rospy.sleep(self.current_baseline_dt)
        if not samples_q2:
            return None
        baseline = {"q2": statistics.median(samples_q2), "q3": statistics.median(samples_q3)}
        rospy.loginfo("Baseline current [%s]: q2=%.1f q3=%.1f", label, baseline["q2"], baseline["q3"])
        return baseline

    def check_contact_from_current(self, baseline, label="target"):
        if baseline is None:
            return False, None
        i2, i3 = self.get_current_q2_q3_raw()
        if i2 is None or i3 is None:
            return False, None
        d2 = i2 - baseline["q2"]
        d3 = i3 - baseline["q3"]
        contact = (d3 >= self.contact_q3_delta_th) and (abs(d2) >= self.contact_q2_abs_delta_th)
        info = {"i2": i2, "i3": i3, "d2": d2, "d3": d3}
        rospy.loginfo_throttle(0.2, "Current monitor [%s]: q2=%.1f(d=%.1f) q3=%.1f(d=%.1f) contact=%s",
                               label, i2, d2, i3, d3, str(contact))
        return contact, info

    def get_current_joint_map(self):
        joint_names = self.arm_group.get_active_joints()
        joint_values = self.arm_group.get_current_joint_values()
        return joint_names, joint_values, {n: i for i, n in enumerate(joint_names)}

    def get_current_tcp_pose(self):
        return self.arm_group.get_current_pose(self.ee_link).pose

    def get_roll_from_pose(self, pose):
        q = [pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w]
        return euler_from_quaternion(q)

    def get_fk_pose(self, joint_names, joint_positions, fk_link):
        if self.fk_srv is None:
            return None
        try:
            req = GetPositionFKRequest()
            req.header.frame_id = self.base_frame
            req.fk_link_names = [fk_link]
            req.robot_state.joint_state.name = list(joint_names)
            req.robot_state.joint_state.position = list(joint_positions)
            res = self.fk_srv(req)
            if res.error_code.val != 1 or not res.pose_stamped:
                return None
            return res.pose_stamped[0].pose
        except Exception:
            return None

    def collect_joint_value_samples(self):
        rospy.sleep(self.settle_initial_wait)
        joint_names = None
        samples = []
        for idx in range(self.settle_num_samples):
            names, values, _ = self.get_current_joint_map()
            if joint_names is None:
                joint_names = list(names)
            samples.append(list(values))
            if idx < self.settle_num_samples - 1:
                rospy.sleep(self.settle_sample_interval)
        arr = np.array(samples, dtype=np.float64)
        return {"joint_names": joint_names, "median_values": np.median(arr, axis=0).tolist()}

    def sample_stabilized_joints(self, label="target", prefix="Sampling stabilized joints"):
        info = self.collect_joint_value_samples()
        rospy.loginfo("%s [%s] median joints: %s", prefix, label,
                      self.format_joint_values(info["joint_names"], info["median_values"]))
        return list(info["joint_names"]), list(info["median_values"])

    def get_stabilized_joint_map(self, label="target"):
        names, joints = self.sample_stabilized_joints(label=label, prefix="Save stabilized joints")
        return {n: float(v) for n, v in zip(names, joints)}

    def move_to_joint_map(self, joint_map, label="target", prefix="Move to joint map"):
        try:
            self.arm_group.set_start_state_to_current_state()
            self.arm_group.set_planning_time(self.planning_time)
            self.arm_group.set_num_planning_attempts(self.num_planning_attempts)
            self.arm_group.set_goal_joint_tolerance(0.02)
            self.arm_group.clear_pose_targets()
            self.arm_group.clear_path_constraints()
            self.arm_group.set_joint_value_target(joint_map)
            ok = self.arm_group.go(wait=True)
            self.arm_group.stop()
            self.arm_group.clear_pose_targets()
            rospy.sleep(0.3)
            rospy.loginfo("%s [%s]: ok=%s", prefix, label, str(ok))
            return bool(ok)
        except Exception as e:
            rospy.logerr("%s failed for [%s]: %s", prefix, label, str(e))
            return False

    def move_to_saved_pre_push_then_start_pose(self, label="target"):
        ok_pre = True
        if self.pre_push_joint_map is not None:
            ok_pre = self.move_to_joint_map(
                self.pre_push_joint_map,
                label=label,
                prefix="Move back to pre-push pose"
            )
            if not ok_pre:
                rospy.logwarn(
                    "Move back to pre-push pose failed [%s], but continue to saved start pose",
                    label
                )
            rospy.sleep(0.5)

        ok_start = self.move_to_joint_map(
            self.start_pose_joint_map,
            label=label,
            prefix="Move back to saved start pose"
        )

        # 재시도 가능 여부는 최종 안전 자세인 start pose 복귀 성공 여부로 판단한다.
        # pre-push 자세 복귀 실패는 별도로 기록하되, start pose가 성공하면 복귀 성공이다.
        return {
            "pre_push_ok": bool(ok_pre),
            "start_pose_ok": bool(ok_start),
        }

    def move_to_finish_pose_after_button(self, label="F3"):
        """
        F3 ON->OFF 확인 후 마무리 자세로 복귀한다.
        기존 move_start_pose_panel.sh의 joint 값과 동일한 목표 자세를 사용한다.
        """
        rospy.loginfo("[arm_mission] move to finish pose after %s button mission", str(label))
        return self.move_to_joint_map(
            self.finish_pose_joint_map,
            label=label,
            prefix="Move to finish pose after button mission"
        )

    def log_target_vs_current_fk(self, target_xyz, label="target", prefix="FK compare"):
        info = self.collect_joint_value_samples()
        fk_pose = self.get_fk_pose(info["joint_names"], info["median_values"], self.ee_link)
        if fk_pose is None:
            return
        err = self.compute_xyz_error(target_xyz, self.pose_to_xyz(fk_pose))
        roll, pitch, yaw = self.get_roll_from_pose(fk_pose)
        rospy.loginfo("%s [%s] delta(target-fk): dx=%.4f dy=%.4f dz=%.4f | norm=%.4f m",
                      prefix, label, err["dx"], err["dy"], err["dz"], err["dist"])
        rospy.loginfo("%s [%s] FK tcp RPY: roll=%.4f pitch=%.4f yaw=%.4f", prefix, label, roll, pitch, yaw)

    def clamp_joint(self, name, value):
        lo, hi = self.joint_limits[name]
        return clamp(value, lo, hi)

    def compute_q5_for_level(self, q2, q3):
        return self.clamp_joint("Revolute5", self.level_roll_ref - q2 - q3)

    def blend_q5_target(self, current_q5, desired_q5, gain=None):
        if gain is None:
            gain = self.comp_q5_gain
        return self.clamp_joint("Revolute5", current_q5 + gain * (desired_q5 - current_q5))

    def score_comp_candidate(self, pos_err, roll_err, z_drop):
        return self.comp_pos_weight * pos_err + self.comp_roll_weight * roll_err + self.comp_z_drop_weight * z_drop

    def evaluate_comp_candidate(self, joint_names, joint_positions, target_xyz):
        fk_pose = self.get_fk_pose(joint_names, joint_positions, self.ee_link)
        if fk_pose is None:
            return None
        pos = self.pose_to_xyz(fk_pose)
        err = self.compute_xyz_error(target_xyz, pos)
        z_drop = max(0.0, target_xyz[2] - pos[2])
        roll, pitch, yaw = self.get_roll_from_pose(fk_pose)
        roll_err = abs(roll - self.level_roll_ref)
        return {"joint_names": list(joint_names), "joints": list(joint_positions), "score": self.score_comp_candidate(err["dist"], roll_err, z_drop)}

    def find_best_compensation_once(self, target_xyz):
        info = self.collect_joint_value_samples()
        joint_names = info["joint_names"]
        current_joints = info["median_values"]
        name_to_idx = {n: i for i, n in enumerate(joint_names)}
        i2, i3, i5 = name_to_idx["Revolute2"], name_to_idx["Revolute3"], name_to_idx["Revolute5"]
        q2_cur, q3_cur = current_joints[i2], current_joints[i3]

        best = None
        for stage in self.comp_adaptive_stages:
            q2_span = self.comp_q2_span * stage["q2_span_scale"]
            q3_span = self.comp_q3_span * stage["q3_span_scale"]
            z_drop_limit = stage["z_drop_limit"]
            q5_gain = stage["q5_gain"]
            for dq2 in np.arange(-q2_span, q2_span + 1e-9, self.comp_step):
                for dq3 in np.arange(-q3_span, q3_span + 1e-9, self.comp_step):
                    cand = list(current_joints)
                    q2_new = self.clamp_joint("Revolute2", q2_cur + dq2)
                    q3_new = self.clamp_joint("Revolute3", q3_cur + dq3)
                    q5_new = self.blend_q5_target(current_joints[i5], self.compute_q5_for_level(q2_new, q3_new), q5_gain)
                    cand[i2], cand[i3], cand[i5] = q2_new, q3_new, q5_new
                    ev = self.evaluate_comp_candidate(joint_names, cand, target_xyz)
                    if ev is None:
                        continue
                    fk_pose = self.get_fk_pose(joint_names, cand, self.ee_link)
                    pos = self.pose_to_xyz(fk_pose)
                    z_drop = max(0.0, target_xyz[2] - pos[2])
                    if z_drop > z_drop_limit:
                        continue
                    if best is None or ev["score"] < best["score"]:
                        best = ev
            if best is not None:
                break
        return best

    def move_to_position_only(self, x, y, z, label="target"):
        try:
            eef_link = self.arm_group.get_end_effector_link()
            self.arm_group.set_start_state_to_current_state()
            self.arm_group.set_goal_position_tolerance(0.01)
            self.arm_group.set_planning_time(self.planning_time)
            self.arm_group.set_num_planning_attempts(self.num_planning_attempts)
            self.arm_group.clear_pose_targets()
            self.arm_group.clear_path_constraints()
            self.arm_group.set_position_target([x, y, z], eef_link)
            rospy.loginfo("Position-only target [%s]: x=%.3f y=%.3f z=%.3f", label, x, y, z)
            ok = self.arm_group.go(wait=True)
            self.arm_group.stop()
            self.arm_group.clear_pose_targets()
            if not ok:
                rospy.logerr("Position-only planning/execution failed for [%s]", label)
                return False
            self.log_target_vs_current_fk([x, y, z], label=label, prefix="After 1st move")
            return True
        except Exception as e:
            rospy.logerr("move_to_position_only failed for [%s]: %s", label, str(e))
            return False

    def compensate_after_move_once(self, target_xyz, label="target"):
        try:
            best = self.find_best_compensation_once(target_xyz)
            if best is None:
                return False
            joint_map = {n: v for n, v in zip(best["joint_names"], best["joints"])}
            self.arm_group.set_start_state_to_current_state()
            self.arm_group.set_planning_time(2.0)
            self.arm_group.set_num_planning_attempts(5)
            self.arm_group.clear_pose_targets()
            self.arm_group.clear_path_constraints()
            self.arm_group.set_joint_value_target(joint_map)
            ok = self.arm_group.go(wait=True)
            self.arm_group.stop()
            self.arm_group.clear_pose_targets()
            if ok:
                self.log_target_vs_current_fk(target_xyz, label=label, prefix="After 2nd correction")
            return bool(ok)
        except Exception as e:
            rospy.logerr("compensate_after_move_once failed for [%s]: %s", label, str(e))
            return False

    def extract_plan_success_and_traj(self, plan_result):
        traj = None
        plan_success = False
        if isinstance(plan_result, tuple):
            if len(plan_result) >= 2:
                plan_success = bool(plan_result[0])
                traj = plan_result[1]
        else:
            traj = plan_result
            try:
                plan_success = traj is not None and len(traj.joint_trajectory.points) > 0
            except Exception:
                plan_success = traj is not None
        return plan_success, traj

    def execute_pose_push(self, label="target", distance_y=None):
        if distance_y is None:
            distance_y = self.pose_push_distance
        try:
            self.pre_push_joint_map = self.get_stabilized_joint_map(label=label)
            current_pose = self.get_current_tcp_pose()
            target_pose = copy.deepcopy(current_pose)
            target_pose.position.y += distance_y

            baseline = self.measure_push_current_baseline(label=label)
            if baseline is None:
                return False

            self.arm_group.set_start_state_to_current_state()
            self.arm_group.set_planning_time(self.pose_push_planning_time)
            self.arm_group.set_num_planning_attempts(self.pose_push_num_planning_attempts)
            self.arm_group.set_max_velocity_scaling_factor(self.pose_push_vel_scale)
            self.arm_group.set_max_acceleration_scaling_factor(self.pose_push_acc_scale)
            self.arm_group.set_goal_position_tolerance(self.pose_push_pos_tol)
            self.arm_group.set_goal_orientation_tolerance(self.pose_push_ori_tol)
            self.arm_group.set_goal_joint_tolerance(self.pose_push_joint_tol)
            self.arm_group.clear_pose_targets()
            self.arm_group.clear_path_constraints()
            self.arm_group.set_pose_target(target_pose, self.ee_link)

            plan_result = self.arm_group.plan()
            plan_success, traj = self.extract_plan_success_and_traj(plan_result)
            if not plan_success or traj is None:
                rospy.logwarn("Pose push planning failed for [%s]", label)
                return False

            self.arm_group.execute(traj, wait=False)
            contact_count = 0
            contact_triggered = False
            start_t = time.time()
            while not rospy.is_shutdown():
                contact, info = self.check_contact_from_current(baseline, label=label)
                if contact:
                    contact_count += 1
                else:
                    contact_count = 0
                if contact_count >= self.contact_consecutive_required:
                    contact_triggered = True
                    self.arm_group.stop()
                    break
                if time.time() - start_t > self.pose_push_timeout:
                    self.arm_group.stop()
                    break
                rospy.sleep(self.current_poll_dt)

            self.arm_group.stop()
            self.arm_group.clear_pose_targets()
            rospy.sleep(0.3)

            # 접촉 성공/실패와 관계없이 안전하게 누르기 전 자세를 거쳐 시작 자세로 복귀한다.
            # 단, 접촉 성공 여부는 별도로 반환하여 이후 버튼 상태 판단 로그와 재시도에 사용한다.
            return_result = self.move_to_saved_pre_push_then_start_pose(label=label)
            returned_to_start = bool(return_result.get("start_pose_ok", False))
            if not returned_to_start:
                rospy.logwarn("Failed to return to saved start pose after push [%s]", label)

            if contact_triggered:
                rospy.loginfo("[push_result] Current contact detected for [%s]", label)
            else:
                rospy.logwarn("[push_result] Current contact was NOT detected for [%s]", label)

            return {
                "motion_ok": True,
                "contact_detected": bool(contact_triggered),
                "returned_to_start": returned_to_start,
                "returned_to_pre_push": bool(return_result.get("pre_push_ok", False)),
            }
        except Exception as e:
            rospy.logerr("execute_pose_push failed for [%s]: %s", label, str(e))
            return False

    def move_to_target_then_compensate(self, x, y, z, label="target"):
        if not self.move_to_position_only(x, y, z, label=label):
            return {
                "motion_ok": False,
                "contact_detected": False,
                "returned_to_start": False,
            }
        target_xyz = [x, y, z]
        self.compensate_after_move_once(target_xyz, label=label)
        result = self.execute_pose_push(label=label)
        if isinstance(result, dict):
            return result
        return {
            "motion_ok": bool(result),
            "contact_detected": False,
            "returned_to_start": bool(result),
        }

    def execute_pending_target_if_requested(self):
        if self.pending_target_label is None:
            return

        cmd = self.pending_target_label.strip().upper()
        self.pending_target_label = None

        if cmd == "LIST":
            self.print_detected_targets_log()
            return

        if cmd == "F3":
            info = self.select_f3_target()
            if info is None:
                if self.f3_select_fail_reason == "need_reobserve" and not self.f3_reobserve_attempted:
                    self.f3_reobserve_attempted = True
                    self.move_to_reobserve_pose()
                self.pending_target_label = cmd
                return
        else:
            if cmd not in self.detected_targets:
                rospy.logwarn_throttle(
                    1.0,
                    "Requested label [%s] is not currently detected yet. Keep waiting...",
                    cmd
                )
                self.pending_target_label = cmd
                return
            info = dict(self.detected_targets[cmd])
            info["source"] = "direct_detected_%s" % cmd
            rospy.loginfo("[target_select] Use directly detected target: %s", cmd)

        source = str(info.get("source", "unknown"))

        rospy.loginfo(
            "RAW target [%s]: x=%.3f y=%.3f z=%.3f state=%s(%.2f) source=%s",
            cmd,
            info["x"], info["y"], info["z"],
            info.get("pressed_label", "unknown"), float(info.get("pressed_conf", 0.0)),
            source
        )

        # F3가 이미 ON처럼 보이더라도 연속 3회 이상 ON이 확인될 때만 누르지 않는다.
        if cmd == "F3" and source == "direct_f3_already_on_wait_off":
            on_confirmed = self.confirm_button_state_consecutive(
                label=cmd,
                desired_state="on",
                required_count=self.pre_button_on_required_count,
                timeout=self.pre_button_state_timeout
            )

            if on_confirmed:
                rospy.loginfo(
                    "[arm_mission] F3 ON confirmed %d consecutive times. Skip movement and wait until OFF.",
                    self.pre_button_on_required_count
                )
                off_ok = self.wait_for_button_off_without_motion(label=cmd)
                if off_ok:
                    rospy.sleep(3.0)
                    rospy.loginfo(
                        "[arm_mission] F3 OFF confirmed without motion. publish event: %s -> %s",
                        self.arm_mission_event_topic,
                        self.arm_mission_done_event
                    )
                    self.pub_arm_mission_event.publish(String(data=self.arm_mission_done_event))
                    self.stop_visualization()
                    finish_ok = self.move_to_finish_pose_after_button(label=cmd)
                    if not finish_ok:
                        rospy.logwarn("[arm_mission] finish pose move failed after already-ON wait")
                else:
                    rospy.logwarn("[arm_mission] F3 was already ON, but OFF was not confirmed. SEXY_BUTTON event not published.")
                return

            # 3회 연속 ON이 아니면 '이미 ON'으로 인정하지 않고 정상 누르기 절차를 수행한다.
            rospy.logwarn(
                "[arm_mission] F3 ON was not confirmed %d consecutive times. Proceed with button push.",
                self.pre_button_on_required_count
            )
            source = "direct_f3_on_not_stable_press"
            info["source"] = source

        x, y, z = self.apply_target_offset(info)

        rospy.loginfo(
            "FINAL target [%s]: x=%.3f y=%.3f z=%.3f after offset | source=%s",
            cmd, x, y, z, source
        )

        if cmd == "F3":
            on_confirmed = False
            attempts_executed = 0

            # 버튼 누르기 시도 후에는 접촉 성공/실패 모두 시작 자세로 복귀한다.
            # 복귀 후 버튼 상태를 다시 확인하여 ON이면 성공, OFF이면 재시도한다.
            # 상태를 확정하지 못하면 안전을 위해 추가 누르기를 중단한다.
            for attempt in range(1, max(1, self.button_push_max_attempts) + 1):
                attempts_executed = attempt
                rospy.loginfo(
                    "[arm_mission] F3 push attempt %d/%d",
                    attempt, max(1, self.button_push_max_attempts)
                )

                push_result = self.move_to_target_then_compensate(x, y, z, label=cmd)

                if not push_result.get("motion_ok", False):
                    rospy.logwarn(
                        "[arm_mission] F3 push motion failed on attempt %d. Retry if attempts remain.",
                        attempt
                    )
                    continue

                if not push_result.get("returned_to_start", False):
                    rospy.logwarn(
                        "[arm_mission] F3 did not return to start pose after attempt %d. Stop retry for safety.",
                        attempt
                    )
                    break

                if push_result.get("contact_detected", False):
                    rospy.loginfo(
                        "[arm_mission] F3 contact detected on attempt %d. Check button state after return.",
                        attempt
                    )
                else:
                    rospy.logwarn(
                        "[arm_mission] F3 contact not detected on attempt %d. Check button state after return.",
                        attempt
                    )

                rospy.sleep(2.0)

                stable_state = self.detect_stable_button_state(
                    label=cmd,
                    required_count=self.button_state_required_count,
                    timeout=self.button_on_confirm_timeout
                )

                if stable_state == "on":
                    on_confirmed = True
                    rospy.loginfo(
                        "[arm_mission] F3 ON confirmed after attempt %d. contact_detected=%s",
                        attempt, str(push_result.get("contact_detected", False))
                    )
                    break

                if stable_state == "off":
                    rospy.logwarn(
                        "[arm_mission] F3 is still OFF after attempt %d. Retry button push if attempts remain.",
                        attempt
                    )
                    continue

                rospy.logwarn(
                    "[arm_mission] F3 state is unknown after attempt %d. Stop retry to avoid an unintended extra press.",
                    attempt
                )
                break

            if not on_confirmed:
                rospy.logwarn(
                    "[arm_mission] F3 ON was not confirmed after %d executed push attempt(s). SEXY_BUTTON event not published.",
                    attempts_executed
                )
                return

            # ON은 이미 확인했으므로 이후 OFF 전환만 기다린다.
            off_ok = self.wait_for_button_off_after_on_confirmed(label=cmd)
            if off_ok:
                rospy.sleep(3.0)
                rospy.loginfo(
                    "[arm_mission] F3 ON->OFF confirmed. publish event: %s -> %s",
                    self.arm_mission_event_topic,
                    self.arm_mission_done_event
                )
                self.pub_arm_mission_event.publish(String(data=self.arm_mission_done_event))
                self.stop_visualization()
                finish_ok = self.move_to_finish_pose_after_button(label=cmd)
                if not finish_ok:
                    rospy.logwarn("[arm_mission] finish pose move failed after SEXY_BUTTON publish")
            else:
                rospy.logwarn("[arm_mission] F3 OFF was not confirmed after ON. SEXY_BUTTON event not published.")

        else:
            self.move_to_target_then_compensate(x, y, z, label=cmd)

    def run(self):
        rate = rospy.Rate(self.loop_hz)
        while not rospy.is_shutdown():
            try:
                if not self.camera_ready():
                    rospy.logwarn_throttle(2.0, "Waiting for camera topics: color/depth/camera_info")
                    rate.sleep()
                    continue

                color_image = self.latest_color_image.copy()
                depth_image = self.latest_depth_image.copy()
                dbg = color_image.copy()
                stamp = self.latest_color_stamp if self.latest_color_stamp is not None else rospy.Time.now()

                detect_result = None
                if self.detection_paused:
                    rospy.loginfo_throttle(0.5, "[detection] paused while arm is moving for re-observation")
                else:
                    detect_result = self.detect_model.predict(
                        source=color_image,
                        conf=self.detect_conf_thresh,
                        iou=self.detect_iou_thresh,
                        verbose=False
                    )[0]

                if (not self.detection_paused) and detect_result is not None and detect_result.boxes is not None and len(detect_result.boxes) > 0:
                    names = detect_result.names
                    for box in detect_result.boxes:
                        cls_id = int(box.cls[0].item())
                        label = str(names.get(cls_id, str(cls_id)))
                        conf = float(box.conf[0].item())
                        if self.target_class and label != self.target_class:
                            continue

                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                        cx_pix = int((x1 + x2) / 2)
                        cy_pix = int((y1 + y2) / 2)

                        z = self.get_depth_robust(depth_image, cx_pix, cy_pix, r=self.depth_patch_radius)
                        if z <= 0.0:
                            continue

                        X, Y, Z = self.pixel_to_camera_xyz(cx_pix, cy_pix, z)
                        base_pt = self.transform_point_to_base(X, Y, Z, stamp)
                        if base_pt is None:
                            continue
                        bx, by, bz = base_pt.point.x, base_pt.point.y, base_pt.point.z

                        pressed_label, pressed_conf = self.classify_pressed_state(color_image, x1, y1, x2, y2)
                        color = (0, 255, 0) if pressed_label == "on" else (0, 0, 255)
                        if pressed_label == "unknown":
                            color = (0, 255, 255)

                        cv2.rectangle(dbg, (x1, y1), (x2, y2), color, 2)
                        text = "{} {}({:.2f}) z={:.2f}m".format(label, pressed_label.upper(), pressed_conf, Z)
                        cv2.putText(dbg, text, (x1, max(20, y1 - 10)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

                        self.update_detected_target(label, bx, by, bz, conf, stamp, pressed_label, pressed_conf)

                self.clear_stale_targets(stamp)
                self.execute_pending_target_if_requested()

                y0 = 25
                cv2.putText(dbg, "Latest detected labels + ON/OFF:", (20, y0),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)
                y0 += 25
                for label_key, info in sorted(self.detected_targets.items()):
                    txt = "{} [{}({:.2f})] -> ({:.3f}, {:.3f}, {:.3f})".format(
                        label_key, info["pressed_label"], info["pressed_conf"],
                        info["x"], info["y"], info["z"]
                    )
                    cv2.putText(dbg, txt, (20, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
                    y0 += 22

                if self.view_image:
                    cv2.imshow("YOLO Detect + PressedClassifier + MoveIt", dbg)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        rospy.signal_shutdown("User requested shutdown")
                        break

                rate.sleep()
            except rospy.ROSInterruptException:
                break
            except Exception as e:
                rospy.logerr_throttle(1.0, "Runtime error: %s", str(e))
                rate.sleep()


    def stop_visualization(self):
        """
        카메라 시각화 창을 닫고 이후 imshow/waitKey 호출을 중단한다.
        카메라 토픽 구독과 객체 인식 루프는 유지한다.
        """
        self.view_image = False
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass

    def shutdown_hook(self):
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass
        try:
            if self.enable_moveit:
                moveit_commander.roscpp_shutdown()
        except Exception:
            pass


if __name__ == "__main__":
    node = YoloObjectTFDetector()
    node.run()
