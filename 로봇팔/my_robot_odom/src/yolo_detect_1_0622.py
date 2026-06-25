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


def normalize_label_text(s):
    """
    YOLO class name / command label 비교용 정규화.
    공백, 개행, 따옴표, 숨은 문자 등이 섞여도 F3 == f3 == " F3 "로 비교되게 한다.
    """
    s = str(s)
    s = s.replace("\n", "").replace("\r", "").replace("\t", "")
    s = s.strip().upper()
    return "".join(ch for ch in s if ch.isalnum())


class YoloObjectTFDetector:
    def __init__(self):
        rospy.init_node("yolo_object_tf_detector", anonymous=False)

        # YOLO models
        self.detect_model_path = rospy.get_param("~detect_model_path", "/home/inwoong/catkin_ws/best1.pt")
        self.pressed_model_path = rospy.get_param("~pressed_model_path", "/home/inwoong/catkin_ws/best2.pt")

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

        self.target_offset_x = float(rospy.get_param("~target_offset_x", 0.00))
        self.target_offset_y = float(rospy.get_param("~target_offset_y", -0.01))
        self.target_offset_z = float(rospy.get_param("~target_offset_z", 0.01))

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
        self.settle_initial_wait = 1.0
        self.settle_num_samples = 3
        self.settle_sample_interval = 0.03
        self.use_median_joint_sampling = True

        self.level_roll_ref = -0.088
        self.comp_q2_span = 0.30
        self.comp_q3_span = 0.30
        self.comp_step = 0.02
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
        self.detected_targets[key] = {
            "x": float(bx), "y": float(by), "z": float(bz),
            "conf": float(conf), "stamp": stamp,
            "pressed_label": str(pressed_label), "pressed_conf": float(pressed_conf)
        }

    def clear_stale_targets(self, now):
        for k in list(self.detected_targets.keys()):
            if (now - self.detected_targets[k]["stamp"]).to_sec() > self.target_timeout:
                del self.detected_targets[k]

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
        현재 color image 한 프레임에서 target_label의 ON/OFF 상태를 확인한다.
        push 이후 이벤트 판단용이며, depth/TF/MoveIt 동작은 수행하지 않는다.

        추가 기능:
          - button_state 확인 중에도 OpenCV 시각화 화면을 계속 갱신한다.
          - 현재 프레임에서 검출된 모든 라벨/ confidence를 로그로 출력한다.
          - 라벨 문자열을 정규화해서 F3가 검출됐는데도 비교 실패하는 문제를 방지한다.
        """
        if self.latest_color_image is None:
            rospy.logwarn_throttle(0.5, "[button_state] latest_color_image is None")
            return None, 0.0

        target_key_raw = str(target_label)
        target_key = normalize_label_text(target_key_raw)

        try:
            color_image = self.latest_color_image.copy()
            dbg = color_image.copy()

            detect_result = self.detect_model.predict(
                source=color_image,
                conf=self.detect_conf_thresh,
                iou=self.detect_iou_thresh,
                verbose=False
            )[0]

            detected_labels = []
            best_state = None
            best_conf = 0.0
            best_det_conf = 0.0

            if detect_result is None or detect_result.boxes is None or len(detect_result.boxes) <= 0:
                cv2.putText(
                    dbg,
                    "button_state: NO DETECTION BOXES",
                    (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 255),
                    2
                )
                rospy.logwarn_throttle(0.5, "[button_state] no detection boxes. target=%s", target_key)

                if self.view_image:
                    cv2.imshow("YOLO Detect + PressedClassifier + MoveIt", dbg)
                    cv2.waitKey(1)

                return None, 0.0

            names = detect_result.names

            for box in detect_result.boxes:
                cls_id = int(box.cls[0].item())
                raw_label = str(names.get(cls_id, str(cls_id)))
                label = raw_label.replace("\n", "").replace("\r", "").replace("\t", "").strip().upper()
                label_key = normalize_label_text(label)
                det_conf = float(box.conf[0].item())

                detected_labels.append("%s|norm=%s(%.2f)" % (label, label_key, det_conf))

                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)

                # 먼저 검출 bbox를 모두 화면에 표시한다.
                color = (180, 180, 180)
                if label_key == target_key:
                    color = (0, 255, 0)

                pressed_label = "-"
                pressed_conf = 0.0

                # target label인 경우에만 best2.pt로 ON/OFF 분류한다.
                if label_key == target_key:
                    pressed_label, pressed_conf = self.classify_pressed_state(color_image, x1, y1, x2, y2)

                    if pressed_conf > best_conf:
                        best_state = str(pressed_label).strip().lower()
                        best_conf = float(pressed_conf)
                        best_det_conf = det_conf

                    if str(pressed_label).lower() == "on":
                        color = (0, 255, 0)
                    elif str(pressed_label).lower() == "off":
                        color = (0, 0, 255)
                    else:
                        color = (0, 255, 255)

                cv2.rectangle(dbg, (x1, y1), (x2, y2), color, 2)
                txt = "%s det=%.2f state=%s(%.2f)" % (label, det_conf, str(pressed_label).upper(), pressed_conf)
                cv2.putText(
                    dbg,
                    txt,
                    (x1, max(20, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    color,
                    2
                )

            # button_state 확인 중에도 화면이 멈추지 않도록 현재 프레임을 계속 갱신한다.
            status_txt = "button_state target=%s best=%s(%.2f) det_conf=%.2f" % (
                target_key,
                str(best_state),
                best_conf,
                best_det_conf
            )
            cv2.putText(
                dbg,
                status_txt,
                (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 0),
                2
            )
            cv2.putText(
                dbg,
                "detected: " + ", ".join(detected_labels[:4]),
                (20, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 0),
                2
            )

            if self.view_image:
                cv2.imshow("YOLO Detect + PressedClassifier + MoveIt", dbg)
                cv2.waitKey(1)

            if best_state is None:
                rospy.logwarn_throttle(
                    0.5,
                    "[button_state] target %s not found. detected=%s",
                    target_key,
                    ", ".join(detected_labels)
                )
                return None, 0.0

            rospy.loginfo_throttle(
                0.3,
                "[button_state] target %s detected. state=%s cls_conf=%.2f det_conf=%.2f detected=%s",
                target_key,
                str(best_state),
                best_conf,
                best_det_conf,
                ", ".join(detected_labels)
            )

            return best_state, best_conf

        except Exception as e:
            rospy.logwarn_throttle(1.0, "detect_button_state_once failed: %s", str(e))
            return None, 0.0

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
            ok_pre = self.move_to_joint_map(self.pre_push_joint_map, label=label, prefix="Move back to pre-push pose")
            rospy.sleep(0.5)
        ok_start = self.move_to_joint_map(self.start_pose_joint_map, label=label, prefix="Move back to saved start pose")
        return ok_pre and ok_start

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
            if contact_triggered:
                self.move_to_saved_pre_push_then_start_pose(label=label)
            return True
        except Exception as e:
            rospy.logerr("execute_pose_push failed for [%s]: %s", label, str(e))
            return False

    def move_to_target_then_compensate(self, x, y, z, label="target"):
        if not self.move_to_position_only(x, y, z, label=label):
            return False
        target_xyz = [x, y, z]
        self.compensate_after_move_once(target_xyz, label=label)
        return self.execute_pose_push(label=label)

    def execute_pending_target_if_requested(self):
        if self.pending_target_label is None:
            return
        cmd = self.pending_target_label.strip().upper()
        self.pending_target_label = None
        if cmd == "LIST":
            self.print_detected_targets_log()
            return
        if cmd not in self.detected_targets:
            rospy.logwarn_throttle(1.0, "Requested label [%s] is not currently detected yet. Keep waiting...", cmd)
            self.pending_target_label = cmd
            return
        info = self.detected_targets[cmd]
        rospy.loginfo("RAW target [%s]: x=%.3f y=%.3f z=%.3f state=%s(%.2f)",
                      cmd, info["x"], info["y"], info["z"], info["pressed_label"], info["pressed_conf"])
        x = info["x"] + self.target_offset_x
        y = info["y"] + self.target_offset_y
        z = info["z"] + self.target_offset_z

        mission_ok = self.move_to_target_then_compensate(x, y, z, label=cmd)
        
        
        rospy.sleep(2.0)
        # F3 버튼은 push 성공 직후 바로 완료 처리하지 않는다.
        # push 후 F3 상태가 ON이 된 것을 확인하고, 이후 OFF로 변경될 때 SEXY_BUTTON을 publish한다.
        if cmd == "F3" and mission_ok:
            transition_ok = self.wait_for_button_on_to_off_after_push(label=cmd)
            if transition_ok:
                rospy.sleep(3.0)
                rospy.loginfo(
                    "[arm_mission] F3 ON->OFF confirmed. publish event: %s -> %s",
                    self.arm_mission_event_topic,
                    self.arm_mission_done_event
                )
                self.pub_arm_mission_event.publish(String(data=self.arm_mission_done_event))

                # SEXY_BUTTON 전송과 별개로, 전송 직후 마무리 자세로 복귀한다.
                finish_ok = self.move_to_finish_pose_after_button(label=cmd)
                if not finish_ok:
                    rospy.logwarn("[arm_mission] finish pose move failed after SEXY_BUTTON publish")
            else:
                rospy.logwarn("[arm_mission] F3 ON->OFF transition not confirmed. SEXY_BUTTON event not published.")
        elif cmd == "F3" and not mission_ok:
            rospy.logwarn("[arm_mission] F3 mission failed. SEXY_BUTTON event not published.")

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

                detect_result = self.detect_model.predict(
                    source=color_image,
                    conf=self.detect_conf_thresh,
                    iou=self.detect_iou_thresh,
                    verbose=False
                )[0]

                if detect_result is not None and detect_result.boxes is not None and len(detect_result.boxes) > 0:
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
