#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import copy
import cv2
import math
import time
import statistics
import rospy
import logging
import numpy as np
import pyrealsense2 as rs
import moveit_commander
import tf2_ros
import tf2_geometry_msgs

from ultralytics import YOLO
from geometry_msgs.msg import TransformStamped, PointStamped
from std_msgs.msg import String, Float64MultiArray
from moveit_msgs.srv import GetPositionFK, GetPositionFKRequest
from tf.transformations import euler_from_quaternion


class YoloObjectTFDetector:
    def __init__(self):
        rospy.init_node("yolo_object_tf_detector", anonymous=False)

        # =========================================================
        # 직접 코드에서 설정
        # =========================================================
        self.model_path = "/home/inwoong/catkin_ws/best1.pt"
        self.target_class = ""   # ""이면 전체 클래스 허용
        self.camera_frame = "camera_color_optical_frame"
        self.base_frame = "base_link"
        self.child_frame_prefix = "yolo_object"
        self.conf_thresh = 0.25
        self.view_image = True
        self.loop_hz = 30.0

        # depth
        self.use_depth_patch = True
        self.depth_patch_radius = 2

        # latest target timeout
        self.target_timeout = 5.0   # 초

        # =========================================================
        # MoveIt
        # =========================================================
        self.enable_moveit = True
        self.move_group_name = "arm"
        self.ee_link = "brk9_1"   # TCP 링크 명시

        self.planning_time = 5.0
        self.num_planning_attempts = 12
        self.max_vel_scale = 0.2
        self.max_acc_scale = 0.2

        # Pose push (버튼 누르기 준비용)
        # 일반 pose planning으로 현재 자세를 기준으로 +y 방향 목표점을 만들어 전진
        # TCP가 완전히 직선이 아니어도 앞으로 가는 동작을 우선시함
        self.pose_push_distance = 0.04
        self.pose_back_distance = 0.04
        self.pose_push_pos_tol = 0.010
        self.pose_push_ori_tol = 0.35
        self.pose_push_joint_tol = 0.05
        self.pose_push_vel_scale = 0.01
        self.pose_push_acc_scale = 0.01
        self.pose_push_planning_time = 2.0
        self.pose_push_num_planning_attempts = 10

        # 현재 실험으로 맞춘 위치 보정
        self.target_offset_x = -0.05
        self.target_offset_y = -0.02
        self.target_offset_z = 0.02


        # =========================================================
        # Push current monitor
        # =========================================================
        self.current_topic = "/arm/joint_current_raw"
        self.latest_joint_current_raw = None
        self.current_msg_time = None
        self.current_idx_q2 = 1
        self.current_idx_q3 = 2

        self.current_baseline_duration = 0.5
        self.current_baseline_dt = 0.05
        self.current_poll_dt = 0.02

        # q3는 메인(상승량), q2는 보조(절대 변화량)
        self.contact_q3_delta_th = 15.0
        self.contact_q2_abs_delta_th = 40.0
        self.contact_consecutive_required = 2

        # push 중 watchdog
        self.pose_push_timeout = 6.0

        # push 직전(pre-push) 자세 저장 후
        # 접촉 시 먼저 이 자세로 복귀한 뒤 초기 자세로 이동
        self.pre_push_joint_map = None

        self.start_pose_joint_map = {
            "Revolute1": 0.0024,
            "Revolute2": -0.2436,
            "Revolute3": -1.1026,
            "Revolute4": 0.0127,
            "Revolute5": 1.4589,
        }

        # =========================================================
        # FK 기반 목표-현재 TCP 비교 옵션
        # target_xyz(카메라→base 변환 좌표)와
        # 현재 joint로 계산한 FK TCP 좌표를 비교
        # dx, dy, dz = target - current_fk
        # =========================================================
        self.enable_fk_error_check = True
        self.fk_error_warn_threshold = 0.03   # 3 cm 이상이면 경고

        # joint_states 안정화 후 중앙값 기반 FK 비교
        self.settle_initial_wait = 1.5
        self.settle_num_samples = 5
        self.settle_sample_interval = 0.03
        self.use_median_joint_sampling = True

        # =========================================================
        # 수평 기준값
        # 수평 자세에서 측정한 기준 roll
        # q2 + q3 + q5 ≈ roll 관계를 사용
        # =========================================================
        self.level_roll_ref = -0.088

        # =========================================================
        # 한 번에 보정 후보 계산용 파라미터
        # q2, q3 후보를 만들고 q5는 계산식으로 바로 결정
        # =========================================================
        self.comp_q2_span = 0.30
        self.comp_q3_span = 0.30
        self.comp_step = 0.02

        # 위치 유지 우선, 그 다음 수평
        self.comp_pos_weight = 40.0
        self.comp_roll_weight = 1.0
        self.comp_z_drop_weight = 60.0

        # q5는 과보정 방지를 위해 계산값을 전부 쓰지 않고 일부만 반영
        self.comp_q5_gain = 0.60

        # 단계적 완화 탐색: 처음엔 z 하강을 강하게 제한하고,
        # 후보가 부족하면 조금씩 완화하되 보정은 스킵하지 않음
        self.comp_adaptive_stages = [
            {"name": "strict",  "q2_span_scale": 1.00, "q3_span_scale": 1.00, "q5_gain": 0.60, "z_drop_limit": 0.010},
            {"name": "medium",  "q2_span_scale": 1.15, "q3_span_scale": 1.15, "q5_gain": 0.70, "z_drop_limit": 0.015},
            {"name": "relaxed", "q2_span_scale": 1.30, "q3_span_scale": 1.30, "q5_gain": 0.80, "z_drop_limit": 0.020},
        ]

        # 너무 과한 개선 요구는 보정 누락으로 이어질 수 있으므로,
        # 단계 탐색 이후에는 improvement가 작아도 가장 덜 나쁜 후보를 사용
        self.comp_min_score_improvement = 0.0

        # joint limits (xacro 기준)
        self.joint_limits = {
            "Revolute1": (-3.141593,  3.141593),
            "Revolute2": (-1.832596,  1.832596),
            "Revolute3": (-1.483530,  1.989675),
            "Revolute4": (-3.141593,  3.141593),
            "Revolute5": (-1.989675,  1.989675),
        }

        # 명령 토픽
        self.command_topic = "/move_target_label"

        # =========================================================
        # 상태 변수
        # =========================================================
        self.detected_targets = {}
        self.pending_target_label = None
        self.last_fk_compare_result = None

        # =========================================================
        # YOLO
        # =========================================================
        if not os.path.exists(self.model_path):
            rospy.logerr("YOLO model not found: %s", self.model_path)
            raise FileNotFoundError(self.model_path)

        rospy.loginfo("Loading YOLO model: %s", self.model_path)
        self.model = YOLO(self.model_path)
        logging.getLogger("ultralytics").setLevel(logging.ERROR)
        rospy.loginfo("YOLO model loaded.")

        # =========================================================
        # TF
        # =========================================================
        self.tf_broadcaster = tf2_ros.TransformBroadcaster()
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        # =========================================================
        # command subscriber
        # =========================================================
        self.cmd_sub = rospy.Subscriber(
            self.command_topic,
            String,
            self.command_callback,
            queue_size=10
        )

        self.current_sub = rospy.Subscriber(
            self.current_topic,
            Float64MultiArray,
            self.current_callback,
            queue_size=20
        )

        # =========================================================
        # MoveIt
        # =========================================================
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

                if self.ee_link:
                    self.arm_group.set_end_effector_link(self.ee_link)

                rospy.loginfo("MoveIt initialized. group=%s base_frame=%s",
                              self.move_group_name, self.base_frame)
                rospy.loginfo("MoveIt eef_link=%s", self.arm_group.get_end_effector_link())
                rospy.loginfo("Level roll reference = %.4f rad", self.level_roll_ref)

                rospy.wait_for_service('/compute_fk', timeout=5.0)
                self.fk_srv = rospy.ServiceProxy('/compute_fk', GetPositionFK)
                rospy.loginfo("Connected to /compute_fk service")

            except Exception as e:
                rospy.logerr("Failed to initialize MoveIt/FK: %s", str(e))
                self.enable_moveit = False
                self.arm_group = None
                self.fk_srv = None

        # =========================================================
        # RealSense
        # =========================================================
        self.pipeline = rs.pipeline()
        self.config = rs.config()

        self.config.enable_stream(rs.stream.depth, 848, 480, rs.format.z16, 30)
        self.config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)

        rospy.loginfo("Starting RealSense pipeline...")
        self.profile = self.pipeline.start(self.config)
        self.align = rs.align(rs.stream.color)

        color_stream = self.profile.get_stream(rs.stream.color).as_video_stream_profile()
        self.color_intrinsics = color_stream.get_intrinsics()

        rospy.loginfo(
            "RealSense started. Color intrinsics: fx=%.3f fy=%.3f ppx=%.3f ppy=%.3f",
            self.color_intrinsics.fx,
            self.color_intrinsics.fy,
            self.color_intrinsics.ppx,
            self.color_intrinsics.ppy
        )

        rospy.loginfo("Command topic ready: %s", self.command_topic)
        rospy.loginfo("Example:")
        rospy.loginfo("  rostopic pub -1 %s std_msgs/String \"data: 'f2'\"", self.command_topic)
        rospy.loginfo("  rostopic pub -1 %s std_msgs/String \"data: 'list'\"", self.command_topic)

        rospy.on_shutdown(self.shutdown_hook)

    # =========================================================
    # 공통 유틸
    # =========================================================
    def clamp(self, v, lo, hi):
        return max(lo, min(hi, v))

    def norm3(self, a, b):
        return math.sqrt(
            (a[0] - b[0]) ** 2 +
            (a[1] - b[1]) ** 2 +
            (a[2] - b[2]) ** 2
        )

    def frange(self, start, stop, step):
        vals = []
        v = start
        while v <= stop + 1e-9:
            vals.append(round(v, 6))
            v += step
        return vals

    def score_comp_candidate(self, pos_err, roll_err, z_drop=0.0):
        return (
            self.comp_pos_weight * pos_err +
            self.comp_roll_weight * roll_err +
            self.comp_z_drop_weight * z_drop
        )

    def pose_to_xyz(self, pose):
        return [pose.position.x, pose.position.y, pose.position.z]

    def compute_xyz_error(self, target_xyz, current_xyz):
        dx = target_xyz[0] - current_xyz[0]
        dy = target_xyz[1] - current_xyz[1]
        dz = target_xyz[2] - current_xyz[2]
        dist = math.sqrt(dx * dx + dy * dy + dz * dz)
        return {
            "dx": dx,
            "dy": dy,
            "dz": dz,
            "dist": dist
        }

    def format_joint_values(self, joint_names, joint_values):
        return ", ".join(["%s=%.4f" % (n, v) for n, v in zip(joint_names, joint_values)])

    # =========================================================
    # command callback
    # =========================================================
    def command_callback(self, msg):
        label = msg.data.strip().upper()
        if label == "":
            return

        self.pending_target_label = label
        rospy.loginfo("Received target label command: %s", label)

    # =========================================================
    # 현재 저장된 타겟 출력
    # =========================================================
    def print_detected_targets_log(self):
        if len(self.detected_targets) == 0:
            rospy.loginfo("No detected targets stored.")
            return

        rospy.loginfo("===== Stored latest targets =====")
        for label, info in sorted(self.detected_targets.items()):
            rospy.loginfo(
                "[TARGET] %s -> x=%.3f y=%.3f z=%.3f conf=%.2f",
                label, info["x"], info["y"], info["z"], info["conf"]
            )

    # =========================================================
    # Depth
    # =========================================================
    def get_depth_robust(self, depth_frame, u, v, r=2):
        if depth_frame is None:
            return 0.0

        if not self.use_depth_patch:
            return depth_frame.get_distance(u, v)

        h = depth_frame.get_height()
        w = depth_frame.get_width()

        vals = []
        for yy in range(max(0, v - r), min(h, v + r + 1)):
            for xx in range(max(0, u - r), min(w, u + r + 1)):
                d = depth_frame.get_distance(xx, yy)
                if d > 0:
                    vals.append(d)

        if len(vals) == 0:
            return 0.0

        return float(np.median(vals))

    # =========================================================
    # TF 변환
    # =========================================================
    def transform_point_to_base(self, x, y, z, stamp):
        p = PointStamped()
        p.header.stamp = stamp
        p.header.frame_id = self.camera_frame
        p.point.x = float(x)
        p.point.y = float(y)
        p.point.z = float(z)

        try:
            p_base = self.tf_buffer.transform(p, self.base_frame, rospy.Duration(0.5))
            return p_base
        except (tf2_ros.LookupException,
                tf2_ros.ConnectivityException,
                tf2_ros.ExtrapolationException) as e:
            rospy.logwarn_throttle(1.0, "Transform to %s failed: %s", self.base_frame, str(e))
            return None

    # =========================================================
    # TF publish
    # =========================================================
    def publish_tf(self, x, y, z, stamp, child_frame_id):
        t = TransformStamped()
        t.header.stamp = stamp
        t.header.frame_id = self.camera_frame
        t.child_frame_id = child_frame_id

        t.transform.translation.x = float(x)
        t.transform.translation.y = float(y)
        t.transform.translation.z = float(z)

        t.transform.rotation.x = 0.0
        t.transform.rotation.y = 0.0
        t.transform.rotation.z = 0.0
        t.transform.rotation.w = 1.0

        self.tf_broadcaster.sendTransform(t)

    # =========================================================
    # 최신 좌표 저장
    # =========================================================
    def update_detected_target(self, label, bx, by, bz, conf, stamp):
        label_key = str(label).strip().upper()

        self.detected_targets[label_key] = {
            "x": float(bx),
            "y": float(by),
            "z": float(bz),
            "conf": float(conf),
            "stamp": stamp
        }

    def clear_stale_targets(self, now):
        remove_keys = []
        for label, info in self.detected_targets.items():
            age = (now - info["stamp"]).to_sec()
            if age > self.target_timeout:
                remove_keys.append(label)

        for k in remove_keys:
            del self.detected_targets[k]

    # =========================================================
    # FK 계산
    # =========================================================
    def get_fk_pose(self, joint_names, joint_positions, fk_link):
        if self.fk_srv is None:
            rospy.logerr("FK service is not ready.")
            return None

        try:
            req = GetPositionFKRequest()
            req.header.frame_id = self.base_frame
            req.fk_link_names = [fk_link]
            req.robot_state.joint_state.name = list(joint_names)
            req.robot_state.joint_state.position = list(joint_positions)

            res = self.fk_srv(req)

            if res.error_code.val != 1:
                return None
            if len(res.pose_stamped) == 0:
                return None

            return res.pose_stamped[0].pose
        except Exception as e:
            rospy.logwarn("FK call failed: %s", str(e))
            return None

    # =========================================================
    # 현재 joint map / pose / roll
    # =========================================================
    def get_current_joint_map(self):
        joint_names = self.arm_group.get_active_joints()
        joint_values = self.arm_group.get_current_joint_values()
        return joint_names, joint_values, {n: i for i, n in enumerate(joint_names)}

    def get_current_tcp_pose(self):
        return self.arm_group.get_current_pose(self.ee_link).pose

    def get_roll_from_pose(self, pose):
        q = [
            pose.orientation.x,
            pose.orientation.y,
            pose.orientation.z,
            pose.orientation.w
        ]
        roll, pitch, yaw = euler_from_quaternion(q)
        return roll, pitch, yaw

    def collect_joint_value_samples(self, initial_wait=None, num_samples=None, sample_interval=None):
        if not self.enable_moveit or self.arm_group is None:
            return None

        wait_time = self.settle_initial_wait if initial_wait is None else initial_wait
        sample_count = self.settle_num_samples if num_samples is None else num_samples
        interval = self.settle_sample_interval if sample_interval is None else sample_interval

        if wait_time > 0.0:
            rospy.loginfo("Waiting %.2f s for joint stabilization before FK comparison.", wait_time)
            rospy.sleep(wait_time)

        joint_names = None
        samples = []

        for idx in range(sample_count):
            names, values, _ = self.get_current_joint_map()
            if joint_names is None:
                joint_names = list(names)
            elif list(names) != joint_names:
                rospy.logwarn("Joint name order changed during sampling. Abort stabilized FK sampling.")
                return None

            samples.append(list(values))

            if idx < sample_count - 1 and interval > 0.0:
                rospy.sleep(interval)

        if joint_names is None or len(samples) == 0:
            return None

        arr = np.array(samples, dtype=np.float64)
        median_values = np.median(arr, axis=0).tolist()

        return {
            "joint_names": joint_names,
            "samples": samples,
            "median_values": median_values,
        }

    def sample_stabilized_joints(self, label="target", prefix="Sampling stabilized joints", initial_wait=None, num_samples=None, sample_interval=None):
        sample_info = self.collect_joint_value_samples(
            initial_wait=initial_wait,
            num_samples=num_samples,
            sample_interval=sample_interval
        )
        if sample_info is None:
            rospy.logwarn("%s [%s] failed: no stabilized joint samples.", prefix, label)
            return None, None

        rospy.loginfo("%s [%s]: wait=%.2f s, samples=%d, dt=%.2f s",
                      prefix, label,
                      self.settle_initial_wait if initial_wait is None else initial_wait,
                      self.settle_num_samples if num_samples is None else num_samples,
                      self.settle_sample_interval if sample_interval is None else sample_interval)

        for idx, sample in enumerate(sample_info["samples"], start=1):
            rospy.loginfo("%s [%s] joint sample %d/%d: %s",
                          prefix, label, idx, len(sample_info["samples"]),
                          self.format_joint_values(sample_info["joint_names"], sample))

        rospy.loginfo("%s [%s] median joints: %s",
                      prefix, label,
                      self.format_joint_values(sample_info["joint_names"], sample_info["median_values"]))

        return list(sample_info["joint_names"]), list(sample_info["median_values"])

    def get_current_fk_tcp_info(self, use_stabilized=False):
        if not self.enable_moveit or self.arm_group is None:
            return None

        sample_info = None
        if use_stabilized:
            sample_info = self.collect_joint_value_samples()
            if sample_info is None:
                return None
            joint_names = sample_info["joint_names"]
            joint_values = sample_info["median_values"]
        else:
            joint_names, joint_values, _ = self.get_current_joint_map()

        fk_pose = self.get_fk_pose(joint_names, joint_values, self.ee_link)
        if fk_pose is None:
            return None

        return {
            "joint_names": joint_names,
            "joint_values": joint_values,
            "pose": fk_pose,
            "xyz": self.pose_to_xyz(fk_pose),
            "sample_info": sample_info,
        }

    def get_joint_seed_for_compensation(self):
        if self.use_median_joint_sampling:
            sample_info = self.collect_joint_value_samples()
            if sample_info is not None:
                return sample_info["joint_names"], sample_info["median_values"], {n: i for i, n in enumerate(sample_info["joint_names"])}, sample_info

        joint_names, joint_values, name_to_idx = self.get_current_joint_map()
        return joint_names, joint_values, name_to_idx, None

    def log_target_vs_current_fk(self, target_xyz, label="target", prefix="FK compare", use_stabilized=False):
        if not self.enable_fk_error_check:
            return None

        fk_info = self.get_current_fk_tcp_info(use_stabilized=use_stabilized)
        if fk_info is None:
            rospy.logwarn("%s [%s] failed: current FK TCP pose is unavailable.", prefix, label)
            return None

        err = self.compute_xyz_error(target_xyz, fk_info["xyz"])
        roll, pitch, yaw = self.get_roll_from_pose(fk_info["pose"])

        if use_stabilized and fk_info.get("sample_info") is not None:
            sample_info = fk_info["sample_info"]
            rospy.loginfo("%s [%s] using stabilized median joints (wait=%.2fs, samples=%d, dt=%.2fs)",
                          prefix, label,
                          self.settle_initial_wait,
                          self.settle_num_samples,
                          self.settle_sample_interval)
            for idx, sample in enumerate(sample_info["samples"], start=1):
                rospy.loginfo("%s [%s] joint sample %d/%d: %s",
                              prefix, label, idx, len(sample_info["samples"]),
                              self.format_joint_values(sample_info["joint_names"], sample))
            rospy.loginfo("%s [%s] median joints: %s",
                          prefix, label,
                          self.format_joint_values(sample_info["joint_names"], sample_info["median_values"]))

        rospy.loginfo("%s [%s] target(base): x=%.4f y=%.4f z=%.4f",
                      prefix, label, target_xyz[0], target_xyz[1], target_xyz[2])
        rospy.loginfo("%s [%s] FK tcp(base): x=%.4f y=%.4f z=%.4f",
                      prefix, label, fk_info["xyz"][0], fk_info["xyz"][1], fk_info["xyz"][2])
        rospy.loginfo("%s [%s] delta(target-fk): dx=%.4f dy=%.4f dz=%.4f | norm=%.4f m",
                      prefix, label, err["dx"], err["dy"], err["dz"], err["dist"])
        rospy.loginfo("%s [%s] FK tcp RPY: roll=%.4f pitch=%.4f yaw=%.4f",
                      prefix, label, roll, pitch, yaw)
        rospy.loginfo("%s [%s] current joints: %s",
                      prefix, label, self.format_joint_values(fk_info["joint_names"], fk_info["joint_values"]))

        if err["dist"] >= self.fk_error_warn_threshold:
            rospy.logwarn("%s [%s] position error is large: %.4f m (threshold=%.4f m)",
                          prefix, label, err["dist"], self.fk_error_warn_threshold)

        self.last_fk_compare_result = {
            "label": label,
            "prefix": prefix,
            "target_xyz": list(target_xyz),
            "fk_xyz": list(fk_info["xyz"]),
            "dx": err["dx"],
            "dy": err["dy"],
            "dz": err["dz"],
            "dist": err["dist"],
            "joint_names": list(fk_info["joint_names"]),
            "joint_values": list(fk_info["joint_values"]),
            "use_stabilized": use_stabilized
        }
        return self.last_fk_compare_result

    # =========================================================
    # 1차: position_only_ik 기반 위치 이동
    # =========================================================
    def move_to_position_only(self, x, y, z, label="target"):
        if not self.enable_moveit or self.arm_group is None:
            rospy.logwarn("MoveIt disabled. Skip command for %s", label)
            return False

        try:
            eef_link = self.arm_group.get_end_effector_link()
            target_xyz = [x, y, z]

            cur_pose = self.get_current_tcp_pose()
            cur_roll, cur_pitch, cur_yaw = self.get_roll_from_pose(cur_pose)

            rospy.loginfo("Current EE position: x=%.4f y=%.4f z=%.4f",
                          cur_pose.position.x, cur_pose.position.y, cur_pose.position.z)
            rospy.loginfo("Current EE RPY: roll=%.4f pitch=%.4f yaw=%.4f",
                          cur_roll, cur_pitch, cur_yaw)

            self.arm_group.set_start_state_to_current_state()
            self.arm_group.set_goal_position_tolerance(0.01)
            self.arm_group.set_planning_time(self.planning_time)
            self.arm_group.set_num_planning_attempts(self.num_planning_attempts)
            self.arm_group.clear_pose_targets()
            self.arm_group.clear_path_constraints()

            self.arm_group.set_position_target(target_xyz, eef_link)

            rospy.loginfo("Position-only target [%s]: x=%.3f y=%.3f z=%.3f",
                          label, x, y, z)

            ok = self.arm_group.go(wait=True)
            self.arm_group.stop()
            self.arm_group.clear_pose_targets()
            self.arm_group.clear_path_constraints()

            if not ok:
                rospy.logerr("Position-only planning/execution failed for [%s]", label)
                return False

            final_pose = self.get_current_tcp_pose()
            final_roll, final_pitch, final_yaw = self.get_roll_from_pose(final_pose)

            rospy.loginfo("Arrived EE position: x=%.4f y=%.4f z=%.4f",
                          final_pose.position.x, final_pose.position.y, final_pose.position.z)
            rospy.loginfo("Arrived EE RPY: roll=%.4f pitch=%.4f yaw=%.4f",
                          final_roll, final_pitch, final_yaw)

            # 핵심 추가:
            # 1차 이동 후 현재 joint값으로 FK를 계산하고,
            # 카메라에서 얻은 목표 target_xyz(base)와 직접 비교
            self.log_target_vs_current_fk(target_xyz, label=label, prefix="After 1st move", use_stabilized=self.use_median_joint_sampling)

            return True

        except Exception as e:
            rospy.logerr("move_to_position_only failed for [%s]: %s", label, str(e))
            try:
                self.arm_group.stop()
                self.arm_group.clear_pose_targets()
                self.arm_group.clear_path_constraints()
            except Exception:
                pass
            return False

    # =========================================================
    # 수평을 맞추기 위한 q5 계산
    # 경험식: roll ≈ q2 + q3 + q5
    # => q5_des = roll_ref - q2 - q3
    # =========================================================
    def compute_q5_for_level(self, q2, q3):
        q5 = self.level_roll_ref - q2 - q3
        lo, hi = self.joint_limits["Revolute5"]
        return self.clamp(q5, lo, hi)

    def blend_q5_target(self, current_q5, desired_q5, gain=None):
        if gain is None:
            gain = self.comp_q5_gain
        q5 = current_q5 + gain * (desired_q5 - current_q5)
        lo, hi = self.joint_limits["Revolute5"]
        return self.clamp(q5, lo, hi)

    # =========================================================
    # 후보 평가
    # q2, q3 후보 -> q5는 계산식으로 결정
    # =========================================================
    def evaluate_comp_candidate(self, joint_names, joint_positions, target_xyz):
        fk_pose = self.get_fk_pose(joint_names, joint_positions, self.ee_link)
        if fk_pose is None:
            return None

        pos = self.pose_to_xyz(fk_pose)
        err = self.compute_xyz_error(target_xyz, pos)
        pos_err = err["dist"]
        z_drop = max(0.0, target_xyz[2] - pos[2])

        roll, pitch, yaw = self.get_roll_from_pose(fk_pose)
        roll_err = abs(roll - self.level_roll_ref)

        return {
            "joint_names": list(joint_names),
            "joints": list(joint_positions),
            "pose": fk_pose,
            "pos": pos,
            "dx": err["dx"],
            "dy": err["dy"],
            "dz": err["dz"],
            "pos_err": pos_err,
            "z_drop": z_drop,
            "roll": roll,
            "roll_err": roll_err,
            "score": self.score_comp_candidate(pos_err, roll_err, z_drop)
        }

    def build_comp_candidate(self, current_joints, name_to_idx, q2_new, q3_new, q5_gain=None):
        cand = list(current_joints)

        i2 = name_to_idx["Revolute2"]
        i3 = name_to_idx["Revolute3"]
        i5 = name_to_idx["Revolute5"]

        q2_new = self.clamp(
            q2_new,
            self.joint_limits["Revolute2"][0],
            self.joint_limits["Revolute2"][1]
        )
        q3_new = self.clamp(
            q3_new,
            self.joint_limits["Revolute3"][0],
            self.joint_limits["Revolute3"][1]
        )
        q5_desired = self.compute_q5_for_level(q2_new, q3_new)
        q5_new = self.blend_q5_target(current_joints[i5], q5_desired, gain=q5_gain)

        cand[i2] = q2_new
        cand[i3] = q3_new
        cand[i5] = q5_new
        return cand

    # =========================================================
    # 한 번에 보정용 최적 후보 계산
    # =========================================================
    def find_best_compensation_once(self, target_xyz, label="target"):
        joint_names, current_joints, name_to_idx, sample_info = self.get_joint_seed_for_compensation()

        if self.use_median_joint_sampling and sample_info is not None:
            rospy.loginfo("Compensation seed [%s] using stabilized median joints (wait=%.2fs, samples=%d, dt=%.2fs)",
                          label,
                          self.settle_initial_wait,
                          self.settle_num_samples,
                          self.settle_sample_interval)
            for idx, sample in enumerate(sample_info["samples"], start=1):
                rospy.loginfo("Compensation seed [%s] joint sample %d/%d: %s",
                              label, idx, len(sample_info["samples"]),
                              self.format_joint_values(sample_info["joint_names"], sample))
            rospy.loginfo("Compensation seed [%s] median joints: %s",
                          label,
                          self.format_joint_values(sample_info["joint_names"], sample_info["median_values"]))

        required = ["Revolute2", "Revolute3", "Revolute5"]
        if not all(n in name_to_idx for n in required):
            rospy.logwarn("Required joints for compensation are missing.")
            return None

        i2 = name_to_idx["Revolute2"]
        i3 = name_to_idx["Revolute3"]

        q2_cur = current_joints[i2]
        q3_cur = current_joints[i3]

        current_eval = self.evaluate_comp_candidate(joint_names, current_joints, target_xyz)
        if current_eval is None:
            rospy.logwarn("Current FK evaluation failed.")
            return None

        rospy.loginfo("Current FK candidate pos: x=%.4f y=%.4f z=%.4f",
                      current_eval["pos"][0], current_eval["pos"][1], current_eval["pos"][2])
        rospy.loginfo("Current TCP delta(target-fk): dx=%.4f dy=%.4f dz=%.4f | norm=%.4f m | z_drop=%.4f m",
                      current_eval["dx"], current_eval["dy"], current_eval["dz"],
                      current_eval["pos_err"], current_eval["z_drop"])
        rospy.loginfo("Current roll = %.4f, roll err = %.4f",
                      current_eval["roll"], current_eval["roll_err"])

        best_valid = None
        best_stage_name = None
        fallback_best = None
        fallback_stage_name = None

        for stage_idx, stage in enumerate(self.comp_adaptive_stages, start=1):
            stage_name = stage["name"]
            q2_span = self.comp_q2_span * stage.get("q2_span_scale", 1.0)
            q3_span = self.comp_q3_span * stage.get("q3_span_scale", 1.0)
            q5_gain = stage.get("q5_gain", self.comp_q5_gain)
            z_drop_limit = stage.get("z_drop_limit", 999.0)

            rospy.loginfo("Compensation stage %d [%s] for [%s]: q2_span=%.3f q3_span=%.3f step=%.3f q5_gain=%.2f z_drop_limit=%.3f m",
                          stage_idx, stage_name, label, q2_span, q3_span, self.comp_step, q5_gain, z_drop_limit)

            stage_best_valid = None
            stage_best_any = None

            for dq2 in self.frange(-q2_span, q2_span, self.comp_step):
                for dq3 in self.frange(-q3_span, q3_span, self.comp_step):
                    cand_joints = self.build_comp_candidate(
                        current_joints,
                        name_to_idx,
                        q2_cur + dq2,
                        q3_cur + dq3,
                        q5_gain=q5_gain
                    )

                    cand_eval = self.evaluate_comp_candidate(joint_names, cand_joints, target_xyz)
                    if cand_eval is None:
                        continue

                    cand_eval["stage_name"] = stage_name
                    cand_eval["q5_gain"] = q5_gain
                    cand_eval["z_drop_limit"] = z_drop_limit

                    if stage_best_any is None or cand_eval["score"] < stage_best_any["score"]:
                        stage_best_any = cand_eval

                    if cand_eval["z_drop"] <= z_drop_limit:
                        if stage_best_valid is None or cand_eval["score"] < stage_best_valid["score"]:
                            stage_best_valid = cand_eval

            if stage_best_any is not None:
                rospy.loginfo("Stage [%s] best-any pos: x=%.4f y=%.4f z=%.4f | dx=%.4f dy=%.4f dz=%.4f | norm=%.4f | z_drop=%.4f | roll_err=%.4f | score=%.4f",
                              stage_name,
                              stage_best_any["pos"][0], stage_best_any["pos"][1], stage_best_any["pos"][2],
                              stage_best_any["dx"], stage_best_any["dy"], stage_best_any["dz"],
                              stage_best_any["pos_err"], stage_best_any["z_drop"], stage_best_any["roll_err"], stage_best_any["score"])
                if fallback_best is None or stage_best_any["score"] < fallback_best["score"]:
                    fallback_best = stage_best_any
                    fallback_stage_name = stage_name

            if stage_best_valid is not None:
                improvement = current_eval["score"] - stage_best_valid["score"]
                rospy.loginfo("Stage [%s] best-valid pos: x=%.4f y=%.4f z=%.4f | dx=%.4f dy=%.4f dz=%.4f | norm=%.4f | z_drop=%.4f | roll_err=%.4f | score=%.4f | improvement=%.4f",
                              stage_name,
                              stage_best_valid["pos"][0], stage_best_valid["pos"][1], stage_best_valid["pos"][2],
                              stage_best_valid["dx"], stage_best_valid["dy"], stage_best_valid["dz"],
                              stage_best_valid["pos_err"], stage_best_valid["z_drop"], stage_best_valid["roll_err"], stage_best_valid["score"], improvement)
                best_valid = stage_best_valid
                best_stage_name = stage_name
                break
            else:
                rospy.logwarn("Stage [%s] found no valid candidate within z_drop_limit=%.3f m. Relax constraints and retry.",
                              stage_name, z_drop_limit)

        if best_valid is not None:
            final_best = best_valid
            improvement = current_eval["score"] - final_best["score"]
            rospy.loginfo("Selected best-valid stage [%s] for [%s]", best_stage_name, label)
            rospy.loginfo("Best FK candidate pos: x=%.4f y=%.4f z=%.4f",
                          final_best["pos"][0], final_best["pos"][1], final_best["pos"][2])
            rospy.loginfo("Best TCP delta(target-fk): dx=%.4f dy=%.4f dz=%.4f | norm=%.4f m | z_drop=%.4f m",
                          final_best["dx"], final_best["dy"], final_best["dz"],
                          final_best["pos_err"], final_best["z_drop"])
            rospy.loginfo("Best candidate roll = %.4f, roll err = %.4f, q5_gain = %.2f",
                          final_best["roll"], final_best["roll_err"], final_best.get("q5_gain", self.comp_q5_gain))
            rospy.loginfo("Best candidate score improvement = %.4f", improvement)
            return final_best

        if fallback_best is not None:
            improvement = current_eval["score"] - fallback_best["score"]
            rospy.logwarn("No z-constrained candidate found for [%s]. Use fallback least-loss candidate from stage [%s] to avoid skipping compensation.",
                          label, fallback_stage_name)
            rospy.loginfo("Fallback FK candidate pos: x=%.4f y=%.4f z=%.4f",
                          fallback_best["pos"][0], fallback_best["pos"][1], fallback_best["pos"][2])
            rospy.loginfo("Fallback TCP delta(target-fk): dx=%.4f dy=%.4f dz=%.4f | norm=%.4f m | z_drop=%.4f m",
                          fallback_best["dx"], fallback_best["dy"], fallback_best["dz"],
                          fallback_best["pos_err"], fallback_best["z_drop"])
            rospy.loginfo("Fallback candidate roll = %.4f, roll err = %.4f, q5_gain = %.2f",
                          fallback_best["roll"], fallback_best["roll_err"], fallback_best.get("q5_gain", self.comp_q5_gain))
            rospy.loginfo("Fallback candidate score improvement = %.4f", improvement)
            return fallback_best

        rospy.logwarn("No compensation candidate could be evaluated for [%s]. Return current posture as fallback.", label)
        current_eval["stage_name"] = "current"
        current_eval["q5_gain"] = 0.0
        return current_eval

    # =========================================================
    # 2차: 한 번에 수평 보정 실행
    # =========================================================
    def compensate_after_move_once(self, target_xyz, label="target"):
        if not self.enable_moveit or self.arm_group is None:
            return False

        try:
            self.log_target_vs_current_fk(target_xyz, label=label, prefix="Before 2nd correction", use_stabilized=self.use_median_joint_sampling)

            best = self.find_best_compensation_once(target_xyz, label=label)
            if best is None:
                rospy.logwarn("No useful compensation candidate found for [%s]", label)
                return False

            joint_names = list(best["joint_names"])
            target_joint_map = {n: v for n, v in zip(joint_names, best["joints"])}

            rospy.loginfo("Execute one-shot compensation for [%s]", label)
            rospy.loginfo("Target joints after compensation: %s",
                          self.format_joint_values(joint_names, best["joints"]))

            self.arm_group.set_start_state_to_current_state()
            self.arm_group.set_planning_time(2.0)
            self.arm_group.set_num_planning_attempts(5)
            self.arm_group.clear_pose_targets()
            self.arm_group.clear_path_constraints()
            self.arm_group.set_joint_value_target(target_joint_map)

            ok = self.arm_group.go(wait=True)
            self.arm_group.stop()
            self.arm_group.clear_pose_targets()
            self.arm_group.clear_path_constraints()

            if not ok:
                rospy.logwarn("One-shot compensation motion failed for [%s]", label)
                return False

            final_pose = self.get_current_tcp_pose()
            final_roll, final_pitch, final_yaw = self.get_roll_from_pose(final_pose)
            final_pos = self.pose_to_xyz(final_pose)
            final_pos_err = self.norm3(final_pos, target_xyz)

            rospy.loginfo("After compensation EE position: x=%.4f y=%.4f z=%.4f",
                          final_pose.position.x, final_pose.position.y, final_pose.position.z)
            rospy.loginfo("After compensation EE RPY: roll=%.4f pitch=%.4f yaw=%.4f",
                          final_roll, final_pitch, final_yaw)
            rospy.loginfo("After compensation TCP pos err = %.4f m", final_pos_err)

            self.log_target_vs_current_fk(target_xyz, label=label, prefix="After 2nd correction", use_stabilized=self.use_median_joint_sampling)
            return True

        except Exception as e:
            rospy.logerr("compensate_after_move_once failed for [%s]: %s", label, str(e))
            try:
                self.arm_group.stop()
                self.arm_group.clear_pose_targets()
                self.arm_group.clear_path_constraints()
            except Exception:
                pass
            return False

    # =========================================================
    # 최종 동작:
    # 1) position_only_ik로 목표 위치 이동
    # 2) 안정화된 joint_states 중앙값 기반 FK로 1차 오차 확인
    # 3) q2, q3, q5를 이용한 2차 수평 보정 재실행
    # =========================================================
    def scale_trajectory_timing(self, traj, vel_scale=1.0, acc_scale=1.0):
        """
        RobotTrajectory 시간 스케일 조정
        vel_scale < 1.0 이면 느리게 실행
        """
        if traj is None:
            return None

        try:
            new_traj = type(traj)()
            new_traj.joint_trajectory = traj.joint_trajectory
            points = []

            vel_scale = max(1e-3, float(vel_scale))
            acc_scale = max(1e-3, float(acc_scale))

            for pt in traj.joint_trajectory.points:
                new_pt = copy.deepcopy(pt)
                new_pt.time_from_start = rospy.Duration(pt.time_from_start.to_sec() / vel_scale)

                if len(new_pt.velocities) > 0:
                    new_pt.velocities = [v * vel_scale for v in pt.velocities]
                if len(new_pt.accelerations) > 0:
                    new_pt.accelerations = [a * acc_scale for a in pt.accelerations]

                points.append(new_pt)

            new_traj.joint_trajectory.points = points
            return new_traj
        except Exception as e:
            rospy.logwarn("scale_trajectory_timing failed, use original trajectory: %s", str(e))
            return traj

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

    def current_callback(self, msg):
        try:
            self.latest_joint_current_raw = list(msg.data)
            self.current_msg_time = rospy.Time.now()
        except Exception:
            self.latest_joint_current_raw = None

    def get_current_q2_q3_raw(self):
        if self.latest_joint_current_raw is None:
            return None, None
        data = self.latest_joint_current_raw
        if len(data) <= max(self.current_idx_q2, self.current_idx_q3):
            return None, None
        return float(data[self.current_idx_q2]), float(data[self.current_idx_q3])

    def measure_push_current_baseline(self, label="target"):
        rospy.loginfo("Measure push current baseline [%s]: duration=%.2f s, dt=%.2f s",
                      label, self.current_baseline_duration, self.current_baseline_dt)

        samples_q2 = []
        samples_q3 = []
        start_t = time.time()
        while time.time() - start_t < self.current_baseline_duration and not rospy.is_shutdown():
            i2, i3 = self.get_current_q2_q3_raw()
            if i2 is not None and i3 is not None:
                samples_q2.append(i2)
                samples_q3.append(i3)
            rospy.sleep(self.current_baseline_dt)

        if len(samples_q2) == 0 or len(samples_q3) == 0:
            rospy.logwarn("No current samples collected for baseline [%s]", label)
            return None

        baseline = {
            "q2": statistics.median(samples_q2),
            "q3": statistics.median(samples_q3),
            "n": len(samples_q2),
        }
        rospy.loginfo("Baseline current [%s]: q2=%.1f, q3=%.1f (n=%d)",
                      label, baseline["q2"], baseline["q3"], baseline["n"])
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

        info = {"i2": i2, "i3": i3, "d2": d2, "d3": d3, "contact": contact}
        rospy.loginfo_throttle(0.2,
            "Current monitor [%s]: q2=%.1f (d=%.1f), q3=%.1f (d=%.1f), contact=%s",
            label, i2, d2, i3, d3, str(contact))
        return contact, info

    def get_stabilized_joint_map(self, label="target"):
        names, joints = self.sample_stabilized_joints(label=label, prefix="Save stabilized joints")
        if not names or not joints or len(names) != len(joints):
            return None
        return {n: float(v) for n, v in zip(names, joints)}

    def move_to_joint_map(self, joint_map, label="target", prefix="Move to joint map"):
        if not joint_map:
            rospy.logwarn("%s [%s] skipped: empty joint map", prefix, label)
            return False
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
            try:
                self.arm_group.stop()
                self.arm_group.clear_pose_targets()
            except Exception:
                pass
            return False

    def move_to_saved_pre_push_then_start_pose(self, label="target"):
        ok_pre = True
        if self.pre_push_joint_map is not None:
            ok_pre = self.move_to_joint_map(self.pre_push_joint_map, label=label, prefix="Move back to pre-push pose")
            rospy.sleep(0.5)
        else:
            rospy.logwarn("No pre-push joint map saved for [%s]; go directly to start pose", label)

        ok_start = self.move_to_joint_map(self.start_pose_joint_map, label=label, prefix="Move back to saved start pose")
        return ok_pre and ok_start

    def execute_pose_push(self, label="target", distance_y=None):
        """
        일반 pose planning으로 현재 pose 기준 +y 방향으로 전진하되,
        push 직전 0.5초 동안 q2/q3 전류 baseline을 측정하고,
        이동 중 q3 상승 + q2 절대 변화량이 기준을 넘으면 즉시 정지한 뒤
        pre-push 자세 -> 초기 자세 순서로 복귀한다.
        """
        if distance_y is None:
            distance_y = self.pose_push_distance

        if not self.enable_moveit or self.arm_group is None:
            rospy.logwarn("Pose push skipped for [%s]: MoveIt disabled", label)
            return False

        try:
            # push 시작 직전의 안정화된 joint를 저장해두었다가
            # 접촉 시 먼저 이 자세로 복귀한다.
            self.pre_push_joint_map = self.get_stabilized_joint_map(label=label)
            if self.pre_push_joint_map is not None:
                rospy.loginfo("Saved pre-push joint map [%s]", label)

            current_pose = self.get_current_tcp_pose()
            target_pose = copy.deepcopy(current_pose)
            target_pose.position.y += distance_y

            rospy.loginfo(
                "Pose target [%s]: start=(%.4f, %.4f, %.4f) -> target=(%.4f, %.4f, %.4f), dy=%.4f",
                label,
                current_pose.position.x, current_pose.position.y, current_pose.position.z,
                target_pose.position.x, target_pose.position.y, target_pose.position.z,
                distance_y
            )

            baseline = self.measure_push_current_baseline(label=label)
            if baseline is None:
                rospy.logwarn("Pose push skipped for [%s]: baseline measurement failed", label)
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

            n_points = 0
            try:
                n_points = len(traj.joint_trajectory.points)
            except Exception:
                n_points = 0

            rospy.loginfo(
                "Pose push plan [%s]: success=%s, points=%d, pos_tol=%.4f, ori_tol=%.4f, joint_tol=%.4f",
                label, str(plan_success), n_points,
                self.pose_push_pos_tol, self.pose_push_ori_tol, self.pose_push_joint_tol
            )

            if not plan_success or n_points <= 0:
                rospy.logwarn("Pose push planning failed for [%s]", label)
                self.arm_group.clear_pose_targets()
                return False

            rospy.loginfo("Execute pose push with current guard [%s]: dy=%.4f m (general planning, slow mode)", label, distance_y)
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
                    rospy.logwarn(
                        "Contact detected [%s]: q2=%.1f (d=%.1f), q3=%.1f (d=%.1f)",
                        label, info["i2"], info["d2"], info["i3"], info["d3"]
                    )
                    contact_triggered = True
                    self.arm_group.stop()
                    break

                pose_now = self.get_current_tcp_pose()
                dy_remain = abs(target_pose.position.y - pose_now.position.y)
                if dy_remain < 0.003:
                    rospy.loginfo("Pose push finished normally [%s]", label)
                    break

                if time.time() - start_t > self.pose_push_timeout:
                    rospy.logwarn("Pose push timeout [%s]", label)
                    self.arm_group.stop()
                    break

                rospy.sleep(self.current_poll_dt)

            self.arm_group.stop()
            self.arm_group.clear_pose_targets()
            rospy.sleep(0.3)

            final_pose = self.get_current_tcp_pose()
            roll, pitch, yaw = self.get_roll_from_pose(final_pose)
            rospy.loginfo(
                "After pose push [%s]: x=%.4f y=%.4f z=%.4f | roll=%.4f pitch=%.4f yaw=%.4f",
                label,
                final_pose.position.x, final_pose.position.y, final_pose.position.z,
                roll, pitch, yaw
            )

            if contact_triggered:
                rospy.loginfo("Return sequence after contact [%s]: pre-push -> start pose", label)
                self.move_to_saved_pre_push_then_start_pose(label=label)

            return True

        except Exception as e:
            rospy.logerr("execute_pose_push failed for [%s]: %s", label, str(e))
            try:
                self.arm_group.stop()
                self.arm_group.clear_pose_targets()
            except Exception:
                pass
            return False

    def execute_pose_back(self, label="target", distance_y=None):
        if distance_y is None:
            distance_y = self.pose_back_distance
        return self.execute_pose_push(label=label + "_back", distance_y=-abs(distance_y))

    def move_to_target_then_compensate(self, x, y, z, label="target"):
        ok = self.move_to_position_only(x, y, z, label=label)
        if not ok:
            return False

        target_xyz = [x, y, z]

        comp_ok = self.compensate_after_move_once(target_xyz, label=label)

        if comp_ok:
            rospy.loginfo("2-stage motion success for [%s] (move + one-shot compensation)", label)
        else:
            rospy.logwarn("2-stage motion partial success for [%s] (move ok, compensation weak/fallback)", label)

        push_ok = self.execute_pose_push(label=label)
        if push_ok:
            rospy.loginfo("Pose push success for [%s]", label)
        else:
            rospy.logwarn("Pose push failed for [%s]", label)

        return comp_ok and push_ok

    # =========================================================
    # 명령 처리
    # =========================================================
    def execute_pending_target_if_requested(self):
        if self.pending_target_label is None:
            return

        cmd = self.pending_target_label.strip().upper()
        self.pending_target_label = None

        if cmd == "LIST":
            self.print_detected_targets_log()
            return

        if cmd not in self.detected_targets:
            rospy.logwarn("Requested label [%s] is not currently detected.", cmd)
            return

        info = self.detected_targets[cmd]

        rospy.loginfo("RAW target [%s]: x=%.3f y=%.3f z=%.3f",
                      cmd, info["x"], info["y"], info["z"])

        x = info["x"] + self.target_offset_x
        y = info["y"] + self.target_offset_y
        z = info["z"] + self.target_offset_z

        rospy.loginfo("Use latest detected coordinate for [%s]: x=%.3f y=%.3f z=%.3f",
                      cmd, x, y, z)

        self.move_to_target_then_compensate(x, y, z, label=cmd)

    # =========================================================
    # 메인 루프
    # =========================================================
    def run(self):
        rate = rospy.Rate(self.loop_hz)

        rospy.loginfo("YOLO object TF detector is running...")
        rospy.loginfo("Python executable: %s", sys.executable)

        while not rospy.is_shutdown():
            try:
                frames = self.pipeline.wait_for_frames()
                aligned_frames = self.align.process(frames)

                depth_frame = aligned_frames.get_depth_frame()
                color_frame = aligned_frames.get_color_frame()

                if not depth_frame or not color_frame:
                    rate.sleep()
                    continue

                color_image = np.asanyarray(color_frame.get_data())
                stamp = rospy.Time.now()

                results = self.model(color_image, stream=True, verbose=False)

                for r in results:
                    for box in r.boxes:
                        conf = float(box.conf[0])
                        if conf < self.conf_thresh:
                            continue

                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        x1i, y1i, x2i, y2i = int(x1), int(y1), int(x2), int(y2)

                        cx = int((x1 + x2) / 2.0)
                        cy = int((y1 + y2) / 2.0)

                        cls = int(box.cls[0])
                        name = str(self.model.names[cls])

                        if self.target_class and name != self.target_class:
                            continue

                        if 0 <= cx < 1280 and 0 <= cy < 720:
                            distance = self.get_depth_robust(
                                depth_frame,
                                cx,
                                cy,
                                r=self.depth_patch_radius
                            )
                        else:
                            distance = 0.0

                        label = f"{name} {conf:.2f} | {distance:.2f}m"
                        cv2.rectangle(color_image, (x1i, y1i), (x2i, y2i), (255, 0, 0), 2)
                        cv2.putText(
                            color_image,
                            label,
                            (x1i, max(20, y1i - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,
                            (255, 255, 255),
                            2
                        )
                        cv2.circle(color_image, (cx, cy), 5, (0, 0, 255), -1)

                        if distance > 0.0:
                            point_3d = rs.rs2_deproject_pixel_to_point(
                                self.color_intrinsics,
                                [cx, cy],
                                distance
                            )

                            X = float(point_3d[0])
                            Y = float(point_3d[1])
                            Z = float(point_3d[2])

                            cv2.putText(
                                color_image,
                                f"X: {X:.3f} m",
                                (cx - 60, cy + 20),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.55,
                                (0, 255, 0),
                                2
                            )
                            cv2.putText(
                                color_image,
                                f"Y: {Y:.3f} m",
                                (cx - 60, cy + 42),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.55,
                                (0, 255, 0),
                                2
                            )
                            cv2.putText(
                                color_image,
                                f"Z: {Z:.3f} m",
                                (cx - 60, cy + 64),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.55,
                                (0, 255, 0),
                                2
                            )

                            child_frame_id = "{}_{}".format(self.child_frame_prefix, name)
                            # self.publish_tf(X, Y, Z, stamp, child_frame_id)

                            base_pt = self.transform_point_to_base(X, Y, Z, stamp)
                            if base_pt is not None:
                                bx = base_pt.point.x
                                by = base_pt.point.y
                                bz = base_pt.point.z

                                cv2.putText(
                                    color_image,
                                    f"BX: {bx:.3f}",
                                    (cx - 60, cy + 88),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    0.55,
                                    (255, 0, 0),
                                    2
                                )
                                cv2.putText(
                                    color_image,
                                    f"BY: {by:.3f}",
                                    (cx - 60, cy + 110),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    0.55,
                                    (255, 0, 0),
                                    2
                                )
                                cv2.putText(
                                    color_image,
                                    f"BZ: {bz:.3f}",
                                    (cx - 60, cy + 132),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    0.55,
                                    (255, 0, 0),
                                    2
                                )

                                label_key = name.strip().upper()
                                self.update_detected_target(label_key, bx, by, bz, conf, stamp)

                                cv2.putText(
                                    color_image,
                                    f"NAME: {label_key}",
                                    (cx - 60, cy + 154),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    0.55,
                                    (0, 255, 255),
                                    2
                                )

                            else:
                                cv2.putText(
                                    color_image,
                                    "base_link TF failed",
                                    (cx - 60, cy + 88),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    0.55,
                                    (0, 0, 255),
                                    2
                                )

                        else:
                            cv2.putText(
                                color_image,
                                "No depth",
                                (cx - 40, cy + 20),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.6,
                                (0, 0, 255),
                                2
                            )

                self.clear_stale_targets(stamp)
                self.execute_pending_target_if_requested()

                y0 = 25
                cv2.putText(
                    color_image,
                    "Latest detected labels:",
                    (20, y0),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (0, 255, 255),
                    2
                )
                y0 += 25

                for label_key, info in sorted(self.detected_targets.items()):
                    txt = "{} -> ({:.3f}, {:.3f}, {:.3f})".format(
                        label_key, info["x"], info["y"], info["z"]
                    )
                    cv2.putText(
                        color_image,
                        txt,
                        (20, y0),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.55,
                        (0, 255, 255),
                        2
                    )
                    y0 += 22

                if self.last_fk_compare_result is not None:
                    r = self.last_fk_compare_result
                    cv2.putText(
                        color_image,
                        "{} {} d=({:+.3f}, {:+.3f}, {:+.3f}) norm={:.3f}m".format(
                            r["prefix"], r["label"], r["dx"], r["dy"], r["dz"], r["dist"]
                        ),
                        (20, y0 + 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.52,
                        (0, 200, 255),
                        2
                    )

                if self.view_image:
                    cv2.imshow("YOLO + RealSense + PositionOnlyIK + OneShotComp", color_image)
                    key = cv2.waitKey(1) & 0xFF

                    if key == ord('q'):
                        rospy.loginfo("Pressed q. Shutdown.")
                        rospy.signal_shutdown("User requested shutdown")
                        break

                rate.sleep()

            except rospy.ROSInterruptException:
                break
            except Exception as e:
                rospy.logerr_throttle(1.0, "Runtime error: %s", str(e))
                rate.sleep()

    def shutdown_hook(self):
        rospy.loginfo("Shutting down YOLO object TF detector...")

        try:
            self.pipeline.stop()
        except Exception:
            pass

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
