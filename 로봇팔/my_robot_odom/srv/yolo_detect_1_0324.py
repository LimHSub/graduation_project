#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import cv2
import math
import rospy
import logging
import numpy as np
import pyrealsense2 as rs
import moveit_commander
import tf2_ros
import tf2_geometry_msgs

from ultralytics import YOLO
from geometry_msgs.msg import TransformStamped, PointStamped
from std_msgs.msg import String
from moveit_msgs.srv import GetPositionFK, GetPositionFKRequest
from tf.transformations import euler_from_quaternion


class YoloObjectTFDetector:
    def __init__(self):
        rospy.init_node("yolo_object_tf_detector", anonymous=False)

        # =========================================================
        # 직접 코드에서 설정
        # =========================================================
        self.model_path = "/home/inwoong/catkin_ws/best.pt"
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

        self.planning_time = 3.0
        self.num_planning_attempts = 10
        self.max_vel_scale = 0.2
        self.max_acc_scale = 0.2

        # 현재 실험으로 맞춘 위치 보정
        self.target_offset_x = -0.04
        self.target_offset_y = 0.0
        self.target_offset_z = 0.03

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

        # 이 이상 좋아져야 실제 2차 보정 실행
        self.comp_min_score_improvement = 0.01

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

    def score_comp_candidate(self, pos_err, roll_err):
        return self.comp_pos_weight * pos_err + self.comp_roll_weight * roll_err

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

    # =========================================================
    # 1차: position_only_ik 기반 위치 이동
    # =========================================================
    def move_to_position_only(self, x, y, z, label="target"):
        if not self.enable_moveit or self.arm_group is None:
            rospy.logwarn("MoveIt disabled. Skip command for %s", label)
            return False

        try:
            eef_link = self.arm_group.get_end_effector_link()

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

            self.arm_group.set_position_target([x, y, z], eef_link)

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

    # =========================================================
    # 후보 평가
    # q2, q3 후보 -> q5는 계산식으로 결정
    # =========================================================
    def evaluate_comp_candidate(self, joint_names, joint_positions, target_xyz):
        fk_pose = self.get_fk_pose(joint_names, joint_positions, self.ee_link)
        if fk_pose is None:
            return None

        pos = [fk_pose.position.x, fk_pose.position.y, fk_pose.position.z]
        pos_err = self.norm3(pos, target_xyz)

        roll, pitch, yaw = self.get_roll_from_pose(fk_pose)
        roll_err = abs(roll - self.level_roll_ref)

        return {
            "joints": list(joint_positions),
            "pose": fk_pose,
            "pos_err": pos_err,
            "roll": roll,
            "roll_err": roll_err,
            "score": self.score_comp_candidate(pos_err, roll_err)
        }

    def build_comp_candidate(self, current_joints, name_to_idx, q2_new, q3_new):
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
        q5_new = self.compute_q5_for_level(q2_new, q3_new)

        cand[i2] = q2_new
        cand[i3] = q3_new
        cand[i5] = q5_new
        return cand

    # =========================================================
    # 한 번에 보정용 최적 후보 계산
    # =========================================================
    def find_best_compensation_once(self, target_xyz):
        joint_names, current_joints, name_to_idx = self.get_current_joint_map()

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

        rospy.loginfo("Current TCP pos err = %.4f m", current_eval["pos_err"])
        rospy.loginfo("Current roll = %.4f, roll err = %.4f",
                      current_eval["roll"], current_eval["roll_err"])

        best = current_eval

        for dq2 in self.frange(-self.comp_q2_span, self.comp_q2_span, self.comp_step):
            for dq3 in self.frange(-self.comp_q3_span, self.comp_q3_span, self.comp_step):
                cand_joints = self.build_comp_candidate(
                    current_joints,
                    name_to_idx,
                    q2_cur + dq2,
                    q3_cur + dq3
                )

                cand_eval = self.evaluate_comp_candidate(joint_names, cand_joints, target_xyz)
                if cand_eval is None:
                    continue

                if cand_eval["score"] < best["score"]:
                    best = cand_eval

        improvement = current_eval["score"] - best["score"]

        rospy.loginfo("Best candidate pos err = %.4f m", best["pos_err"])
        rospy.loginfo("Best candidate roll = %.4f, roll err = %.4f",
                      best["roll"], best["roll_err"])
        rospy.loginfo("Best candidate score improvement = %.4f", improvement)

        if improvement < self.comp_min_score_improvement:
            rospy.logwarn("Compensation improvement is too small. Skip 2nd correction.")
            return None

        return best

    # =========================================================
    # 2차: 한 번에 수평 보정 실행
    # =========================================================
    def compensate_after_move_once(self, target_xyz, label="target"):
        if not self.enable_moveit or self.arm_group is None:
            return False

        try:
            best = self.find_best_compensation_once(target_xyz)
            if best is None:
                rospy.logwarn("No useful compensation candidate found for [%s]", label)
                return False

            joint_names, _, _ = self.get_current_joint_map()

            rospy.loginfo("Execute one-shot compensation for [%s]", label)
            rospy.loginfo("Target joints after compensation: %s",
                          ["%s=%.4f" % (n, v) for n, v in zip(joint_names, best["joints"])])

            self.arm_group.set_start_state_to_current_state()
            self.arm_group.set_planning_time(2.0)
            self.arm_group.set_num_planning_attempts(5)
            self.arm_group.clear_pose_targets()
            self.arm_group.clear_path_constraints()
            self.arm_group.set_joint_value_target(best["joints"])

            ok = self.arm_group.go(wait=True)
            self.arm_group.stop()
            self.arm_group.clear_pose_targets()
            self.arm_group.clear_path_constraints()

            if not ok:
                rospy.logwarn("One-shot compensation motion failed for [%s]", label)
                return False

            final_pose = self.get_current_tcp_pose()
            final_roll, final_pitch, final_yaw = self.get_roll_from_pose(final_pose)
            final_pos = [final_pose.position.x, final_pose.position.y, final_pose.position.z]
            final_pos_err = self.norm3(final_pos, target_xyz)

            rospy.loginfo("After compensation EE position: x=%.4f y=%.4f z=%.4f",
                          final_pose.position.x, final_pose.position.y, final_pose.position.z)
            rospy.loginfo("After compensation EE RPY: roll=%.4f pitch=%.4f yaw=%.4f",
                          final_roll, final_pitch, final_yaw)
            rospy.loginfo("After compensation TCP pos err = %.4f m", final_pos_err)

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
    # 2) q2/q3/q5를 한 번에 계산해서 수평 보정 실행
    # =========================================================
    def move_to_target_then_compensate(self, x, y, z, label="target"):
        ok = self.move_to_position_only(x, y, z, label=label)
        if not ok:
            return False

        target_xyz = [x, y, z]
        comp_ok = self.compensate_after_move_once(target_xyz, label=label)

        if comp_ok:
            rospy.loginfo("2-stage motion success for [%s] (move + one-shot compensation)", label)
        else:
            rospy.logwarn("2-stage motion partial success for [%s] (move ok, compensation weak/skip)", label)

        return True

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
