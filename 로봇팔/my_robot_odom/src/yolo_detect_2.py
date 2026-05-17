#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import math
import copy
import numpy as np
import rospy
import moveit_commander

from tf.transformations import euler_from_quaternion


def wrap_to_pi(angle):
    while angle > math.pi:
        angle -= 2.0 * math.pi
    while angle < -math.pi:
        angle += 2.0 * math.pi
    return angle


def circular_mean(angles):
    s = np.mean(np.sin(angles))
    c = np.mean(np.cos(angles))
    return math.atan2(s, c)


class WeightedJacobianLeveler(object):
    def __init__(self):
        moveit_commander.roscpp_initialize(sys.argv)
        rospy.init_node("weighted_jacobian_leveler", anonymous=True)

        # -------------------------------
        # 기본 파라미터
        # -------------------------------
        self.group_name = rospy.get_param("~group_name", "arm")
        self.eef_link = rospy.get_param("~eef_link", "")

        # active joint 기준 index
        # 예: [q1, q2, q3, q4, q5] 면 q2=1, q3=2, q5=4
        self.q2_idx = rospy.get_param("~q2_index", 1)
        self.q3_idx = rospy.get_param("~q3_index", 2)
        self.q5_idx = rospy.get_param("~q5_index", 4)

        # Jacobian row 선택
        # 일반적으로 x=0, z=2, pitch(about y)=4 를 많이 씀
        # 만약 네 로봇이 다른 축 기준이면 이 값 수정
        self.jacobian_x_row = rospy.get_param("~jacobian_x_row", 0)
        self.jacobian_z_row = rospy.get_param("~jacobian_z_row", 2)
        self.jacobian_pitch_row = rospy.get_param("~jacobian_pitch_row", 4)

        # 수평 목표 pitch (라디안)
        # 로봇 구조에 따라 0.0 또는 +/-pi/2 또는 pi 일 수 있음
        self.target_pitch = rospy.get_param("~target_pitch", 0.0)

        # 안정화
        self.settle_sleep = rospy.get_param("~settle_sleep", 0.5)
        self.sample_count = rospy.get_param("~sample_count", 5)
        self.sample_dt = rospy.get_param("~sample_dt", 0.05)

        # 보정 반복 횟수
        self.max_correction_iters = rospy.get_param("~max_correction_iters", 6)

        # 허용 오차
        self.pos_tol = rospy.get_param("~pos_tol", 0.003)  # m
        self.pitch_tol = rospy.get_param("~pitch_tol", math.radians(1.5))  # rad

        # 오차 gain
        self.k_pos_x = rospy.get_param("~k_pos_x", 1.0)
        self.k_pos_z = rospy.get_param("~k_pos_z", 1.0)
        self.k_pitch = rospy.get_param("~k_pitch", 1.0)

        # DLS damping
        self.damping = rospy.get_param("~damping", 1e-3)

        # q2 : q3 = 90 : 10
        # weighted least squares에서 "더 많이 움직이고 싶은 관절"은 cost를 더 작게 준다.
        q2_share = rospy.get_param("~q2_share", 0.9)
        q3_share = rospy.get_param("~q3_share", 0.1)

        raw_q2_cost = 1.0 / max(q2_share, 1e-6)
        raw_q3_cost = 1.0 / max(q3_share, 1e-6)
        min_cost = min(raw_q2_cost, raw_q3_cost)

        # 정규화하면 q2_cost = 1, q3_cost = 9 비슷한 형태가 됨
        self.q2_cost = raw_q2_cost / min_cost
        self.q3_cost = raw_q3_cost / min_cost

        # q5는 수평 조절 핵심이므로 중간 정도 cost
        self.q5_cost = rospy.get_param("~q5_cost", 2.0)

        # 1회 보정 스텝 제한
        self.max_step_q2 = rospy.get_param("~max_step_q2", 0.12)  # rad
        self.max_step_q3 = rospy.get_param("~max_step_q3", 0.05)  # rad
        self.max_step_q5 = rospy.get_param("~max_step_q5", 0.10)  # rad

        # 실행 속도
        self.vel_scale = rospy.get_param("~vel_scale", 0.2)
        self.acc_scale = rospy.get_param("~acc_scale", 0.2)

        # joint goal 입력용
        self.initial_joint_goal = rospy.get_param("~joint_goal", [])

        # -------------------------------
        # MoveIt 초기화
        # -------------------------------
        self.robot = moveit_commander.RobotCommander()
        self.group = moveit_commander.MoveGroupCommander(self.group_name)

        if self.eef_link:
            self.group.set_end_effector_link(self.eef_link)
        else:
            self.eef_link = self.group.get_end_effector_link()

        self.group.set_max_velocity_scaling_factor(self.vel_scale)
        self.group.set_max_acceleration_scaling_factor(self.acc_scale)
        self.group.set_num_planning_attempts(10)
        self.group.set_planning_time(3.0)

        rospy.loginfo("=== Weighted Jacobian Leveler Start ===")
        rospy.loginfo("group_name         : %s", self.group_name)
        rospy.loginfo("eef_link           : %s", self.eef_link)
        rospy.loginfo("q2/q3/q5 idx       : %d / %d / %d", self.q2_idx, self.q3_idx, self.q5_idx)
        rospy.loginfo("target_pitch(rad)  : %.4f", self.target_pitch)
        rospy.loginfo("q2/q3/q5 cost      : %.3f / %.3f / %.3f",
                      self.q2_cost, self.q3_cost, self.q5_cost)

    # ------------------------------------------------------------
    # Pose / RPY
    # ------------------------------------------------------------
    def pose_to_rpy(self, pose):
        q = pose.orientation
        quat = [q.x, q.y, q.z, q.w]
        roll, pitch, yaw = euler_from_quaternion(quat)
        return roll, pitch, yaw

    def get_pose_xyz_pitch(self):
        pose = self.group.get_current_pose(self.eef_link).pose
        roll, pitch, yaw = self.pose_to_rpy(pose)
        xyz = np.array([pose.position.x, pose.position.y, pose.position.z], dtype=float)
        return pose, xyz, pitch

    # ------------------------------------------------------------
    # 안정화 후 평균 상태 읽기
    # ------------------------------------------------------------
    def get_stable_state(self):
        self.group.stop()
        rospy.sleep(self.settle_sleep)

        joint_samples = []
        pos_samples = []
        pitch_samples = []

        for _ in range(self.sample_count):
            joints = np.array(self.group.get_current_joint_values(), dtype=float)
            _, xyz, pitch = self.get_pose_xyz_pitch()

            joint_samples.append(joints)
            pos_samples.append(xyz)
            pitch_samples.append(pitch)

            rospy.sleep(self.sample_dt)

        mean_joints = np.mean(np.vstack(joint_samples), axis=0)
        mean_pos = np.mean(np.vstack(pos_samples), axis=0)
        mean_pitch = circular_mean(np.array(pitch_samples))

        return mean_joints.tolist(), mean_pos, mean_pitch

    # ------------------------------------------------------------
    # planning 결과 파싱
    # ------------------------------------------------------------
    def extract_plan(self, plan_result):
        """
        MoveIt / 환경에 따라 plan() 반환형이 다를 수 있으므로 안전하게 처리
        """
        if isinstance(plan_result, tuple):
            # 흔한 경우:
            # (success, plan, planning_time, error_code)
            # 또는 (plan, fraction) 류
            if len(plan_result) >= 2 and isinstance(plan_result[0], bool):
                success = plan_result[0]
                plan = plan_result[1]
                return success, plan

            # 첫 원소가 trajectory일 가능성
            if len(plan_result) >= 1:
                plan = plan_result[0]
                has_traj = hasattr(plan, "joint_trajectory")
                if has_traj and len(plan.joint_trajectory.points) > 0:
                    return True, plan
                return False, plan_result

        # 단일 RobotTrajectory
        if hasattr(plan_result, "joint_trajectory"):
            if len(plan_result.joint_trajectory.points) > 0:
                return True, plan_result
            return False, plan_result

        return False, plan_result

    # ------------------------------------------------------------
    # joint goal 실행
    # ------------------------------------------------------------
    def execute_joint_goal(self, joint_goal):
        self.group.stop()
        rospy.sleep(0.1)

        # 현재 상태를 start state로 다시 맞춤
        self.group.set_start_state_to_current_state()
        self.group.set_joint_value_target(joint_goal)

        plan_result = self.group.plan()
        success, plan = self.extract_plan(plan_result)

        if not success:
            rospy.logerr("Planning failed")
            return False

        if not hasattr(plan, "joint_trajectory") or len(plan.joint_trajectory.points) == 0:
            rospy.logerr("Empty trajectory")
            return False

        ok = self.group.execute(plan, wait=True)
        self.group.stop()
        self.group.clear_pose_targets()
        rospy.sleep(0.1)

        if not ok:
            rospy.logerr("Trajectory execution failed")
            return False

        return True

    # ------------------------------------------------------------
    # weighted DLS Jacobian
    # ------------------------------------------------------------
    def solve_weighted_dls(self, J_sel, err_vec):
        """
        dq = W^-1 J^T (J W^-1 J^T + lambda^2 I)^-1 e
        """
        W = np.diag([self.q2_cost, self.q3_cost, self.q5_cost])
        W_inv = np.linalg.inv(W)

        A = J_sel @ W_inv @ J_sel.T + (self.damping ** 2) * np.eye(J_sel.shape[0])
        dq = W_inv @ J_sel.T @ np.linalg.solve(A, err_vec)
        return dq

    # ------------------------------------------------------------
    # 보정 루프
    # ------------------------------------------------------------
    def level_with_weighted_jacobian(self, target_pitch=None):
        """
        현재 TCP 위치를 reference로 잡고,
        pitch를 target_pitch로 맞추면서 x/z 위치 오차를 최소화한다.
        """

        if target_pitch is None:
            target_pitch = self.target_pitch

        # 보정 시작 직전의 TCP 위치를 "유지해야 할 reference position"으로 사용
        ref_joints, ref_pos, ref_pitch = self.get_stable_state()
        ref_x = ref_pos[0]
        ref_z = ref_pos[2]

        rospy.loginfo("Correction reference x=%.4f z=%.4f pitch=%.4f",
                      ref_x, ref_z, ref_pitch)

        for it in range(self.max_correction_iters):
            cur_joints, cur_pos, cur_pitch = self.get_stable_state()

            err_x = ref_x - cur_pos[0]
            err_z = ref_z - cur_pos[2]
            err_pitch = wrap_to_pi(target_pitch - cur_pitch)

            rospy.loginfo(
                "[Iter %d] pos_err=(%.5f, %.5f), pitch_err=%.5f deg",
                it + 1, err_x, err_z, math.degrees(err_pitch)
            )

            if (abs(err_x) < self.pos_tol and
                abs(err_z) < self.pos_tol and
                abs(err_pitch) < self.pitch_tol):
                rospy.loginfo("Correction converged")
                return True

            # MoveIt Jacobian: 6 x N
            J_full = np.array(self.group.get_jacobian_matrix(cur_joints), dtype=float)

            # x, z, pitch 행 / q2,q3,q5 열만 사용
            rows = [self.jacobian_x_row, self.jacobian_z_row, self.jacobian_pitch_row]
            cols = [self.q2_idx, self.q3_idx, self.q5_idx]
            J_sel = J_full[np.ix_(rows, cols)]

            err_vec = np.array([
                self.k_pos_x * err_x,
                self.k_pos_z * err_z,
                self.k_pitch * err_pitch
            ], dtype=float)

            dq_sel = self.solve_weighted_dls(J_sel, err_vec)

            # joint별 step limit
            dq_sel[0] = np.clip(dq_sel[0], -self.max_step_q2, self.max_step_q2)
            dq_sel[1] = np.clip(dq_sel[1], -self.max_step_q3, self.max_step_q3)
            dq_sel[2] = np.clip(dq_sel[2], -self.max_step_q5, self.max_step_q5)

            rospy.loginfo(
                "[Iter %d] dq(q2,q3,q5) = (%.5f, %.5f, %.5f) rad",
                it + 1, dq_sel[0], dq_sel[1], dq_sel[2]
            )

            next_goal = list(cur_joints)
            next_goal[self.q2_idx] += dq_sel[0]
            next_goal[self.q3_idx] += dq_sel[1]
            next_goal[self.q5_idx] += dq_sel[2]

            ok = self.execute_joint_goal(next_goal)
            if not ok:
                rospy.logerr("Correction execution failed at iter %d", it + 1)
                return False

        # 마지막 한 번 더 체크
        cur_joints, cur_pos, cur_pitch = self.get_stable_state()
        err_x = ref_x - cur_pos[0]
        err_z = ref_z - cur_pos[2]
        err_pitch = wrap_to_pi(target_pitch - cur_pitch)

        rospy.logwarn(
            "Max correction iterations reached. final err: x=%.5f z=%.5f pitch=%.5f deg",
            err_x, err_z, math.degrees(err_pitch)
        )

        return (abs(err_x) < self.pos_tol and
                abs(err_z) < self.pos_tol and
                abs(err_pitch) < self.pitch_tol)

    # ------------------------------------------------------------
    # 초기 이동 + 보정
    # ------------------------------------------------------------
    def run(self):
        # 초기 joint goal이 있으면 먼저 이동
        if isinstance(self.initial_joint_goal, list) and len(self.initial_joint_goal) > 0:
            rospy.loginfo("Initial joint goal received: %s", str(self.initial_joint_goal))

            ok = self.execute_joint_goal(self.initial_joint_goal)
            if not ok:
                rospy.logerr("Initial move failed")
                return False

        # 여기서 0.5초 정지 + 평균 읽기 + Jacobian 보정 수행
        ok = self.level_with_weighted_jacobian(self.target_pitch)
        if ok:
            rospy.loginfo("Weighted Jacobian leveling success")
        else:
            rospy.logwarn("Weighted Jacobian leveling ended with residual error")

        return ok


def main():
    node = WeightedJacobianLeveler()
    node.run()


if __name__ == "__main__":
    main()
