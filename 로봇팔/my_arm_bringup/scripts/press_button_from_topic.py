#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""press_button_from_topic_B.py

B안: PRE→PUSH→PRE 전체를 '하나의 JointTrajectory'로 만들어
/arm_controller/follow_joint_trajectory 로 한 번에 전송합니다.

- Subscribes:
  * ~target_pose_topic   (geometry_msgs/PoseStamped)
  * ~target_normal_topic (geometry_msgs/Vector3Stamped)
- TF transform to ~base_frame (default: base_link)
- IK: /compute_ik (seed는 직전 해를 사용해 브랜치 점프 감소)
- Execution: actionlib FollowJointTrajectory (한 번 전송)

Run:
  rosrun my_arm_bringup press_button_from_topic_B.py _align:=false
"""

import sys
import math
import rospy
import numpy as np
import actionlib
import moveit_commander

import tf2_ros
import tf2_geometry_msgs  # noqa: F401

from geometry_msgs.msg import PoseStamped, Vector3Stamped, Quaternion
from moveit_msgs.srv import GetPositionIK, GetPositionIKRequest
from tf.transformations import quaternion_from_matrix

from control_msgs.msg import FollowJointTrajectoryAction, FollowJointTrajectoryGoal
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint


def norm3(x, y, z):
    return math.sqrt(x*x + y*y + z*z)

def normalize3(x, y, z):
    n = norm3(x, y, z)
    if n < 1e-9:
        return (1.0, 0.0, 0.0)
    return (x/n, y/n, z/n)

def _unit(v):
    v = np.array(v, dtype=float)
    n = np.linalg.norm(v)
    if n < 1e-9:
        return v
    return v / n

def quat_from_forward_y(tool_forward_in_base, world_up=(0, 0, 1)):
    """EE의 +Y가 tool_forward 방향을 바라보도록 quaternion 생성."""
    f = _unit(tool_forward_in_base)  # forward (+Y)
    up = _unit(world_up)
    if abs(float(np.dot(f, up))) > 0.98:
        up = _unit((1, 0, 0))
    r = _unit(np.cross(up, f))  # right (+X)
    u = _unit(np.cross(f, r))   # up (+Z)

    R = np.eye(4)
    R[0, 0:3] = r
    R[1, 0:3] = f
    R[2, 0:3] = u
    q = quaternion_from_matrix(R)  # (x,y,z,w)
    return Quaternion(x=q[0], y=q[1], z=q[2], w=q[3])

def compute_ik(ik_srv, group_name, ik_link, frame_id, target_xyz,
               seed_names, seed_pos, timeout=1.0,
               use_orientation=False, q_des=None):
    req = GetPositionIKRequest()
    req.ik_request.group_name = group_name
    req.ik_request.ik_link_name = ik_link
    req.ik_request.avoid_collisions = False

    req.ik_request.robot_state.joint_state.name = list(seed_names)
    req.ik_request.robot_state.joint_state.position = list(seed_pos)

    req.ik_request.pose_stamped.header.frame_id = frame_id
    req.ik_request.pose_stamped.pose.position.x = float(target_xyz[0])
    req.ik_request.pose_stamped.pose.position.y = float(target_xyz[1])
    req.ik_request.pose_stamped.pose.position.z = float(target_xyz[2])

    if use_orientation and (q_des is not None):
        req.ik_request.pose_stamped.pose.orientation = q_des
    else:
        req.ik_request.pose_stamped.pose.orientation.x = 0.0
        req.ik_request.pose_stamped.pose.orientation.y = 0.0
        req.ik_request.pose_stamped.pose.orientation.z = 0.0
        req.ik_request.pose_stamped.pose.orientation.w = 1.0

    req.ik_request.timeout.secs = int(timeout)
    req.ik_request.timeout.nsecs = int((timeout - int(timeout)) * 1e9)
    return ik_srv(req)

def extract_positions(solution_joint_state, wanted_joint_names):
    """IK 결과 JointState에서 원하는 관절들만 순서대로 뽑기."""
    name_to_pos = {n: p for n, p in zip(solution_joint_state.name, solution_joint_state.position)}
    pos = []
    missing = []
    for jn in wanted_joint_names:
        if jn not in name_to_pos:
            missing.append(jn)
        else:
            pos.append(float(name_to_pos[jn]))
    if missing:
        raise KeyError("IK solution missing joints: " + ", ".join(missing))
    return pos


class PressButtonFromTopicB:
    def __init__(self):
        rospy.init_node("press_button_from_topic_B", anonymous=True)
        moveit_commander.roscpp_initialize(sys.argv)

        # Params
        self.group_name = rospy.get_param("~group", "manipulator")
        self.base_frame = rospy.get_param("~base_frame", "base_link")

        self.pose_topic = rospy.get_param("~target_pose_topic", "/yolo_pick/target_pose")
        self.normal_topic = rospy.get_param("~target_normal_topic", "/yolo_pick/target_normal")

        self.controller_action = rospy.get_param("~controller_action", "/arm_controller/follow_joint_trajectory")

        self.tool_offset = float(rospy.get_param("~tool_offset", 0.05))
        self.d_pre = float(rospy.get_param("~pre", 0.05))
        self.d_push = float(rospy.get_param("~push", 0.006))
        self.step = float(rospy.get_param("~step", 0.01))

        # timing (seconds)
        self.pre_time = float(rospy.get_param("~pre_time", 2.0))          # current -> PRE
        self.step_dt = float(rospy.get_param("~step_dt", 0.25))           # per step
        self.retract_time = float(rospy.get_param("~retract_time", 1.5))  # back to PRE

        self.align = bool(rospy.get_param("~align", False))
        self.ik_timeout = float(rospy.get_param("~ik_timeout", 1.0))

        self.auto_run = bool(rospy.get_param("~auto_run", True))
        self.repeat = bool(rospy.get_param("~repeat", False))
        self.max_msg_age = float(rospy.get_param("~max_msg_age", 2.0))

        # MoveIt
        self.group = moveit_commander.MoveGroupCommander(self.group_name)
        self.planning_frame = self.group.get_planning_frame()
        self.eef = self.group.get_end_effector_link()
        self.joint_names = list(self.group.get_active_joints())

        rospy.loginfo("planning frame: %s", self.planning_frame)
        rospy.loginfo("eef link      : %s", self.eef)
        rospy.loginfo("base_frame    : %s", self.base_frame)
        rospy.loginfo("pose_topic    : %s", self.pose_topic)
        rospy.loginfo("normal_topic  : %s", self.normal_topic)
        rospy.loginfo("controller    : %s", self.controller_action)
        rospy.loginfo("active joints : %s", ", ".join(self.joint_names))

        # TF2
        self.tfbuf = tf2_ros.Buffer(cache_time=rospy.Duration(5.0))
        self.tfl = tf2_ros.TransformListener(self.tfbuf)

        # IK service
        rospy.wait_for_service("/compute_ik")
        self.ik_srv = rospy.ServiceProxy("/compute_ik", GetPositionIK)

        # Trajectory action client
        self.client = actionlib.SimpleActionClient(self.controller_action, FollowJointTrajectoryAction)
        rospy.loginfo("Waiting for trajectory action server ...")
        if not self.client.wait_for_server(rospy.Duration(10.0)):
            rospy.logerr("FollowJointTrajectory action server not available: %s", self.controller_action)
        else:
            rospy.loginfo("Trajectory action server connected.")

        self.last_pose = None
        self.last_normal = None
        self.running = False
        self.done_once = False

        rospy.Subscriber(self.pose_topic, PoseStamped, self._cb_pose, queue_size=1)
        rospy.Subscriber(self.normal_topic, Vector3Stamped, self._cb_normal, queue_size=1)

        rospy.loginfo("Waiting for target_pose + target_normal ...")

    def _age_ok(self, header):
        if self.max_msg_age <= 0.0:
            return True
        if header.stamp == rospy.Time(0):
            return True
        age = (rospy.Time.now() - header.stamp).to_sec()
        return (age >= 0.0) and (age <= self.max_msg_age)

    def _cb_pose(self, msg):
        self.last_pose = msg
        if self.auto_run:
            self._try_execute()

    def _cb_normal(self, msg):
        self.last_normal = msg
        if self.auto_run:
            self._try_execute()

    def _try_execute(self):
        if self.running:
            return
        if self.done_once and (not self.repeat):
            return
        if self.last_pose is None or self.last_normal is None:
            return
        if (not self._age_ok(self.last_pose.header)) or (not self._age_ok(self.last_normal.header)):
            return

        self.running = True
        try:
            self._execute(self.last_pose, self.last_normal)
            self.done_once = True
        finally:
            self.running = False
            if not self.repeat:
                self.last_pose = None
                self.last_normal = None

    def _execute(self, pose_cam, normal_cam):
        if not self.client.wait_for_server(rospy.Duration(0.1)):
            rospy.logerr("Action server not ready: %s", self.controller_action)
            return

        # TF to base_frame
        try:
            pose_b = self.tfbuf.transform(pose_cam, self.base_frame, rospy.Duration(0.2))
            n_b = self.tfbuf.transform(normal_cam, self.base_frame, rospy.Duration(0.2))
        except Exception as e:
            rospy.logwarn("TF transform failed: %s", str(e))
            return

        bx, by, bz = pose_b.pose.position.x, pose_b.pose.position.y, pose_b.pose.position.z
        nx, ny, nz = normalize3(n_b.vector.x, n_b.vector.y, n_b.vector.z)

        q_des = None
        if self.align:
            tool_forward = -_unit((nx, ny, nz))
            q_des = quat_from_forward_y(tool_forward, world_up=(0, 0, 1))
            rospy.loginfo("ALIGN: tool_forward=-n (%.3f %.3f %.3f)", tool_forward[0], tool_forward[1], tool_forward[2])

        # points
        p_btn = (bx, by, bz)
        p_eef_target = (bx - nx*self.tool_offset, by - ny*self.tool_offset, bz - nz*self.tool_offset)
        p_pre = (p_eef_target[0] - nx*self.d_pre, p_eef_target[1] - ny*self.d_pre, p_eef_target[2] - nz*self.d_pre)
        p_push = (p_eef_target[0] + nx*self.d_push, p_eef_target[1] + ny*self.d_push, p_eef_target[2] + nz*self.d_push)

        rospy.loginfo("BTN(%s): (%.4f %.4f %.4f)", self.base_frame, *p_btn)
        rospy.loginfo("n(%s,outward): (%.3f %.3f %.3f)", self.base_frame, nx, ny, nz)
        rospy.loginfo("PRE:  (%.4f %.4f %.4f)", *p_pre)
        rospy.loginfo("PUSH: (%.4f %.4f %.4f)", *p_push)

        dx, dy, dz = (p_push[0]-p_pre[0], p_push[1]-p_pre[1], p_push[2]-p_pre[2])
        dist = math.sqrt(dx*dx + dy*dy + dz*dz)
        N = max(1, int(math.ceil(dist / max(self.step, 1e-6))))
        rospy.loginfo("Trajectory steps: dist=%.4f step=%.4f N=%d", dist, self.step, N)

        # IK seed: current
        seed_names = list(self.joint_names)
        seed_pos = list(self.group.get_current_joint_values())

        # PRE IK
        res_pre = compute_ik(self.ik_srv, self.group_name, self.eef, self.planning_frame, p_pre,
                             seed_names, seed_pos, timeout=self.ik_timeout,
                             use_orientation=self.align, q_des=q_des)
        if res_pre.error_code.val != 1:
            rospy.logerr("IK(pre) failed, error=%d", res_pre.error_code.val)
            return

        try:
            q_pre = extract_positions(res_pre.solution.joint_state, self.joint_names)
        except Exception as e:
            rospy.logerr("Extract pre joints failed: %s", str(e))
            return

        traj = JointTrajectory()
        traj.joint_names = list(self.joint_names)
        points = []

        # PRE point at t=pre_time
        t = float(self.pre_time)
        pt_pre = JointTrajectoryPoint()
        pt_pre.positions = list(q_pre)
        pt_pre.time_from_start = rospy.Duration.from_sec(t)
        points.append(pt_pre)

        # seed update from IK solution
        seed_names = list(res_pre.solution.joint_state.name)
        seed_pos = list(res_pre.solution.joint_state.position)

        # PUSH points
        for i in range(1, N+1):
            a = float(i) / float(N)
            p_i = (p_pre[0] + dx*a, p_pre[1] + dy*a, p_pre[2] + dz*a)

            res_i = compute_ik(self.ik_srv, self.group_name, self.eef, self.planning_frame, p_i,
                               seed_names, seed_pos, timeout=self.ik_timeout,
                               use_orientation=self.align, q_des=q_des)
            if res_i.error_code.val != 1:
                rospy.logerr("IK(step %d/%d) failed, error=%d", i, N, res_i.error_code.val)
                return

            try:
                q_i = extract_positions(res_i.solution.joint_state, self.joint_names)
            except Exception as e:
                rospy.logerr("Extract joints failed at step %d/%d: %s", i, N, str(e))
                return

            t += float(self.step_dt)
            pt = JointTrajectoryPoint()
            pt.positions = list(q_i)
            pt.time_from_start = rospy.Duration.from_sec(t)
            points.append(pt)

            seed_names = list(res_i.solution.joint_state.name)
            seed_pos = list(res_i.solution.joint_state.position)

        # Retract back to PRE
        t += float(self.retract_time)
        pt_ret = JointTrajectoryPoint()
        pt_ret.positions = list(q_pre)
        pt_ret.time_from_start = rospy.Duration.from_sec(t)
        points.append(pt_ret)

        traj.points = points

        goal = FollowJointTrajectoryGoal()
        goal.trajectory = traj

        rospy.loginfo("Sending trajectory: points=%d total_time=%.2fs", len(points), t)
        self.client.send_goal(goal)
        ok = self.client.wait_for_result(rospy.Duration.from_sec(t + 5.0))
        if not ok:
            rospy.logerr("Trajectory execution timeout")
            self.client.cancel_goal()
            return

        state = self.client.get_state()
        result = self.client.get_result()
        rospy.loginfo("Trajectory finished. action_state=%d result=%s", state, str(result))


def main():
    PressButtonFromTopicB()
    rospy.spin()


if __name__ == "__main__":
    main()
