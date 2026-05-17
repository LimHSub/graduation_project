#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""press_button_from_topic.py

- Subscribes:
  * ~target_pose_topic   (geometry_msgs/PoseStamped)   : button pose (camera frame OK)
  * ~target_normal_topic (geometry_msgs/Vector3Stamped): panel outward normal (camera frame OK)
- Transforms both to ~base_frame (default: base_link) using tf2
  * Pose: rotation + translation
  * Normal: rotation only (Vector3Stamped)
- Executes press sequence using /compute_ik + joint-go (no MoveIt Cartesian planning)

Run:
  rosrun my_arm_bringup press_button_from_topic.py _align:=true
"""

import sys
import math
import rospy
import numpy as np
import moveit_commander

import tf2_ros
import tf2_geometry_msgs  # noqa: F401 (registers conversions)

from geometry_msgs.msg import PoseStamped, Vector3Stamped, Quaternion
from moveit_msgs.srv import GetPositionIK, GetPositionIKRequest
from tf.transformations import quaternion_from_matrix


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
    """EE의 +Y(전방)이 tool_forward 방향을 바라보도록 하는 quaternion 생성."""
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

def get_current_joint_state(group):
    jn = group.get_active_joints()
    jp = group.get_current_joint_values()
    return jn, jp

def compute_ik(ik_srv, group_name, ik_link, frame_id, target_xyz,
               seed_names, seed_pos, timeout=1.0,
               use_orientation=False, q_des=None):
    req = GetPositionIKRequest()
    req.ik_request.group_name = group_name
    req.ik_request.ik_link_name = ik_link
    req.ik_request.avoid_collisions = False

    req.ik_request.robot_state.joint_state.name = seed_names
    req.ik_request.robot_state.joint_state.position = seed_pos

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

def go_joint_target(group, joint_names, joint_pos, wait=True):
    tgt = {name: pos for name, pos in zip(joint_names, joint_pos)}
    group.set_joint_value_target(tgt)
    ok = group.go(wait=wait)
    group.stop()
    group.clear_pose_targets()
    return ok


class PressButtonFromTopic:
    def __init__(self):
        rospy.init_node("press_button_from_topic", anonymous=True)
        moveit_commander.roscpp_initialize(sys.argv)

        # Params
        self.group_name = rospy.get_param("~group", "manipulator")
        self.base_frame = rospy.get_param("~base_frame", "base_link")
        self.pose_topic = rospy.get_param("~target_pose_topic", "/yolo_pick/target_pose")
        self.normal_topic = rospy.get_param("~target_normal_topic", "/yolo_pick/target_normal")

        self.tool_offset = float(rospy.get_param("~tool_offset", 0.05))
        self.d_pre = float(rospy.get_param("~pre", 0.05))
        self.d_push = float(rospy.get_param("~push", 0.012))
        self.step = float(rospy.get_param("~step", 0.002))

        self.align = bool(rospy.get_param("~align", False))
        self.ik_timeout_pre = float(rospy.get_param("~ik_timeout_pre", 2.0))
        self.ik_timeout_step = float(rospy.get_param("~ik_timeout_step", 1.0))

        self.auto_run = bool(rospy.get_param("~auto_run", True))
        self.repeat = bool(rospy.get_param("~repeat", False))
        self.max_msg_age = float(rospy.get_param("~max_msg_age", 2.0))

        self.group = moveit_commander.MoveGroupCommander(self.group_name)
        self.planning_frame = self.group.get_planning_frame()
        self.eef = self.group.get_end_effector_link()

        rospy.loginfo("planning frame: %s", self.planning_frame)
        rospy.loginfo("eef link      : %s", self.eef)
        rospy.loginfo("base_frame    : %s", self.base_frame)
        rospy.loginfo("pose_topic    : %s", self.pose_topic)
        rospy.loginfo("normal_topic  : %s", self.normal_topic)

        self.tfbuf = tf2_ros.Buffer(cache_time=rospy.Duration(5.0))
        self.tfl = tf2_ros.TransformListener(self.tfbuf)

        rospy.wait_for_service("/compute_ik")
        self.ik_srv = rospy.ServiceProxy("/compute_ik", GetPositionIK)

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
            self._execute_press(self.last_pose, self.last_normal)
            self.done_once = True
        finally:
            self.running = False
            if not self.repeat:
                self.last_pose = None
                self.last_normal = None

    def _execute_press(self, pose_cam, normal_cam):
        # TF transform to base_frame
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
            tool_forward = -_unit((nx, ny, nz))  # EE +Y looks toward panel inward
            q_des = quat_from_forward_y(tool_forward, world_up=(0, 0, 1))
            rospy.loginfo("ALIGN: tool_forward=-n (%.3f %.3f %.3f)", tool_forward[0], tool_forward[1], tool_forward[2])

        # Points
        p_btn = (bx, by, bz)
        p_eef_target = (bx - nx*self.tool_offset, by - ny*self.tool_offset, bz - nz*self.tool_offset)
        p_pre = (p_eef_target[0] - nx*self.d_pre, p_eef_target[1] - ny*self.d_pre, p_eef_target[2] - nz*self.d_pre)
        p_push = (p_eef_target[0] + nx*self.d_push, p_eef_target[1] + ny*self.d_push, p_eef_target[2] + nz*self.d_push)

        rospy.loginfo("BTN(%s): (%.4f %.4f %.4f)", self.base_frame, *p_btn)
        rospy.loginfo("n(%s,outward): (%.3f %.3f %.3f)", self.base_frame, nx, ny, nz)
        rospy.loginfo("PRE:  (%.4f %.4f %.4f)", *p_pre)
        rospy.loginfo("PUSH: (%.4f %.4f %.4f)", *p_push)

        # IK seed
        seed_names, seed_pos = get_current_joint_state(self.group)

        # PRE move
        res = compute_ik(self.ik_srv, self.group_name, self.eef, self.planning_frame, p_pre,
                         seed_names, seed_pos, timeout=self.ik_timeout_pre,
                         use_orientation=self.align, q_des=q_des)
        if res.error_code.val != 1:
            rospy.logerr("IK(pre) failed, error=%d", res.error_code.val)
            return
        if not go_joint_target(self.group, res.solution.joint_state.name, res.solution.joint_state.position):
            rospy.logerr("Move(pre) failed")
            return

        # Step push
        dx, dy, dz = (p_push[0]-p_pre[0], p_push[1]-p_pre[1], p_push[2]-p_pre[2])
        dist = math.sqrt(dx*dx + dy*dy + dz*dz)
        N = max(1, int(math.ceil(dist / max(self.step, 1e-6))))
        rospy.loginfo("PUSH steps: dist=%.4f step=%.4f N=%d", dist, self.step, N)

        seed_names = res.solution.joint_state.name
        seed_pos = res.solution.joint_state.position

        for i in range(1, N+1):
            a = float(i)/float(N)
            p_i = (p_pre[0] + dx*a, p_pre[1] + dy*a, p_pre[2] + dz*a)
            res_i = compute_ik(self.ik_srv, self.group_name, self.eef, self.planning_frame, p_i,
                               seed_names, seed_pos, timeout=self.ik_timeout_step,
                               use_orientation=self.align, q_des=q_des)
            if res_i.error_code.val != 1:
                rospy.logerr("IK(step %d/%d) failed, error=%d", i, N, res_i.error_code.val)
                return
            if not go_joint_target(self.group, res_i.solution.joint_state.name, res_i.solution.joint_state.position):
                rospy.logerr("Move(step %d/%d) failed", i, N)
                return
            seed_names = res_i.solution.joint_state.name
            seed_pos = res_i.solution.joint_state.position

        # Retract to PRE
        res_r = compute_ik(self.ik_srv, self.group_name, self.eef, self.planning_frame, p_pre,
                           seed_names, seed_pos, timeout=self.ik_timeout_pre,
                           use_orientation=self.align, q_des=q_des)
        if res_r.error_code.val != 1:
            rospy.logerr("IK(retract) failed, error=%d", res_r.error_code.val)
            return
        if not go_joint_target(self.group, res_r.solution.joint_state.name, res_r.solution.joint_state.position):
            rospy.logerr("Move(retract) failed")
            return

        rospy.loginfo("DONE press sequence")


def main():
    PressButtonFromTopic()
    rospy.spin()


if __name__ == "__main__":
    main()
