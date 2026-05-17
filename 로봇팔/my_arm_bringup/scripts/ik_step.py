#!/usr/bin/env python3
import sys
import rospy
import actionlib

from moveit_msgs.srv import GetPositionIK, GetPositionIKRequest
from control_msgs.msg import FollowJointTrajectoryAction, FollowJointTrajectoryGoal
from trajectory_msgs.msg import JointTrajectoryPoint
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import JointState

JOINTS = ['Revolute1','Revolute2','Revolute3','Revolute4','Revolute5']

def wait_joint_state(timeout=2.0):
    msg = rospy.wait_for_message('/joint_states', JointState, timeout=timeout)
    # 필요한 관절만 추출
    pos_map = {n:p for n,p in zip(msg.name, msg.position)}
    names = []
    positions = []
    for j in JOINTS:
        if j in pos_map:
            names.append(j)
            positions.append(pos_map[j])
    if len(names) != len(JOINTS):
        rospy.logerr("joint_states missing joints. got=%s", str(msg.name))
        return None, None
    return names, positions

def call_ik(x, y, z, qx, qy, qz, qw,
            frame='base_link', group='manipulator', ik_link='link55_1', timeout=1.0):

    # ✅ 현재 로봇 상태를 반드시 포함
    names, positions = wait_joint_state(timeout=3.0)
    if names is None:
        return None

    rospy.wait_for_service('/compute_ik')
    ik_srv = rospy.ServiceProxy('/compute_ik', GetPositionIK)

    req = GetPositionIKRequest()
    req.ik_request.group_name = group
    req.ik_request.ik_link_name = ik_link
    req.ik_request.avoid_collisions = False

    # ✅ robot_state 채우기
    req.ik_request.robot_state.joint_state.name = names
    req.ik_request.robot_state.joint_state.position = positions

    ps = PoseStamped()
    ps.header.frame_id = frame
    ps.pose.position.x = x
    ps.pose.position.y = y
    ps.pose.position.z = z
    ps.pose.orientation.x = qx
    ps.pose.orientation.y = qy
    ps.pose.orientation.z = qz
    ps.pose.orientation.w = qw
    req.ik_request.pose_stamped = ps

    req.ik_request.timeout.secs = int(timeout)
    req.ik_request.timeout.nsecs = int((timeout - int(timeout)) * 1e9)

    return ik_srv(req)

def send_joint_traj(joints, positions, duration=2.0):
    client = actionlib.SimpleActionClient('/arm_controller/follow_joint_trajectory', FollowJointTrajectoryAction)
    if not client.wait_for_server(rospy.Duration(5.0)):
        rospy.logerr("follow_joint_trajectory action server not available")
        return False

    goal = FollowJointTrajectoryGoal()
    goal.trajectory.joint_names = joints

    pt = JointTrajectoryPoint()
    pt.positions = positions
    pt.time_from_start = rospy.Duration(duration)
    goal.trajectory.points.append(pt)

    client.send_goal(goal)
    ok = client.wait_for_result(rospy.Duration(duration + 5.0))
    if not ok:
        rospy.logerr("Trajectory action timed out")
        return False

    res = client.get_result()
    rospy.loginfo("Trajectory result: %s", str(res))
    return True

def main():
    rospy.init_node('ik_step', anonymous=True)

    if len(sys.argv) < 4:
        print("Usage: rosrun my_arm_bringup ik_step.py x y z [duration_sec]")
        sys.exit(1)

    x = float(sys.argv[1])
    y = float(sys.argv[2])
    z = float(sys.argv[3])
    duration = float(sys.argv[4]) if len(sys.argv) >= 5 else 2.0

    # 일단 고정 orientation (현재값 기반)
    qx, qy, qz, qw = (0.0200, 0.0004, -0.0015, 0.9998)

    res = call_ik(x, y, z, qx, qy, qz, qw)
    if res is None:
        return

    if res.error_code.val != 1:
        rospy.logerr("IK failed, error_code=%d", res.error_code.val)
        return

    js = res.solution.joint_state
    if not js.name or not js.position:
        rospy.logerr("IK returned empty joint_state")
        return

    pos_map = {n:p for n,p in zip(js.name, js.position)}
    positions = [pos_map[j] for j in JOINTS if j in pos_map]
    if len(positions) != len(JOINTS):
        rospy.logerr("IK solution missing some joints. got=%s", str(js.name))
        return

    rospy.loginfo("IK solution(rad): %s", str(positions))
    send_joint_traj(JOINTS, positions, duration=duration)

if __name__ == "__main__":
    main()
