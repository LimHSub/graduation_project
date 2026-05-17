#!/usr/bin/env python3
import sys
import rospy
import actionlib
from control_msgs.msg import FollowJointTrajectoryAction, FollowJointTrajectoryGoal
from trajectory_msgs.msg import JointTrajectoryPoint

def main():
    rospy.init_node("joint_goal_once")

    # joint 이름은 /joint_states와 동일해야 합니다.
    joint_names = ["Revolute1","Revolute2","Revolute3","Revolute4","Revolute5"]

    # 예: 약간만 움직이기(라디안)
    # 사용자가 원하는 값으로 바꾸세요.
    target = [0.2, -0.8, -0.6, 2.8, -1.6]

    client = actionlib.SimpleActionClient("/arm_controller/follow_joint_trajectory", FollowJointTrajectoryAction)
    rospy.loginfo("waiting action server...")
    client.wait_for_server()

    goal = FollowJointTrajectoryGoal()
    goal.trajectory.joint_names = joint_names

    p = JointTrajectoryPoint()
    p.positions = target
    p.time_from_start = rospy.Duration(2.0)  # 2초에 도달
    goal.trajectory.points = [p]

    rospy.loginfo("send joint goal")
    client.send_goal(goal)
    ok = client.wait_for_result(rospy.Duration(5.0))
    rospy.loginfo("done=%s state=%s", ok, client.get_state())

if __name__ == "__main__":
    main()
