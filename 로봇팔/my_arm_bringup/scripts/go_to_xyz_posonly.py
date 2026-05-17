#!/usr/bin/env python3
import sys
import rospy
import moveit_commander
from geometry_msgs.msg import PoseStamped

def usage():
    print("Usage:")
    print("  rosrun my_arm_bringup go_to_xyz_posonly.py x y z")
    print("  rosrun my_arm_bringup go_to_xyz_posonly.py --rel dx dy dz")
    print("  rosrun my_arm_bringup go_to_xyz_posonly.py --print")
    sys.exit(1)

def main():
    moveit_commander.roscpp_initialize(sys.argv)
    rospy.init_node("go_to_xyz_posonly", anonymous=True)

    group_name = rospy.get_param("~group", "manipulator")
    group = moveit_commander.MoveGroupCommander(group_name)

    # 기본 튜닝(너무 작으면 실패/타임아웃 잘 납니다)
    group.set_planning_time(10.0)
    group.set_num_planning_attempts(20)
    group.set_max_velocity_scaling_factor(0.3)
    group.set_max_acceleration_scaling_factor(0.3)

    # 허용오차(빡빡하면 실패 잘 납니다)
    group.set_goal_position_tolerance(0.005)  # 5mm
    # orientation은 아예 강제하지 않을 거라 별 의미 없지만, 혹시 내부에서 걸리면 넉넉히
    group.set_goal_orientation_tolerance(3.14159)

    planning_frame = group.get_planning_frame()
    eef = group.get_end_effector_link()
    rospy.loginfo("planning frame: %s", planning_frame)
    rospy.loginfo("eef link: %s", eef)

    # 현재 EE 출력
    cur_pose = group.get_current_pose(eef).pose
    rospy.loginfo("current EE pos  : x=%.4f y=%.4f z=%.4f", cur_pose.position.x, cur_pose.position.y, cur_pose.position.z)

    if len(sys.argv) == 2 and sys.argv[1] == "--print":
        return

    # 목표 좌표 계산
    if len(sys.argv) == 5 and sys.argv[1] == "--rel":
        dx, dy, dz = map(float, sys.argv[2:5])
        tx = cur_pose.position.x + dx
        ty = cur_pose.position.y + dy
        tz = cur_pose.position.z + dz
        rospy.loginfo("TARGET (REL pos-only): x=%.4f y=%.4f z=%.4f", tx, ty, tz)
    elif len(sys.argv) == 4:
        tx, ty, tz = map(float, sys.argv[1:4])
        rospy.loginfo("TARGET (ABS pos-only): x=%.4f y=%.4f z=%.4f", tx, ty, tz)
    else:
        usage()

    # ★ 핵심: pose_target(자세 포함) 대신 position_target만 사용
    group.clear_pose_targets()
    group.set_position_target([tx, ty, tz], eef)

    plan = group.plan()

    # Noetic/moveit_commander는 버전에 따라 plan이 tuple일 수 있음
    traj = None
    if isinstance(plan, tuple):
        # (success_flag, plan, planning_time, error_code) 형태가 흔함
        for item in plan:
            if hasattr(item, "joint_trajectory"):
                traj = item
                break
        if traj is None:
            # 두 번째가 plan인 케이스
            if len(plan) > 1 and hasattr(plan[1], "joint_trajectory"):
                traj = plan[1]
    else:
        traj = plan

    if traj is None or not hasattr(traj, "joint_trajectory") or len(traj.joint_trajectory.points) == 0:
        rospy.logerr("Planning failed: empty trajectory.")
        return

    rospy.loginfo("Plan OK. Executing...")
    ok = group.execute(traj, wait=True)
    group.stop()
    group.clear_pose_targets()

    rospy.loginfo("Execute result: %s", str(ok))

if __name__ == "__main__":
    main()

