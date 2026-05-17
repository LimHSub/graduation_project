#!/usr/bin/env python3
import sys
import rospy
import moveit_commander
from geometry_msgs.msg import PoseStamped

def usage():
    print("""
Usage:
  # 현재 EE pose 출력
  rosrun my_arm_bringup go_to_xyz.py --print

  # (A) 자세 고정 + 상대이동 (지금은 이게 실패 중)
  rosrun my_arm_bringup go_to_xyz.py --rel 0.02 0 0

  # ✅ (B) position-only 상대이동 (추천: IK 되는지 판별용)
  rosrun my_arm_bringup go_to_xyz.py --relpos 0.02 0 0

  # ✅ (C) position-only 절대좌표 (base_link 기준)
  rosrun my_arm_bringup go_to_xyz.py --pos 0.20 0.00 0.15
""")

def plan_and_exec(group):
    result = group.plan()
    plan = None
    success = None

    if isinstance(result, tuple):
        # (success, plan, planning_time, error_code)
        if len(result) >= 2:
            success = result[0]
            plan = result[1]
    else:
        plan = result

    if plan is None or not hasattr(plan, "joint_trajectory") or len(plan.joint_trajectory.points) == 0:
        rospy.logerr("Planning failed: empty trajectory.")
        return False

    if success is False:
        rospy.logerr("Planning reported failure.")
        return False

    rospy.loginfo("Planning OK. Executing...")
    ok = group.execute(plan, wait=True)
    group.stop()
    group.clear_pose_targets()
    rospy.loginfo("Execute result: %s", str(ok))
    return ok

def main():
    moveit_commander.roscpp_initialize(sys.argv)
    rospy.init_node("go_to_xyz", anonymous=True)

    group = moveit_commander.MoveGroupCommander("manipulator")

    rospy.loginfo("planning frame: %s", group.get_planning_frame())
    rospy.loginfo("eef link: %s", group.get_end_effector_link())

    cur = group.get_current_pose().pose
    rospy.loginfo("current EE pos  : x=%.4f y=%.4f z=%.4f", cur.position.x, cur.position.y, cur.position.z)
    rospy.loginfo("current EE quat : x=%.4f y=%.4f z=%.4f w=%.4f",
                  cur.orientation.x, cur.orientation.y, cur.orientation.z, cur.orientation.w)

    if len(sys.argv) < 2:
        usage()
        return

    mode = sys.argv[1]

    if mode == "--print":
        return

    if mode not in ("--rel", "--abs", "--relpos", "--pos"):
        usage()
        return

    if len(sys.argv) != 5:
        usage()
        return

    a = float(sys.argv[2])
    b = float(sys.argv[3])
    c = float(sys.argv[4])

    # 플래닝 완화(성공 여부 확인용)
    group.set_planning_time(10.0)
    group.set_num_planning_attempts(50)

    if mode == "--rel":
        # 자세 고정 + 상대이동 (현재 실패 중)
        target = PoseStamped()
        target.header.frame_id = group.get_planning_frame()
        target.header.stamp = rospy.Time.now()
        target.pose.orientation = cur.orientation
        target.pose.position.x = cur.position.x + a
        target.pose.position.y = cur.position.y + b
        target.pose.position.z = cur.position.z + c
        rospy.loginfo("TARGET (REL+ORI): x=%.4f y=%.4f z=%.4f",
                      target.pose.position.x, target.pose.position.y, target.pose.position.z)
        group.set_pose_target(target)

        plan_and_exec(group)
        return

    if mode == "--abs":
        # 자세 고정 + 절대이동
        target = PoseStamped()
        target.header.frame_id = group.get_planning_frame()
        target.header.stamp = rospy.Time.now()
        target.pose.orientation = cur.orientation
        target.pose.position.x = a
        target.pose.position.y = b
        target.pose.position.z = c
        rospy.loginfo("TARGET (ABS+ORI): x=%.4f y=%.4f z=%.4f", a, b, c)
        group.set_pose_target(target)

        plan_and_exec(group)
        return

    if mode == "--relpos":
        # ✅ position-only 상대이동 (orientation 자유)
        x = cur.position.x + a
        y = cur.position.y + b
        z = cur.position.z + c
        rospy.loginfo("TARGET (REL POS-only): x=%.4f y=%.4f z=%.4f", x, y, z)

        # position-only: orientation을 아예 주지 않습니다.
        group.set_position_target([x, y, z], end_effector_link=group.get_end_effector_link())

        plan_and_exec(group)
        return

    if mode == "--pos":
        # ✅ position-only 절대좌표
        rospy.loginfo("TARGET (ABS POS-only): x=%.4f y=%.4f z=%.4f", a, b, c)
        group.set_position_target([a, b, c], end_effector_link=group.get_end_effector_link())

        plan_and_exec(group)
        return

if __name__ == "__main__":
    try:
        main()
    except rospy.ROSInterruptException:
        pass
