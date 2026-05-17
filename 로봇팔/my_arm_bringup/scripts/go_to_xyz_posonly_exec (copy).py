#!/usr/bin/env python3
import sys
import rospy
import moveit_commander
from moveit_commander.move_group import MoveGroupCommander
from geometry_msgs.msg import Pose
from moveit_msgs.srv import GetPositionIK, GetPositionIKRequest

def call_ik(group_name, ik_link, frame_id, x, y, z, seed_names, seed_positions, timeout=2.0):
    rospy.wait_for_service('/compute_ik')
    srv = rospy.ServiceProxy('/compute_ik', GetPositionIK)

    req = GetPositionIKRequest()
    req.ik_request.group_name = group_name
    req.ik_request.ik_link_name = ik_link
    req.ik_request.avoid_collisions = False

    # seed state (현재 관절각)
    req.ik_request.robot_state.joint_state.name = list(seed_names)
    req.ik_request.robot_state.joint_state.position = list(seed_positions)

    req.ik_request.pose_stamped.header.frame_id = frame_id
    req.ik_request.pose_stamped.pose.position.x = x
    req.ik_request.pose_stamped.pose.position.y = y
    req.ik_request.pose_stamped.pose.position.z = z

    # orientation은 의미 없지만 메시지 형식상 넣어둠 (position_only_ik=true라 무시됨)
    req.ik_request.pose_stamped.pose.orientation.w = 1.0

    req.ik_request.timeout.secs = int(timeout)
    req.ik_request.timeout.nsecs = int((timeout - int(timeout)) * 1e9)

    res = srv(req)
    return res

def main():
    rospy.init_node('go_to_xyz_posonly_exec', anonymous=True)
    moveit_commander.roscpp_initialize(sys.argv)

    if len(sys.argv) != 4:
        print("Usage: rosrun my_arm_bringup go_to_xyz_posonly_exec.py X Y Z")
        sys.exit(1)

    x = float(sys.argv[1]); y = float(sys.argv[2]); z = float(sys.argv[3])

    group = MoveGroupCommander("manipulator")

    planning_frame = group.get_planning_frame()
    eef_link = group.get_end_effector_link()

    rospy.loginfo("planning frame: %s", planning_frame)
    rospy.loginfo("eef link      : %s", eef_link)

    # 현재 관절각
    cur = group.get_current_joint_values()
    joint_names = group.get_active_joints()
    rospy.loginfo("current joints: %s", ["%s=%.4f" % (n,v) for n,v in zip(joint_names, cur)])

    # IK 호출
    ik_res = call_ik(
        group_name="manipulator",
        ik_link=eef_link,
        frame_id=planning_frame,   # 지금 출력상 base_link
        x=x, y=y, z=z,
        seed_names=joint_names,
        seed_positions=cur,
        timeout=2.0
    )

    code = ik_res.error_code.val
    if code != 1:
        rospy.logerr("IK failed, error_code=%d", code)
        sys.exit(2)

    sol = ik_res.solution.joint_state
    sol_map = dict(zip(sol.name, sol.position))

    target = []
    missing = []
    for n in joint_names:
        if n not in sol_map:
            missing.append(n)
        else:
            target.append(sol_map[n])

    if missing:
        rospy.logerr("IK solution missing joints: %s", missing)
        sys.exit(3)

    rospy.loginfo("IK target joints: %s", ["%s=%.4f" % (n,v) for n,v in zip(joint_names, target)])

    # 실행: “포인트 이동”이면 이게 제일 직관적
    group.set_joint_value_target(target)

    # planning 시간을 조금 줌
    group.set_planning_time(2.0)

    ok = group.go(wait=True)
    group.stop()
    group.clear_pose_targets()

    if not ok:
        rospy.logerr("Execution failed (group.go returned False).")
        sys.exit(4)

    rospy.loginfo("DONE")
    sys.exit(0)

if __name__ == "__main__":
    main()
