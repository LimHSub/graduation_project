#!/usr/bin/env python3
import sys
import math
import rospy
import moveit_commander
from geometry_msgs.msg import Pose
from moveit_msgs.srv import GetPositionIK, GetPositionIKRequest

def norm3(x,y,z):
    return math.sqrt(x*x+y*y+z*z)

def normalize3(x,y,z):
    n = norm3(x,y,z)
    if n < 1e-9:
        return (1.0, 0.0, 0.0)
    return (x/n, y/n, z/n)

def get_current_joint_state(group):
    jn = group.get_active_joints()
    jp = group.get_current_joint_values()
    return jn, jp

def compute_ik(ik_srv, group_name, ik_link, frame_id, target_xyz, seed_names, seed_pos, timeout=1.0):
    req = GetPositionIKRequest()
    req.ik_request.group_name = group_name
    req.ik_request.ik_link_name = ik_link
    req.ik_request.avoid_collisions = False

    req.ik_request.robot_state.joint_state.name = seed_names
    req.ik_request.robot_state.joint_state.position = seed_pos

    req.ik_request.pose_stamped.header.frame_id = frame_id
    req.ik_request.pose_stamped.pose.position.x = target_xyz[0]
    req.ik_request.pose_stamped.pose.position.y = target_xyz[1]
    req.ik_request.pose_stamped.pose.position.z = target_xyz[2]

    # position_only_ik를 쓰는 경우 orientation은 크게 의미 없어서 단위쿼터니언
    req.ik_request.pose_stamped.pose.orientation.x = 0.0
    req.ik_request.pose_stamped.pose.orientation.y = 0.0
    req.ik_request.pose_stamped.pose.orientation.z = 0.0
    req.ik_request.pose_stamped.pose.orientation.w = 1.0

    req.ik_request.timeout.secs = int(timeout)
    req.ik_request.timeout.nsecs = int((timeout - int(timeout)) * 1e9)

    res = ik_srv(req)
    return res

def go_joint_target(group, joint_names, joint_pos, wait=True):
    # MoveIt commander는 dict로 주는 게 안전합니다
    tgt = {name: pos for name, pos in zip(joint_names, joint_pos)}
    group.set_joint_value_target(tgt)
    ok = group.go(wait=wait)
    group.stop()
    group.clear_pose_targets()
    return ok

def main():
    rospy.init_node("press_button_base", anonymous=True)
    moveit_commander.roscpp_initialize(sys.argv)

    # --------- 인자 파싱 ----------
    # 사용:
    # rosrun my_arm_bringup press_button_base.py x y z nx ny nz
    # 예) rosrun my_arm_bringup press_button_base.py 0.10 -0.33 0.37  0 1 0
    if len(sys.argv) < 4:
        print("Usage: press_button_base.py x y z [nx ny nz] [--tool 0.05] [--pre 0.05] [--push 0.012] [--step 0.002]")
        return

    # 기본값
    tool_offset = 0.05
    d_pre = 0.05
    d_push = 0.012
    step = 0.002

    # 좌표
    x = float(sys.argv[1]); y = float(sys.argv[2]); z = float(sys.argv[3])

    # 법선
    nx, ny, nz = 0.0, 1.0, 0.0  # 기본값(원하시면 바꾸세요)
    idx = 4
    if len(sys.argv) >= 7 and not sys.argv[4].startswith("--"):
        nx = float(sys.argv[4]); ny = float(sys.argv[5]); nz = float(sys.argv[6])
        idx = 7

    # 옵션 파싱(간단)
    while idx < len(sys.argv):
        if sys.argv[idx] == "--tool":
            tool_offset = float(sys.argv[idx+1]); idx += 2
        elif sys.argv[idx] == "--pre":
            d_pre = float(sys.argv[idx+1]); idx += 2
        elif sys.argv[idx] == "--push":
            d_push = float(sys.argv[idx+1]); idx += 2
        elif sys.argv[idx] == "--step":
            step = float(sys.argv[idx+1]); idx += 2
        else:
            idx += 1

    nx, ny, nz = normalize3(nx, ny, nz)

    group = moveit_commander.MoveGroupCommander("manipulator")
    group.set_planning_time(2.0)
    group.set_num_planning_attempts(5)

    planning_frame = group.get_planning_frame()
    eef = group.get_end_effector_link()
    rospy.loginfo("planning frame: %s", planning_frame)
    rospy.loginfo("eef link      : %s", eef)

    rospy.wait_for_service("/compute_ik")
    ik_srv = rospy.ServiceProxy("/compute_ik", GetPositionIK)

    # 현재 joint seed
    seed_names, seed_pos = get_current_joint_state(group)

    # ------------- 목표점 계산(EEF 기준) -------------
    # 버튼점(p_btn) -> EEF가 서야 할 점(p_eef_target)
    p_btn = (x, y, z)
    p_eef_target = (
        p_btn[0] - nx * tool_offset,
        p_btn[1] - ny * tool_offset,
        p_btn[2] - nz * tool_offset
    )

    p_pre = (
        p_eef_target[0] - nx * d_pre,
        p_eef_target[1] - ny * d_pre,
        p_eef_target[2] - nz * d_pre
    )
    p_push = (
        p_eef_target[0] + nx * d_push,
        p_eef_target[1] + ny * d_push,
        p_eef_target[2] + nz * d_push
    )

    rospy.loginfo("BTN(base)     : x=%.4f y=%.4f z=%.4f", *p_btn)
    rospy.loginfo("n             : nx=%.3f ny=%.3f nz=%.3f", nx, ny, nz)
    rospy.loginfo("EEF target    : x=%.4f y=%.4f z=%.4f (tool=%.3fm)", *p_eef_target, tool_offset)
    rospy.loginfo("PRE           : x=%.4f y=%.4f z=%.4f (pre=%.3fm)", *p_pre, d_pre)
    rospy.loginfo("PUSH          : x=%.4f y=%.4f z=%.4f (push=%.3fm)", *p_push, d_push)

    # ------------- 1) PRE로 이동 -------------
    res = compute_ik(ik_srv, "manipulator", eef, planning_frame, p_pre, seed_names, seed_pos, timeout=2.0)
    if res.error_code.val != 1:
        rospy.logerr("IK(pre) failed, error=%d", res.error_code.val)
        return
    ok = go_joint_target(group, res.solution.joint_state.name, res.solution.joint_state.position)
    if not ok:
        rospy.logerr("Move(pre) failed")
        return

    # ------------- 2) PUSH를 step으로 분할 이동 -------------
    # pre -> push 직선 이동을 N step으로 쪼개기
    dx = p_push[0] - p_pre[0]
    dy = p_push[1] - p_pre[1]
    dz = p_push[2] - p_pre[2]
    dist = math.sqrt(dx*dx + dy*dy + dz*dz)
    if dist < 1e-6:
        rospy.logwarn("push distance too small")
        return
    N = max(1, int(math.ceil(dist / step)))

    rospy.loginfo("PUSH steps: dist=%.4f, step=%.4f, N=%d", dist, step, N)

    # 각 step마다 현재 joint를 seed로 사용(성공률↑)
    for i in range(1, N+1):
        a = float(i) / float(N)
        p_i = (p_pre[0] + dx*a, p_pre[1] + dy*a, p_pre[2] + dz*a)

        # seed 업데이트
        seed_names, seed_pos = get_current_joint_state(group)

        res = compute_ik(ik_srv, "manipulator", eef, planning_frame, p_i, seed_names, seed_pos, timeout=1.0)
        if res.error_code.val != 1:
            rospy.logerr("IK(step %d/%d) failed, error=%d", i, N, res.error_code.val)
            return
        ok = go_joint_target(group, res.solution.joint_state.name, res.solution.joint_state.position)
        if not ok:
            rospy.logerr("Move(step %d/%d) failed", i, N)
            return

    # ------------- 3) RETRACT(pre로 복귀) -------------
    seed_names, seed_pos = get_current_joint_state(group)
    res = compute_ik(ik_srv, "manipulator", eef, planning_frame, p_pre, seed_names, seed_pos, timeout=2.0)
    if res.error_code.val != 1:
        rospy.logerr("IK(retract) failed, error=%d", res.error_code.val)
        return
    ok = go_joint_target(group, res.solution.joint_state.name, res.solution.joint_state.position)
    if not ok:
        rospy.logerr("Move(retract) failed")
        return

    rospy.loginfo("DONE (press sequence)")

if __name__ == "__main__":
    main()
