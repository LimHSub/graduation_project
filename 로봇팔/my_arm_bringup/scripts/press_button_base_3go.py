#!/usr/bin/env python3
import sys
import math
import rospy
import moveit_commander
from geometry_msgs.msg import Pose

def norm3(x, y, z):
    n = math.sqrt(x*x + y*y + z*z)
    if n < 1e-9:
        return (0.0, 0.0, 0.0, False)
    return (x/n, y/n, z/n, True)

def pose_xyz(x, y, z, qx=0.0, qy=0.0, qz=0.0, qw=1.0):
    p = Pose()
    p.position.x = x
    p.position.y = y
    p.position.z = z
    p.orientation.x = qx
    p.orientation.y = qy
    p.orientation.z = qz
    p.orientation.w = qw
    return p

def move_pose(group, pose, label, timeout=6.0):
    group.set_pose_target(pose)
    ok = group.go(wait=True)
    group.stop()
    group.clear_pose_targets()
    if not ok:
        rospy.logerr("Move failed at [%s]", label)
    return ok

def main():
    rospy.init_node("press_button_base_3go", anonymous=True)

    # 사용법:
    # rosrun my_arm_bringup press_button_base_3go.py bx by bz nx ny nz [--tool 0.05] [--pre 0.05] [--push 0.012]
    argv = rospy.myargv(sys.argv)

    if len(argv) < 7:
        rospy.logerr("Usage: %s bx by bz nx ny nz [--tool 0.05] [--pre 0.05] [--push 0.012] [--v 0.2] [--a 0.2] [--timeout 6.0]",
                     argv[0])
        sys.exit(1)

    bx = float(argv[1]); by = float(argv[2]); bz = float(argv[3])
    nx = float(argv[4]); ny = float(argv[5]); nz = float(argv[6])

    # 기본 파라미터 (원하시면 값만 바꿔서 쓰시면 됩니다)
    tool = 0.05   # 버튼 표면에서 EEF(link55_1)까지 "툴 길이" 가정 (m)
    pre  = 0.05   # 버튼 누르기 전 대기 거리 (m)
    push = 0.012  # 실제 누르는 전진 거리 (m)
    vscale = 0.2  # 속도 스케일(0~1)
    ascale = 0.2  # 가속 스케일(0~1)
    timeout = 6.0

    i = 7
    while i < len(argv):
        if argv[i] == "--tool":
            tool = float(argv[i+1]); i += 2
        elif argv[i] == "--pre":
            pre = float(argv[i+1]); i += 2
        elif argv[i] == "--push":
            push = float(argv[i+1]); i += 2
        elif argv[i] == "--v":
            vscale = float(argv[i+1]); i += 2
        elif argv[i] == "--a":
            ascale = float(argv[i+1]); i += 2
        elif argv[i] == "--timeout":
            timeout = float(argv[i+1]); i += 2
        else:
            rospy.logwarn("Unknown arg: %s", argv[i])
            i += 1

    nx, ny, nz, ok = norm3(nx, ny, nz)
    if not ok:
        rospy.logerr("Normal vector is zero. nx ny nz cannot be all 0.")
        sys.exit(1)

    moveit_commander.roscpp_initialize(sys.argv)
    group = moveit_commander.MoveGroupCommander("manipulator")

    rospy.loginfo("planning frame: %s", group.get_planning_frame())
    rospy.loginfo("eef link      : %s", group.get_end_effector_link())

    # 속도/가속 스케일(너무 크면 CONTROL_FAILED 가능성 커집니다)
    group.set_max_velocity_scaling_factor(vscale)
    group.set_max_acceleration_scaling_factor(ascale)

    # 현재 EE 자세(orientation)는 그대로 유지하는 편이 보통 안전합니다.
    cur = group.get_current_pose().pose
    qx, qy, qz, qw = cur.orientation.x, cur.orientation.y, cur.orientation.z, cur.orientation.w

    rospy.loginfo("BTN(base)     : x=%.4f y=%.4f z=%.4f", bx, by, bz)
    rospy.loginfo("n             : nx=%.3f ny=%.3f nz=%.3f", nx, ny, nz)
    rospy.loginfo("tool=%.3f pre=%.3f push=%.3f v=%.2f a=%.2f timeout=%.2f",
                  tool, pre, push, vscale, ascale, timeout)

    # 핵심 계산:
    # - 버튼 위치(BTN)는 "툴팁"이 닿아야 하는 점이라고 가정
    # - EEF(link55_1)는 버튼에서 -n 방향으로 tool 만큼 뒤에 있어야 함
    # - PRE는 거기서 -n 방향으로 pre 만큼 더 뒤
    # - PUSH는 EEF target에서 +n 방향으로 push 만큼 전진 (버튼을 누르는 동작)
    eef_x = bx - nx * tool
    eef_y = by - ny * tool
    eef_z = bz - nz * tool

    pre_x = eef_x - nx * pre
    pre_y = eef_y - ny * pre
    pre_z = eef_z - nz * pre

    push_x = eef_x + nx * push
    push_y = eef_y + ny * push
    push_z = eef_z + nz * push

    rospy.loginfo("EEF target    : x=%.4f y=%.4f z=%.4f", eef_x, eef_y, eef_z)
    rospy.loginfo("PRE           : x=%.4f y=%.4f z=%.4f", pre_x, pre_y, pre_z)
    rospy.loginfo("PUSH          : x=%.4f y=%.4f z=%.4f", push_x, push_y, push_z)

    pre_pose  = pose_xyz(pre_x,  pre_y,  pre_z,  qx, qy, qz, qw)
    push_pose = pose_xyz(push_x, push_y, push_z, qx, qy, qz, qw)
    back_pose = pre_pose  # 복귀는 PRE로

    # 3번만 go()
    rospy.loginfo("1) Move to PRE")
    if not move_pose(group, pre_pose, "PRE", timeout=timeout):
        sys.exit(2)

    rospy.loginfo("2) Move to PUSH")
    if not move_pose(group, push_pose, "PUSH", timeout=timeout):
        sys.exit(3)

    rospy.loginfo("3) Back to PRE")
    if not move_pose(group, back_pose, "BACK(PRE)", timeout=timeout):
        sys.exit(4)

    rospy.loginfo("DONE")

if __name__ == "__main__":
    main()
