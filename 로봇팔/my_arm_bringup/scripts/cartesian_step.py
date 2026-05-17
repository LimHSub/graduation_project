#!/usr/bin/env python3
import sys
import rospy
import moveit_commander
from geometry_msgs.msg import Pose

def main():
    rospy.init_node("cartesian_step", anonymous=True)
    moveit_commander.roscpp_initialize(sys.argv)

    group = moveit_commander.MoveGroupCommander("manipulator")
    group.set_pose_reference_frame("base_link")
    group.set_max_velocity_scaling_factor(0.2)
    group.set_max_acceleration_scaling_factor(0.2)

    cur = group.get_current_pose().pose
    rospy.loginfo("cur xyz = %.4f %.4f %.4f", cur.position.x, cur.position.y, cur.position.z)

    # args: dx dy dz (meters)
    if len(sys.argv) != 4:
        print("usage: cartesian_step.py dx dy dz")
        sys.exit(1)
    dx, dy, dz = map(float, sys.argv[1:])

    wpose = Pose()
    wpose.position.x = cur.position.x + dx
    wpose.position.y = cur.position.y + dy
    wpose.position.z = cur.position.z + dz
    # orientation은 현재 유지
    wpose.orientation = cur.orientation

    waypoints = [wpose]
    (plan, fraction) = group.compute_cartesian_path(
        waypoints,   # waypoints
        0.005,       # eef_step (5mm)
        False
    )

    rospy.loginfo("cartesian fraction=%.3f, points=%d",
                  fraction, len(plan.joint_trajectory.points))

    if fraction < 0.5 or len(plan.joint_trajectory.points) == 0:
        rospy.logerr("Cartesian path failed/too small.")
        return

    ok = group.execute(plan, wait=True)
    rospy.loginfo("execute returned: %s", str(ok))
    group.stop()

if __name__ == "__main__":
    main()
