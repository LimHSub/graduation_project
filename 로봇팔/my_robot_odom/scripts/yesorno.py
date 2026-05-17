#!/usr/bin/env python3
import rospy
from actionlib_msgs.msg import GoalStatusArray

def status_cb(msg):
    if msg.status_list:
        last = msg.status_list[-1]
        if last.status == 3:
            print("\033[1;32m>>> GOAL REACHED! <<<\033[0m")  # 색깔 강조

rospy.init_node('goal_reached_print')
rospy.Subscriber('/move_base/status', GoalStatusArray, status_cb)
rospy.spin()

