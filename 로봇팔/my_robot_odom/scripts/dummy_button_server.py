#!/usr/bin/env python3
import rospy
from std_srvs.srv import Empty, EmptyResponse

def handle_button(req):
    rospy.loginfo("[dummy_button] button service called")
    return EmptyResponse()

def main():
    rospy.init_node('dummy_button_server')
    
    rospy.Service('/arm_mission/button', Empty, handle_button)
    rospy.loginfo("[dummy_button] /arm_mission/button service ready")
    
    rospy.spin()

if __name__ == '__main__':
    main()
