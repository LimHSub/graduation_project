#!/usr/bin/env python3
import rospy
from std_msgs.msg import Float64
from geometry_msgs.msg import Twist
from collections import deque

class MarkerPoseController:
    def __init__(self):
        rospy.init_node('marker_pose_controller')

        # 사용할 회전값 선택 (yaw or pitch)
        self.control_mode = rospy.get_param("~control_mode", "pitch")  # 또는 'yaw'
        self.kp = rospy.get_param("~kp", 0.05)  # 회전 제어 계수
        self.stop_distance = rospy.get_param("~stop_distance", 0.4)  # 정지 거리 (단위: m)

        self.cmd_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=1)

        self.current_angle = 0.0
        self.current_distance = float('inf')
        
        self.angle_window = deque(maxlen=5)

        # 회전 제어용 topic 구독
        if self.control_mode == "yaw":
            rospy.Subscriber('/aruco/yaw', Float64, self.angle_callback)
        elif self.control_mode == "pitch":
            rospy.Subscriber('/aruco/pitch', Float64, self.angle_callback)
        else:
            rospy.logerr("Invalid control_mode. Use 'yaw' or 'pitch'.")
            rospy.signal_shutdown("Invalid control_mode")
            return

        # 거리 제어용 topic 구독
        rospy.Subscriber('/aruco/pose_z', Float64, self.distance_callback)

        rospy.loginfo(f"[marker_pose_controller] Started with control_mode = {self.control_mode}")
        rospy.spin()

    def angle_callback(self, msg):
        raw_angle = msg.data
        self.angle_window.append(raw_angle)

        # 이동 평균 계산
        smoothed_angle = sum(self.angle_window) / len(self.angle_window)
        self.current_angle = smoothed_angle
        self.publish_cmd()
        

    def distance_callback(self, msg):
        self.current_distance = msg.data
        self.publish_cmd()

    def publish_cmd(self):
        twist = Twist()

        if self.current_distance < self.stop_distance:
            rospy.loginfo("Too close to marker. Stop all motion.")
            self.cmd_pub.publish(twist)  # 속도 0으로 정지
            return

        angle = self.current_angle
        rospy.loginfo(f"[{self.control_mode.upper()}] angle: {angle:.2f}°, distance: {self.current_distance:.2f}m")

        if abs(angle) < 7.0:
            twist.linear.x = 0.1  # ← 직진 속도 설정 (너무 빠르면 줄이기)
            twist.angular.z = 0.0
            rospy.loginfo("Angle within deadzone. Moving forward.")
        else:
            twist.linear.x = 0.0
            twist.angular.z = -self.kp * angle  # 회전만

        self.cmd_pub.publish(twist)

if __name__ == '__main__':
    try:
        MarkerPoseController()
    except rospy.ROSInterruptException:
        pass

