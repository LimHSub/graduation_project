#!/usr/bin/env python3
import rospy
from std_msgs.msg import Float64
from geometry_msgs.msg import Twist
from collections import deque
import serial
import math

class MarkerPoseController:
    def __init__(self):
        rospy.init_node('marker_pose_controller')
        
        self.last_command = None
        self.last_send_time = rospy.Time.now()
        port = rospy.get_param("~port", "/dev/ttyACM0")  # 필요에 따라 포트 수정
        baud = rospy.get_param("~baud", 115200)
        self.ser = serial.Serial(port, baud, timeout=1)

        # 사용할 회전값 선택 (yaw or pitch)
        self.control_mode = rospy.get_param("~control_mode", "pose_x")
        self.angle_threshold = rospy.get_param("~angle_threshold", 0.03)  # deadzone (±3cm)
        self.stop_distance = rospy.get_param("~stop_distance", 0.4)  # 정지 거리 (단위: m)
        self.pitch_threshold = rospy.get_param("~pitch_threshold", 0.3) 

        self.cmd_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=1)

        self.current_angle = 0.0
        self.current_distance = float('inf')
        self.current_pitch = 0.0
        self.alignment_done = False  # pitch 보정용 플래그

        # 회전 제어용 topic 구독
        if self.control_mode == "pose_x":
            rospy.Subscriber('/aruco/pose_x', Float64, self.angle_callback)
        else:
            rospy.logerr("Invalid control_mode. Use 'pose_x'.")
            rospy.signal_shutdown("Invalid control_mode")
            return

        # 거리 제어용 topic 구독
        rospy.Subscriber('/aruco/pose_z', Float64, self.distance_callback)
        rospy.Subscriber('/aruco/pitch', Float64, self.pitch_callback)
        
        rospy.loginfo(f"[marker_pose_controller] Started with control_mode = {self.control_mode}")
        rospy.spin()
        
    def compute_heading_angle_deg(pose_x, pose_z):
        if pose_z == 0:
            return 0.0
        angle_rad = math.atan2(pose_x, pose_z)
        return math.degrees(angle_rad)

    def angle_callback(self, msg):
        self.current_angle = msg.data  # smoothing 제거
        rospy.loginfo(f"[DEBUG] raw_angle received: {self.current_angle:.4f}")
        self.publish_cmd()
        

    def distance_callback(self, msg):
        self.current_distance = msg.data
        self.publish_cmd()
        
    def pitch_callback(self, msg):
        self.current_pitch = msg.data
        self.publish_cmd()

    def publish_cmd(self):
        twist = Twist()

        if self.current_distance < self.stop_distance:
            if not self.alignment_done:
                pitch = self.current_pitch
                rospy.loginfo(f"[ALIGNMENT] pitch: {pitch:.2f} deg")

                if pitch > self.pitch_threshold:
                    rospy.loginfo("PITCH > 5°, Turn Backward to Align")
                    twist.angular.z = -2.0
                    self.send_serial('h')  # 예: 반시계방향 회전
                elif pitch < -self.pitch_threshold:
                    rospy.loginfo("PITCH < -5°, Turn Forward to Align")
                    twist.angular.z = 2.0
                    self.send_serial('f')  # 예: 시계방향 회전
                else:
                    rospy.loginfo("Pitch Alignment Done")
                    twist.angular.z = 0.0
                    self.send_serial('x')
                    self.alignment_done = True  # 한 번만 정렬

                self.cmd_pub.publish(twist)
            return

        angle = self.current_angle
        rospy.loginfo(f"[POS_X] angle: {angle:.2f} m, distance: {self.current_distance:.2f} m, threshold: {self.angle_threshold}")


        if abs(angle) < 0.03:   # 3cm
            twist.linear.x = 0.1  # ← 직진 속도 설정 (너무 빠르면 줄이기)
            twist.angular.z = 0.0
            rospy.loginfo("Moving forward.")
            self.send_serial('t') 
        elif angle > 0:
            # 마커가 오른쪽 → 우회전
            twist.linear.x = 0.0
            twist.angular.z = -3  # 고정된 저속 회전값
            rospy.loginfo("Turn Right (toward marker)")
            self.send_serial('f')
            
        else:
            twist.linear.x = 0.0
            twist.angular.z = 3
            rospy.loginfo("Turn Left (toward marker)")
            self.send_serial('h')

        self.cmd_pub.publish(twist)
        
        
    def send_serial(self, command):
      if self.last_command != command:
          self.ser.write(command.encode())
          #rospy.loginfo(f"[DEBUG] angle: {angle:.2f}, threshold: {self.angle_threshold}")

          self.last_command = command 

if __name__ == '__main__':
    try:
        MarkerPoseController()
    except rospy.ROSInterruptException:
        pass

