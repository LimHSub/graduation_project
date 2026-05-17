#!/usr/bin/env python3
import rospy
from std_msgs.msg import Float64
from geometry_msgs.msg import Twist
import math
import serial
import numpy as np

class MarkerPoseController:
    def __init__(self):
        rospy.init_node('marker_pose_controller')
        self.curve_generated = False  # 이거 추가!

        self.curve_points = []
        self.reached = False
        self.last_command = None

        self.pose_x = 0.0
        self.pose_z = float('inf')

        # 파라미터
        port = rospy.get_param("~port", "/dev/ttyACM0")
        baud = rospy.get_param("~baud", 115200)
        self.ser = serial.Serial(port, baud, timeout=1)

        self.angle_threshold_deg = rospy.get_param("~angle_threshold_deg", 1.0)
        self.stop_distance = rospy.get_param("~stop_distance", 0.3)

        self.cmd_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=1)

        # 구독
        rospy.Subscriber('/aruco/pose_x', Float64, self.pose_x_callback)
        rospy.Subscriber('/aruco/pose_z', Float64, self.pose_z_callback)

        rospy.loginfo("[marker_pose_controller] Ready.")
        rospy.spin()

    def pose_x_callback(self, msg):
        self.pose_x = msg.data
        self.update_curve()
        self.publish_cmd()

    def pose_z_callback(self, msg):
        self.pose_z = msg.data
        self.update_curve()
        self.publish_cmd()

    def update_curve(self):
	    if self.pose_z < 2.0 and not self.reached and not self.curve_generated:
		self.curve_generated = True  # ❗바로 True로 설정

		p0 = np.array([0.0, 0.0])
		p1 = np.array([0.0, 0.5])
		p3 = np.array([self.pose_x, self.pose_z - self.stop_distance])
		p2 = p3 - np.array([0.0, 0.5])
		self.curve_points = self.compute_bezier_curve(p0, p1, p2, p3)

		rospy.loginfo("[Bezier] 경로 생성 완료")
            

    def compute_bezier_curve(self, p0, p1, p2, p3, steps=50):
        points = []
        for t in np.linspace(0, 1, steps):
            pt = (1 - t)**3 * p0 + 3*(1 - t)**2*t * p1 + 3*(1 - t)*t**2 * p2 + t**3 * p3
            points.append(pt)
        return points

    def find_nearest_point_index(self, current):
        dists = [np.linalg.norm(np.array(pt) - current) for pt in self.curve_points]
        return int(np.argmin(dists))

    def publish_cmd(self):
        if not self.curve_points or self.reached:
            return

        twist = Twist()

        current = np.array([0.0, 0.0])
        nearest_idx = self.find_nearest_point_index(current)

        if nearest_idx >= len(self.curve_points) - 1:
           self.reached = True
           self.curve_generated = False  # ❗도달 후 다음 목표 위해 재생성 허용
           rospy.loginfo("[Bezier] 목표 지점 도달 → 정지")
           self.send_serial('x')
           self.cmd_pub.publish(Twist())  # 정지 메시지도 발행
           return

        next_pt = self.curve_points[nearest_idx + 1]
        dx = next_pt[0] - current[0]
        dz = next_pt[1] - current[1]
        heading = math.degrees(math.atan2(dx, dz))
        distance = np.linalg.norm(next_pt - current)

        rospy.loginfo(f"[Bezier] Heading: {heading:.2f}°, Distance: {distance:.2f} m")

        if self.pose_z < self.stop_distance:
            if abs(heading) < self.angle_threshold_deg:
                rospy.loginfo("정렬 완료 및 근접 → 정지")
                self.send_serial('x')
            elif heading > 0:
                twist.angular.z = -2.0
                rospy.loginfo("가까움 → 우회전")
                self.send_serial('h')
            else:
                twist.angular.z = 2.0
                rospy.loginfo("가까움 → 좌회전")
                self.send_serial('f')
            twist.linear.x = 0.0
            self.cmd_pub.publish(twist)
            return

        if abs(heading) < self.angle_threshold_deg:
            twist.linear.x = 0.1
            twist.angular.z = 0.0
            rospy.loginfo("→ 직진")
            self.send_serial('t')
        elif heading > 0:
            twist.linear.x = 0.0
            twist.angular.z = -3.0
            rospy.loginfo("→ 우회전")
            self.send_serial('h')
        else:
            twist.linear.x = 0.0
            twist.angular.z = 3.0
            rospy.loginfo("→ 좌회전")
            self.send_serial('f')

        self.cmd_pub.publish(twist)

    def send_serial(self, command):
        if self.last_command != command:
            self.ser.write(command.encode())
            self.last_command = command

if __name__ == '__main__':
    try:
        MarkerPoseController()
    except rospy.ROSInterruptException:
        pass

