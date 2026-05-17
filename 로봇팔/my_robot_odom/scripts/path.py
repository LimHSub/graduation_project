#!/usr/bin/env python3
import rospy
import serial
import math
import threading
import sys
import termios
import tty
import select
import time

from nav_msgs.msg import Odometry, Path
from geometry_msgs.msg import Quaternion, Twist, Point, Pose, Vector3, PoseStamped
from std_msgs.msg import Header
from sensor_msgs.msg import Imu
import tf

# ----------- 설정 ----------
SERIAL_PORT = '/dev/ttyACM0' # /dev/ttyACM0 or /dev/ttyUSB1
BAUDRATE = 115200
WHEEL_RADIUS_L = 0.079425
WHEEL_RADIUS_R = 0.079425
WHEEL_BASE = 0.388
ENCODER_RES = 6200 #3275
MAX_REASONABLE_TICK_DIFF = 300
MIN_DT = 0.01  # 최소 시간 간격(초)로 속도 튐 방지

# 경로 자동주행 파라미터
DIST_THRESH = 0.25    # waypoint 도달로 간주하는 거리 [m]
ANGLE_THRESH = 0.2    # 정면 차이 각도 허용 [rad]
# ---------------------------

class UnifiedController:
    def __init__(self):
        self.x = 0.0
        self.y = 0.0
        self.theta = 0.0
        self.last_left_tick = None
        self.last_right_tick = None
        self.last_time = time.time()
        self.last_vx = 0.0
        self.last_vth = 0.0
        self.running = True

        # ROS 퍼블리셔
        self.odom_pub = rospy.Publisher('/odom', Odometry, queue_size=10)
        self.imu_pub = rospy.Publisher('/imu', Imu, queue_size=10)
        self.odom_broadcaster = tf.TransformBroadcaster()
        self.cmd_sub = rospy.Subscriber('/cmd_vel', Twist, self.cmd_vel_callback)

        # Path 자동주행 관련
        self.path_sub = rospy.Subscriber('/waypoints', Path, self.path_callback)
        self.path = []
        self.current_idx = 0
        self.path_mode = False    # True면 자동주행
        self.path_lock = threading.Lock()

        self.ser = serial.Serial(SERIAL_PORT, BAUDRATE, timeout=1)

        threading.Thread(target=self.read_serial, daemon=True).start()
        threading.Thread(target=self.read_keyboard, daemon=True).start()
        threading.Thread(target=self.path_follower_loop, daemon=True).start()  # path auto follower

        self.publish_loop()

    def read_serial(self):
        while self.running and not rospy.is_shutdown():
            try:
                line = self.ser.readline().decode('utf-8', errors='ignore').strip()
                # STM: left_tick,right_tick,yaw,gz
                parts = line.split(',')
                if len(parts) == 4:
                    left_tick = int(parts[0])
                    right_tick = int(parts[1])
                    yaw_deg = float(parts[2])     # 라디안 단위 (★ 아래에서 안 씀!)
                    yaw = math.radians(yaw_deg)
                    gz = float(parts[3])      # 라디안/초 단위
                    
                    #print(f"[TICK] Left: {left_tick}  Right: {right_tick}")
                    self.process_encoder(left_tick, right_tick)
                    self.process_imu(yaw, gz)
            except Exception as e:
                rospy.logwarn(f"Serial read error: {e}")
                continue

    def read_keyboard(self):
        fd = sys.stdin.fileno()
        old = termios.tcgetattr(fd)
        tty.setcbreak(fd)
        print("w/s/a/d: 수동 움직임 | x: 정지 | q: 종료 | p: 경로자동모드 전환")
        try:
            while self.running and not rospy.is_shutdown():
                if select.select([sys.stdin], [], [], 0.01)[0]:
                    key = sys.stdin.read(1)
                    if key == 'q':
                        self.running = False
                        rospy.signal_shutdown('Keyboard quit')
                        break
                    elif key in ['w', 's', 'a', 'd', 'x']:
                        self.path_mode = False    # 수동 조작 시 자동모드 강제 OFF
                        self.ser.write(key.encode())
                        print(f"[KEYBOARD] Sent: {key}")
                    elif key == 'p':
                        # 자동 path 모드 토글
                        self.path_mode = not self.path_mode
                        print(f"[PATH FOLLOW MODE] {'ON' if self.path_mode else 'OFF'}")
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old)

    def cmd_vel_callback(self, msg):
        lin = msg.linear.x
        ang = msg.angular.z
        thresh_lin = 0.01
        thresh_ang = 0.2

        if abs(lin) < thresh_lin and abs(ang) < thresh_ang:
            cmd = 'x'
        elif abs(ang) >= thresh_ang:
            cmd = 'a' if ang > 0 else 'd'
        elif lin > 0:
            cmd = 'w'
        elif lin < 0:
            cmd = 's'
        else:
            cmd = 'x'

        self.ser.write(cmd.encode())
        rospy.loginfo(f"[CMD_VEL] linear={lin:.2f} angular={ang:.2f} → '{cmd}' sent")

    def process_encoder(self, left, right):
        if self.last_left_tick is None or self.last_right_tick is None:
            self.last_left_tick = left
            self.last_right_tick = right
            self.last_time = time.time()
            return

        d_left = left - self.last_left_tick
        d_right = right - self.last_right_tick

        if abs(d_left) > MAX_REASONABLE_TICK_DIFF or abs(d_right) > MAX_REASONABLE_TICK_DIFF:
            self.last_left_tick = left
            self.last_right_tick = right
            self.last_time = time.time()
            return

        self.last_left_tick = left
        self.last_right_tick = right

        dL = 2 * math.pi * WHEEL_RADIUS_L * (d_left / ENCODER_RES)
        dR = 2 * math.pi * WHEEL_RADIUS_R * (d_right / ENCODER_RES)
        dS = (dL + dR) / 2.0
        dTheta = (dR - dL) / WHEEL_BASE

        self.x += dS * math.cos(self.theta + dTheta / 2.0)
        self.y += dS * math.sin(self.theta + dTheta / 2.0)
        self.theta += dTheta

        now = time.time()
        dt = max(now - self.last_time, MIN_DT)
        self.last_time = now
        self.last_vx = dS / dt if dt > 0 else 0.0
        self.last_vth = dTheta / dt if dt > 0 else 0.0

    def process_imu(self, yaw, gz):
        self.latest_yaw = yaw
        self.latest_gz = gz

    # ------ Path/Waypoint 자동 주행 ------
    def path_callback(self, msg):
        with self.path_lock:
            self.path = msg.poses
            self.current_idx = 0
            print(f"[PATH] 새 경로 {len(self.path)}개 waypoint 수신.")

    def path_follower_loop(self):
        rate = rospy.Rate(10)
        while self.running and not rospy.is_shutdown():
            if self.path_mode:
                self.path_follow()
            rate.sleep()

    def path_follow(self):
     with self.path_lock:
        if not self.path or self.current_idx >= len(self.path):
            print(f"[DEBUG] path 길이={len(self.path)}, current_idx={self.current_idx}, path_mode={self.path_mode}")
            print("[DEBUG] path 없음 or 인덱스 초과, 아무 동작 안함")
            return
        target_pose = self.path[self.current_idx].pose.position
        dx = target_pose.x - self.x
        dy = target_pose.y - self.y
        dist = math.hypot(dx, dy)
        angle_to_goal = math.atan2(dy, dx)
        diff_angle = self._normalize_angle(angle_to_goal - self.theta)

        print(f"[DEBUG] 내 위치: ({self.x:.2f}, {self.y:.2f}, θ={self.theta:.2f}), "
              f"목표: ({target_pose.x:.2f}, {target_pose.y:.2f}), "
              f"idx={self.current_idx}/{len(self.path)}, 거리: {dist:.2f}, 각도차: {diff_angle:.2f}")

        if dist < DIST_THRESH:
            self.ser.write(b'x')  # 정지
            print(f"[PATH] {self.current_idx+1}/{len(self.path)} 도달! (거리: {dist:.2f}) → x 전송")
            time.sleep(0.5)
            self.current_idx += 1
            if self.current_idx >= len(self.path):
                print("[PATH] 모든 waypoint 도달. 자동주행 종료.")
                self.path_mode = False
        elif abs(diff_angle) > ANGLE_THRESH:
            if diff_angle > 0:
                self.ser.write(b'a')
                print(f"[AUTO] 좌회전(a) 각도차={diff_angle:.2f}")
            else:
                self.ser.write(b'd')
                print(f"[AUTO] 우회전(d) 각도차={diff_angle:.2f}")
        else:
            self.ser.write(b'w')
            print(f"[AUTO] 직진(w) 거리={dist:.2f}")
    def _normalize_angle(self, a):
        while a > math.pi:
            a -= 2 * math.pi
        while a < -math.pi:
            a += 2 * math.pi
        return a

    # ----------- 기존 오도메트리/IMU ----------
    def publish_loop(self):
        rate = rospy.Rate(20)
        while not rospy.is_shutdown() and self.running:
            self.update_odom()
            self.publish_imu()
            rate.sleep()

    def update_odom(self):
        current_time = rospy.Time.now()
        odom_quat = tf.transformations.quaternion_from_euler(0, 0, self.theta)

        odom = Odometry()
        odom.header = Header()
        odom.header.stamp = current_time
        odom.header.frame_id = "odom"
        odom.child_frame_id = "base_link"
        odom.pose.pose = Pose(Point(self.x, self.y, 0.), Quaternion(*odom_quat))
        odom.twist.twist = Twist(
            Vector3(getattr(self, "last_vx", 0.0), 0, 0),
            Vector3(0, 0, getattr(self, "last_vth", 0.0))
        )
        odom.pose.covariance = [0.1, 0, 0, 0, 0, 0,
                                0, 0.1, 0, 0, 0, 0,
                                0, 0, 99999, 0, 0, 0,
                                0, 0, 0, 99999, 0, 0,
                                0, 0, 0, 0, 99999, 0,
                                0, 0, 0, 0, 0, 0.1]
        odom.twist.covariance = odom.pose.covariance
        self.odom_pub.publish(odom)

    def publish_imu(self):
        if hasattr(self, "latest_yaw") and hasattr(self, "latest_gz"):
            imu_msg = Imu()
            imu_msg.header.stamp = rospy.Time.now()
            imu_msg.header.frame_id = "imu_link"
            q = tf.transformations.quaternion_from_euler(0, 0, self.latest_yaw)
            imu_msg.orientation = Quaternion(*q)
            imu_msg.angular_velocity = Vector3(0.0, 0.0, self.latest_gz)
            imu_msg.orientation_covariance = [0.01, 0, 0, 0, 0.01, 0, 0, 0, 0.01]
            imu_msg.angular_velocity_covariance = [0.01, 0, 0, 0, 0.01, 0, 0, 0, 0.01]
            imu_msg.linear_acceleration_covariance = [-1, 0, 0, 0, -1, 0, 0, 0, -1]
            self.imu_pub.publish(imu_msg)

def main():
    rospy.init_node('unified_controller')
    try:
        UnifiedController()
    except rospy.ROSInterruptException:
        pass

if __name__ == '__main__':
    main()

