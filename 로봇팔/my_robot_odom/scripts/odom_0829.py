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

from nav_msgs.msg import Odometry
from geometry_msgs.msg import Quaternion, Twist, Point, Pose, Vector3
from std_msgs.msg import Header
from sensor_msgs.msg import Imu
import tf

# ----------구동부 설정 ----------
SERIAL_PORT = '/dev/ttyACM0' # /dev/ttyACM0 or /dev/ttyUSB1
BAUDRATE = 115200
WHEEL_RADIUS_L = 0.079425
WHEEL_RADIUS_R = 0.079425
WHEEL_BASE = 0.388      # 
ENCODER_RES = 2700 #3275
MAX_REASONABLE_TICK_DIFF = 5000
MIN_DT = 0.01  # 최소 시간 간격(초)로 속도 튐 방지
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

        self.ser = serial.Serial(SERIAL_PORT, BAUDRATE, timeout=1)

        threading.Thread(target=self.read_serial, daemon=True).start()
        threading.Thread(target=self.read_keyboard, daemon=True).start()

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
                    
                    print(f"[TICK] Left: {left_tick}  Right: {right_tick}")
                    #print(f"[IMU] Yaw(rad): {yaw:.6f}  Gz(rad/s): {gz:.6f}")
                    self.process_encoder(left_tick, right_tick)
                    self.process_imu(yaw, gz)
            except Exception as e:
                rospy.logwarn(f"Serial read error: {e}")
                continue

    def read_keyboard(self):
        fd = sys.stdin.fileno()
        old = termios.tcgetattr(fd)
        tty.setcbreak(fd)
        print("t/f/g/h: 움직임 | x: 정지 | q: 종료")
        try:
            while self.running and not rospy.is_shutdown():
                if select.select([sys.stdin], [], [], 0.01)[0]:
                    key = sys.stdin.read(1)
                    if key == 'q':
                        self.running = False
                        rospy.signal_shutdown('Keyboard quit')
                        break
                    elif key in ['t', 'g', 'f', 'h', 'x']:
                        self.ser.write(key.encode())
                        print(f"[KEYBOARD] Sent: {key}")
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
            direction = 'f' if ang > 0 else 'h'
            speed_percent = min(int(abs(ang) * 100), 100)  # 0~100 사이로 제한
            cmd = f"{direction}{speed_percent:02d}"  # 예: a30, d80
        elif lin > 0:
            cmd = 't'
        elif lin < 0:
            cmd = 'g'
        else:
            cmd = 'x'

        self.ser.write(cmd.encode())
        rospy.loginfo(f"[CMD_VEL] linear={lin:.2f} angular={ang:.2f} → '{cmd}' sent")

    def process_encoder(self, left, right):
        if self.last_left_tick is None or self.last_right_tick is None:
            self.last_left_tick = left
            self.last_right_tick = right
            self.last_time = time.time()   # ★ 꼭 갱신!
            return

        d_left = left - self.last_left_tick
        d_right = right - self.last_right_tick

        if abs(d_left) > MAX_REASONABLE_TICK_DIFF or abs(d_right) > MAX_REASONABLE_TICK_DIFF:
            self.last_left_tick = left
            self.last_right_tick = right
            self.last_time = time.time()   # ★ 꼭 갱신!
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
        dt = max(now - self.last_time, MIN_DT)  # ★ 최소값 보장
        self.last_time = now
        self.last_vx = dS / dt if dt > 0 else 0.0
        self.last_vth = dTheta / dt if dt > 0 else 0.0

    def process_imu(self, yaw, gz):
        # 순수 IMU만 따로 저장, 오도메트리 계산에 절대 사용 X!
        self.latest_yaw = yaw
        self.latest_gz = gz

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
        # covariance는 tuning 가능 (여기선 x, y, theta만 신뢰, 나머지 99999)
        odom.pose.covariance = [0.1, 0, 0, 0, 0, 0,# 3275가 맞다면 0.5
                                0, 0.1, 0, 0, 0, 0,
                                0, 0, 99999, 0, 0, 0,
                                0, 0, 0, 99999, 0, 0,
                                0, 0, 0, 0, 99999, 0,
                                0, 0, 0, 0, 0, 0.1]
        odom.twist.covariance = odom.pose.covariance
        self.odom_pub.publish(odom)
      #  self.odom_broadcaster.sendTransform(
      #      (self.x, self.y, 0.),
      #      odom_quat,
      #      current_time,
      #      "base_link",
      #      "odom"
      #  )

    def publish_imu(self):
        if hasattr(self, "latest_yaw") and hasattr(self, "latest_gz"):
            imu_msg = Imu()
            imu_msg.header.stamp = rospy.Time.now()
            imu_msg.header.frame_id = "imu_link"
            # yaw만 있음 (roll, pitch 0)
            q = tf.transformations.quaternion_from_euler(0, 0, self.latest_yaw)
            imu_msg.orientation = Quaternion(*q)
            imu_msg.angular_velocity = Vector3(0.0, 0.0, self.latest_gz)
            # IMU covariance(신뢰도)는 높게!
            imu_msg.orientation_covariance = [0.01, 0, 0, 0, 0.01, 0, 0, 0, 0.01]
            imu_msg.angular_velocity_covariance = [0.01, 0, 0, 0, 0.01, 0, 0, 0, 0.01]
            imu_msg.linear_acceleration_covariance = [-1, 0, 0, 0, -1, 0, 0, 0, -1]  # 사용 안함
            self.imu_pub.publish(imu_msg)

def main():
    rospy.init_node('unified_controller')
    try:
        UnifiedController()
    except rospy.ROSInterruptException:
        pass

if __name__ == '__main__':
    main()

