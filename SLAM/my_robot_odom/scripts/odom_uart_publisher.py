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
import tf

# ----------- 설정 ----------
SERIAL_PORT = '/dev/ttyACM0'
BAUDRATE = 115200
WHEEL_RADIUS_L = 0.079425
WHEEL_RADIUS_R = 0.079425
WHEEL_BASE = 0.388
ENCODER_RES = 3275
MAX_REASONABLE_TICK_DIFF = 300
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

        self.odom_pub = rospy.Publisher('/odom', Odometry, queue_size=10)
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
                if "," in line:
                    left_tick, right_tick = map(int, line.split(','))
                    self.process_encoder(left_tick, right_tick)
            except Exception:
                continue

    def read_keyboard(self):
        fd = sys.stdin.fileno()
        old = termios.tcgetattr(fd)
        tty.setcbreak(fd)
        print("w/s/a/d: 움직임 | x: 정지 | q: 종료")
        try:
            while self.running and not rospy.is_shutdown():
                if select.select([sys.stdin], [], [], 0.01)[0]:
                    key = sys.stdin.read(1)
                    if key == 'q':
                        self.running = False
                        rospy.signal_shutdown('Keyboard quit')
                        break
                    elif key in ['w', 's', 'a', 'd', 'x']:
                        self.ser.write(key.encode())
                        print(f"[KEYBOARD] Sent: {key}")
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old)

    def cmd_vel_callback(self, msg):
        lin = msg.linear.x
        ang = msg.angular.z
        thresh_lin = 0.01
        thresh_ang = 0.2  # 회전 명령을 판단하는 임계값

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
        #print(f"[ENCODER] LEFT: {left}, RIGHT: {right} | X: {self.x:.3f}, Y: {self.y:.3f}, Theta: {self.theta:.3f}")

        if self.last_left_tick is None or self.last_right_tick is None:
            self.last_left_tick = left
            self.last_right_tick = right
            return

        d_left = left - self.last_left_tick
        d_right = right - self.last_right_tick

        if abs(d_left) > MAX_REASONABLE_TICK_DIFF or abs(d_right) > MAX_REASONABLE_TICK_DIFF:
            self.last_left_tick = left
            self.last_right_tick = right
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
        dt = now - self.last_time
        self.last_time = now
        self.last_vx = dS / dt if dt > 0 else 0.0
        self.last_vth = dTheta / dt if dt > 0 else 0.0

    def publish_loop(self):
        rate = rospy.Rate(20)
        while not rospy.is_shutdown() and self.running:
            self.update_odom()
            rate.sleep()

    def update_odom(self):
        #print(f"[ODOM_PUBLISH] X: {self.x:.3f}, Y: {self.y:.3f}, Theta: {self.theta:.3f}")

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
        self.odom_pub.publish(odom)
        self.odom_broadcaster.sendTransform(
            (self.x, self.y, 0.),
            odom_quat,
            current_time,       
            "base_link",
            "odom"
        )

def main():
    rospy.init_node('unified_controller')
    try:
        UnifiedController()
    except rospy.ROSInterruptException:
        pass

if __name__ == '__main__':
    main()



