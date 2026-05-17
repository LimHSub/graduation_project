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
from std_msgs.msg import UInt8, String, Int16MultiArray
from sensor_msgs.msg import Imu
import tf

from my_robot_odom.srv import SetMode, SetModeResponse

from evdev import InputDevice, ecodes

SERIAL_PORT = '/dev/ttyACM0'
BAUDRATE = 115200
WHEEL_RADIUS_L = 0.0965
WHEEL_RADIUS_R = 0.0965
WHEEL_BASE = 0.4706
ENCODER_RES = 30900
MAX_REASONABLE_TICK_DIFF = 10000000
MIN_DT = 0.01

MODE_MANUAL   = 0
MODE_WAYPOINT = 1
MODE_DOCKING  = 2


class UnifiedController:
    def __init__(self):
        # --- 파라미터 ---
        self.serial_port = rospy.get_param('~serial_port', SERIAL_PORT)
        self.baudrate = rospy.get_param('~baudrate', BAUDRATE)
        self.mode = rospy.get_param('~mode_initial', MODE_WAYPOINT)
        self.allow_keyboard = rospy.get_param('~allow_keyboard_override', False)

        self.thresh_lin = float(rospy.get_param('~thresh_lin', 0.01))
        self.thresh_ang = float(rospy.get_param('~thresh_ang', 0.2))

        self.turn_deadband = float(rospy.get_param('~turn_deadband', 0.2))
        self.ang1 = float(rospy.get_param('~ang1', 0.6))
        self.pwm_max = int(rospy.get_param('~pwm_max', 30))
        self.pwm_min = int(rospy.get_param('~pwm_min', 5))
        self.joy_forward_pwm = int(rospy.get_param('~joy_forward_pwm', 50))
        self.joy_backward_pwm = int(rospy.get_param('~joy_backward_pwm', 30))
        self.base_straight = int(rospy.get_param('~base_straight', 30))
        self.base_turn = int(rospy.get_param('~base_turn', 30))
        self.base_spin = int(rospy.get_param('~base_spin', 15))
        self.base_spin = max(0, min(self.base_turn, self.base_spin))

        self.delta_max_turn = int(rospy.get_param('~delta_max_turn', self.base_turn))
        self.topic_nav = rospy.get_param('~topic_cmd_nav', '/cmd_vel_nav')

        # EMA
        self.ang_ema_alpha = float(rospy.get_param('~ang_ema_alpha', 0.85))
        self.ang_ema = 0.0
        self.ang_ema_inited = False

        self.last_sent_cmd = None

        self.max_cmd_vel_x = float(rospy.get_param('~max_cmd_vel_x', 0.30))
        self.max_cmd_vel_theta = float(rospy.get_param('~max_cmd_vel_theta', 0.30))

        self.pwm_deadband = int(rospy.get_param('~pwm_deadband', self.pwm_min))
        self.pwm_slew_step = int(rospy.get_param('~pwm_slew_step', 3))

        self.cur_pwm_L = 0
        self.cur_pwm_R = 0

        # ===== 도킹(cmd_vel_doc)용 파라미터 =====
        self.topic_doc = rospy.get_param('~topic_cmd_doc', '/cmd_vel_doc')
        self.max_doc_vel_x = float(rospy.get_param('~max_doc_vel_x', 0.15))
        self.max_doc_vel_theta = float(rospy.get_param('~max_doc_vel_theta', 0.80))

        self.doc_pwm_min = int(rospy.get_param('~doc_pwm_min', 5))
        self.doc_pwm_max = int(rospy.get_param('~doc_pwm_max', 30))
        self.doc_pwm_deadband = int(rospy.get_param('~doc_pwm_deadband', self.doc_pwm_min))
        self.doc_pwm_slew_step = int(rospy.get_param('~doc_pwm_slew_step', 2))

        self.doc_thresh_lin = float(rospy.get_param('~doc_thresh_lin', 0.005))
        self.doc_thresh_ang = float(rospy.get_param('~doc_thresh_ang', 0.05))
        self.doc_turn_deadband = float(rospy.get_param('~doc_turn_deadband', 0.03))

        self.doc_ang_ema_alpha = float(rospy.get_param('~doc_ang_ema_alpha', 0.7))
        self.doc_ang_ema = 0.0
        self.doc_ang_ema_inited = False

        self.topic_doc_char = rospy.get_param('~topic_doc_char', '/doc_cmd_char')

        # ===== 추가: 도킹 고정 PWM 직접 명령 =====
        self.topic_doc_pwm = rospy.get_param('~topic_doc_pwm', '/doc_pwm_cmd')

        # --- 상태 ---
        self.x = self.y = self.theta = 0.0
        self.last_left_tick = None
        self.last_right_tick = None
        self.last_time = time.time()

        self.last_vx = 0.0
        self.last_vth = 0.0

        self.latest_yaw = 0.0
        self.latest_gz = 0.0

        self.running = True
        self._lock = threading.Lock()

        # --- ROS ---
        self.odom_pub = rospy.Publisher('/odom', Odometry, queue_size=10)
        self.imu_pub = rospy.Publisher('/imu', Imu, queue_size=10)
        self.mode_pub = rospy.Publisher('~mode', UInt8, queue_size=1, latch=True)
        self.odom_broadcaster = tf.TransformBroadcaster()

        self.sub_nav = rospy.Subscriber(self.topic_nav, Twist, self._cb_cmd_vel_nav, queue_size=20)
        self.sub_doc = rospy.Subscriber(self.topic_doc, Twist, self._cb_cmd_vel_doc, queue_size=20)
        self.sub_doc_char = rospy.Subscriber(self.topic_doc_char, String, self._cb_doc_char, queue_size=20)

        # ===== 추가: 도킹 고정 PWM 직접 구독 =====
        self.sub_doc_pwm = rospy.Subscriber(self.topic_doc_pwm, Int16MultiArray, self._cb_doc_pwm, queue_size=20)

        self.mode_srv = rospy.Service('~set_mode', SetMode, self._srv_set_mode)

        self.ser = serial.Serial(self.serial_port, self.baudrate, timeout=1)

        threading.Thread(target=self._read_serial_loop, daemon=True).start()
        threading.Thread(target=self._publish_loop, daemon=True).start()
        threading.Thread(target=self._gamepad_loop, daemon=True).start()

        if self.allow_keyboard:
            threading.Thread(target=self._keyboard_loop, daemon=True).start()

        self._publish_mode()

    # ================= MODE =================
    def _publish_mode(self):
        self.mode_pub.publish(UInt8(self.mode))
        rospy.loginfo(f"[MODE] {self.mode}")

    def _srv_set_mode(self, req):
        if req.mode not in (MODE_MANUAL, MODE_WAYPOINT, MODE_DOCKING):
            return SetModeResponse(False, "invalid mode")
        self.mode = req.mode

        self.last_sent_cmd = None
        self.ang_ema = 0.0
        self.ang_ema_inited = False
        self.doc_ang_ema = 0.0
        self.doc_ang_ema_inited = False
        self.cur_pwm_L = 0
        self.cur_pwm_R = 0

        self._publish_mode()
        return SetModeResponse(True, "mode set")

    # ================= PWM UTIL =================
    def _clamp_pwm_signed(self, v: int) -> int:
        if v == 0:
            return 0
        s = 1 if v > 0 else -1
        a = abs(v)

        if a < self.pwm_min:
            return 0
        if a > self.pwm_max:
            a = self.pwm_max
        return s * a

    def _clamp_pwm_signed_doc(self, v: int) -> int:
        if v == 0:
            return 0
        s = 1 if v > 0 else -1
        a = abs(v)

        if a < self.doc_pwm_min:
            return 0
        if a > self.doc_pwm_max:
            a = self.doc_pwm_max
        return s * a

    def _slew(self, current: int, target: int, step: int) -> int:
        if step <= 0:
            return target

        if current == 0 and abs(target) >= self.pwm_min:
            s = 1 if target > 0 else -1
            jump = self.pwm_min if step < self.pwm_min else min(abs(target), step)
            return s * jump

        if target > current + step:
            return current + step
        if target < current - step:
            return current - step
        return target

    def _slew_doc(self, current: int, target: int, step: int) -> int:
        if step <= 0:
            return target

        if current == 0 and abs(target) >= self.doc_pwm_min:
            s = 1 if target > 0 else -1
            jump = self.doc_pwm_min if step < self.doc_pwm_min else min(abs(target), step)
            return s * jump

        if target > current + step:
            return current + step
        if target < current - step:
            return current - step
        return target

    def _vel_to_pwm_mag(self, v_abs: float, v_abs_max: float) -> int:
        if v_abs_max <= 1e-6:
            return 0
        v_abs = max(0.0, min(v_abs, v_abs_max))

        if v_abs < self.thresh_lin:
            return 0

        t = v_abs / v_abs_max
        pwm = int(round(self.pwm_min + t * (self.pwm_max - self.pwm_min)))

        if pwm < self.pwm_deadband:
            pwm = 0

        return max(0, min(self.pwm_max, pwm))

    def _vel_to_pwm_mag_doc(self, v_abs: float, v_abs_max: float) -> int:
        if v_abs_max <= 1e-6:
            return 0
        v_abs = max(0.0, min(v_abs, v_abs_max))

        if v_abs < self.doc_thresh_lin:
            return 0

        t = v_abs / v_abs_max
        pwm = int(round(self.doc_pwm_min + t * (self.doc_pwm_max - self.doc_pwm_min)))

        if pwm < self.doc_pwm_deadband:
            pwm = 0

        return max(0, min(self.doc_pwm_max, pwm))

    def _mix_cmdvel_to_pwm(self, lin: float, ang: float):
        if abs(lin) < self.thresh_lin:
            lin = 0.0
        if abs(ang) < 1e-3:
            ang = 0.0

        v_l = lin - ang * (WHEEL_BASE / 2.0)
        v_r = lin + ang * (WHEEL_BASE / 2.0)

        v_abs_max = self.max_cmd_vel_x

        pwm_l_mag = self._vel_to_pwm_mag(abs(v_l), v_abs_max)
        pwm_r_mag = self._vel_to_pwm_mag(abs(v_r), v_abs_max)

        pwm_l = pwm_l_mag if v_l >= 0 else -pwm_l_mag
        pwm_r = pwm_r_mag if v_r >= 0 else -pwm_r_mag

        pwm_l = self._clamp_pwm_signed(pwm_l)
        pwm_r = self._clamp_pwm_signed(pwm_r)

        self.cur_pwm_L = self._slew(self.cur_pwm_L, pwm_l, self.pwm_slew_step)
        self.cur_pwm_R = self._slew(self.cur_pwm_R, pwm_r, self.pwm_slew_step)

        self.cur_pwm_L = self._clamp_pwm_signed(self.cur_pwm_L)
        self.cur_pwm_R = self._clamp_pwm_signed(self.cur_pwm_R)

        return self.cur_pwm_L, self.cur_pwm_R

    def _mix_cmdvel_to_pwm_doc(self, lin: float, ang: float):
        if abs(lin) < self.doc_thresh_lin:
            lin = 0.0
        if abs(ang) < 1e-3:
            ang = 0.0

        v_l = lin - ang * (WHEEL_BASE / 2.0)
        v_r = lin + ang * (WHEEL_BASE / 2.0)

        v_abs_max = self.max_doc_vel_x
        if v_abs_max <= 1e-6:
            v_abs_max = 0.15

        pwm_l_mag = self._vel_to_pwm_mag_doc(abs(v_l), v_abs_max)
        pwm_r_mag = self._vel_to_pwm_mag_doc(abs(v_r), v_abs_max)

        pwm_l = pwm_l_mag if v_l >= 0 else -pwm_l_mag
        pwm_r = pwm_r_mag if v_r >= 0 else -pwm_r_mag

        pwm_l = self._clamp_pwm_signed_doc(pwm_l)
        pwm_r = self._clamp_pwm_signed_doc(pwm_r)

        self.cur_pwm_L = self._slew_doc(self.cur_pwm_L, pwm_l, self.doc_pwm_slew_step)
        self.cur_pwm_R = self._slew_doc(self.cur_pwm_R, pwm_r, self.doc_pwm_slew_step)

        self.cur_pwm_L = self._clamp_pwm_signed_doc(self.cur_pwm_L)
        self.cur_pwm_R = self._clamp_pwm_signed_doc(self.cur_pwm_R)

        return self.cur_pwm_L, self.cur_pwm_R

    # ================= CMD VEL =================
    def _cb_cmd_vel_nav(self, msg):
        if self.mode != MODE_WAYPOINT:
            return

        lin = msg.linear.x
        ang = msg.angular.z

        # raw 0 명령이면 EMA 무시하고 즉시 정지
        if abs(lin) < self.thresh_lin and abs(ang) < self.thresh_ang:
            self.ang_ema = 0.0
            self.ang_ema_inited = False
            self.cur_pwm_L = 0
            self.cur_pwm_R = 0
            self._write_pwm(0, 0)
            return

        alpha = self.ang_ema_alpha
        if not self.ang_ema_inited:
            self.ang_ema = ang
            self.ang_ema_inited = True
        else:
            self.ang_ema = alpha * self.ang_ema + (1.0 - alpha) * ang
        ang_f = self.ang_ema

        if abs(ang_f) <= self.turn_deadband:
            ang_f = 0.0

        if abs(lin) < self.thresh_lin and abs(ang_f) < self.thresh_ang:
            self.cur_pwm_L = 0
            self.cur_pwm_R = 0
            self._write_pwm(0, 0)
            return

        L, R = self._mix_cmdvel_to_pwm(lin, ang_f)
        self._write_pwm(L, R)

    def _cb_cmd_vel_doc(self, msg):
        if self.mode != MODE_DOCKING:
            return

        lin = msg.linear.x
        ang = msg.angular.z

        # raw 0 명령이면 EMA 무시하고 즉시 정지
        if abs(lin) < self.doc_thresh_lin and abs(ang) < self.doc_thresh_ang:
            self.doc_ang_ema = 0.0
            self.doc_ang_ema_inited = False
            self.cur_pwm_L = 0
            self.cur_pwm_R = 0
            self._write_pwm(0, 0)
            return

        alpha = self.doc_ang_ema_alpha
        if not self.doc_ang_ema_inited:
            self.doc_ang_ema = ang
            self.doc_ang_ema_inited = True
        else:
            self.doc_ang_ema = alpha * self.doc_ang_ema + (1.0 - alpha) * ang
        ang_f = self.doc_ang_ema

        if abs(ang_f) <= self.doc_turn_deadband:
            ang_f = 0.0

        if abs(lin) < self.doc_thresh_lin and abs(ang_f) < self.doc_thresh_ang:
            self.cur_pwm_L = 0
            self.cur_pwm_R = 0
            self._write_pwm(0, 0)
            return

        L, R = self._mix_cmdvel_to_pwm_doc(lin, ang_f)
        self._write_pwm(L, R)

    # ===== 추가: 도킹 고정 PWM 직접 명령 =====
    def _cb_doc_pwm(self, msg):
        if self.mode != MODE_DOCKING:
            return

        if len(msg.data) < 2:
            return

        L = int(msg.data[0])
        R = int(msg.data[1])

        # 도킹 direct pwm이므로 환산 없이 바로 전송
        self.cur_pwm_L = 0 if L == 0 else L
        self.cur_pwm_R = 0 if R == 0 else R
        self._write_pwm(L, R)

    def _cb_doc_char(self, msg):
        try:
            cmd = str(msg.data).strip().lower()
        except Exception:
            return

        if cmd == 'x':
            self.cur_pwm_L = 0
            self.cur_pwm_R = 0
            self._write_pwm(0, 0)

    # ================= GAMEPAD =================
    def _gamepad_loop(self):
        while not rospy.is_shutdown() and self.running:
            try:
                dev = InputDevice("/dev/input/event28")
                rospy.loginfo(f"[GAMEPAD] Using {dev.path} ({dev.name})")

                for event in dev.read_loop():
                    if rospy.is_shutdown() or not self.running:
                        break
                    if self.mode != MODE_WAYPOINT:
                        continue

                    if event.type == ecodes.EV_ABS:
                        if event.code == ecodes.ABS_HAT0Y:
                            if event.value == -1:
                                self._write_pwm_joy(self.joy_forward_pwm, self.joy_forward_pwm)
                            elif event.value == 1:
                                self._write_pwm_joy(-self.joy_backward_pwm, -self.joy_backward_pwm)

                        elif event.code == ecodes.ABS_HAT0X:
                            if event.value == -1:
                                p = self.base_spin
                                if abs(p) < self.pwm_min:
                                    p = 0
                                self._write_pwm_joy(-p, +p)
                            elif event.value == 1:
                                p = self.base_spin
                                if abs(p) < self.pwm_min:
                                    p = 0
                                self._write_pwm_joy(+p, -p)

                    elif event.type == ecodes.EV_KEY and event.code == ecodes.BTN_SOUTH:
                        if event.value == 1:
                            self._write_pwm_joy(0, 0)

            except OSError as e:
                rospy.logwarn(f"[GAMEPAD] device lost: {e}. Reconnecting...")
                time.sleep(0.5)
                continue
            except Exception as e:
                rospy.logwarn(f"[GAMEPAD] error: {e}. Retrying...")
                time.sleep(0.5)
                continue

    # ================= SERIAL =================
    def _write_pwm(self, L, R):
        try:
            Li = int(L)
            Ri = int(R)
        except Exception:
            return

        if self.mode == MODE_DOCKING:
            Li = self._clamp_pwm_signed_doc(Li)
            Ri = self._clamp_pwm_signed_doc(Ri)
        else:
            Li = self._clamp_pwm_signed(Li)
            Ri = self._clamp_pwm_signed(Ri)

        if (Li, Ri) == self.last_sent_cmd:
            return

        try:
            out = f"{Li},{Ri}\n"
            self.ser.write(out.encode())
            self.last_sent_cmd = (Li, Ri)
            rospy.loginfo(f"[SEND] {Li},{Ri}")
        except Exception as e:
            rospy.logwarn(f"[SERIAL] write error: {e}")

    def _write_pwm_joy(self, L, R):
        try:
            Li = int(L)
            Ri = int(R)
        except Exception:
            return

        joy_limit = max(abs(self.joy_forward_pwm), abs(self.joy_backward_pwm))
        Li = max(-joy_limit, min(joy_limit, Li))
        Ri = max(-joy_limit, min(joy_limit, Ri))

        if abs(Li) < self.pwm_min:
            Li = 0
        if abs(Ri) < self.pwm_min:
            Ri = 0

        if (Li, Ri) == self.last_sent_cmd:
            return

        try:
            out = f"{Li},{Ri}\n"
            self.ser.write(out.encode())
            self.last_sent_cmd = (Li, Ri)
            rospy.loginfo(f"[SEND] {Li},{Ri}")
        except Exception as e:
            rospy.logwarn(f"[SERIAL] write error: {e}")

    def _read_serial_loop(self):
        rospy.loginfo("[SERIAL] read loop started")

        while self.running and not rospy.is_shutdown():
            try:
                line = self.ser.readline().decode(errors='ignore').strip()
                if not line:
                    continue

                parts = line.split(',')
                if len(parts) != 4:
                    continue

                left_tick = int(parts[0])
                right_tick = int(parts[1])

                #rospy.loginfo_throttle(0.2, f"[ENC TICK] L={left_tick} R={right_tick}")

                self._process_encoder(left_tick, right_tick)
                self._process_imu(math.radians(float(parts[2])), float(parts[3]))

            except Exception:
                pass

    # ================= KEYBOARD =================
    def _keyboard_loop(self):
        fd = sys.stdin.fileno()
        old = termios.tcgetattr(fd)
        tty.setcbreak(fd)
        try:
            while self.running and not rospy.is_shutdown():
                if select.select([sys.stdin], [], [], 0.01)[0]:
                    key = sys.stdin.read(1)

                    if key == 'w':
                        self._write_pwm(30, 30)
                    elif key == 'a':
                        p = self.base_spin
                        if abs(p) < self.pwm_min:
                            p = 0
                        self._write_pwm(-p, +p)
                    elif key == 'd':
                        p = self.base_spin
                        if abs(p) < self.pwm_min:
                            p = 0
                        self._write_pwm(+p, -p)
                    elif key == 's':
                        self._write_pwm(-30, -30)
                    elif key == 'x':
                        self._write_pwm(0, 0)
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old)

    # ================= ODOM / IMU =================
    def _process_encoder(self, left, right):
        if self.last_left_tick is None:
            self.last_left_tick = left
            self.last_right_tick = right
            self.last_time = time.time()
            return

        d_left_ticks = left - self.last_left_tick
        d_right_ticks = right - self.last_right_tick

        if abs(d_left_ticks) > MAX_REASONABLE_TICK_DIFF or abs(d_right_ticks) > MAX_REASONABLE_TICK_DIFF:
            self.last_left_tick = left
            self.last_right_tick = right
            self.last_time = time.time()
            return

        self.last_left_tick = left
        self.last_right_tick = right

        dL = 2 * math.pi * WHEEL_RADIUS_L * (d_left_ticks / ENCODER_RES)
        dR = 2 * math.pi * WHEEL_RADIUS_R * (d_right_ticks / ENCODER_RES)

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

    def _process_imu(self, yaw, gz):
        self.latest_yaw = yaw
        self.latest_gz = gz

    def _publish_loop(self):
        rate = rospy.Rate(20)
        while not rospy.is_shutdown() and self.running:
            self._pub_odom()
            self._pub_imu()
            rate.sleep()

    def _pub_odom(self):
        q = tf.transformations.quaternion_from_euler(0, 0, self.theta)
        odom = Odometry()
        odom.header.stamp = rospy.Time.now()
        odom.header.frame_id = "odom"
        odom.child_frame_id = "base_link"
        odom.pose.pose = Pose(Point(self.x, self.y, 0), Quaternion(*q))

        odom.twist.twist = Twist(
            Vector3(self.last_vx, 0.0, 0.0),
            Vector3(0.0, 0.0, self.last_vth)
        )

        odom.pose.covariance = [
            0.1, 0, 0, 0, 0, 0,
            0, 0.1, 0, 0, 0, 0,
            0, 0, 99999, 0, 0, 0,
            0, 0, 0, 99999, 0, 0,
            0, 0, 0, 0, 99999, 0,
            0, 0, 0, 0, 0, 0.1
        ]
        odom.twist.covariance = odom.pose.covariance

        self.odom_pub.publish(odom)

    def _pub_imu(self):
        q = tf.transformations.quaternion_from_euler(0, 0, self.latest_yaw)
        imu = Imu()
        imu.header.stamp = rospy.Time.now()
        imu.header.frame_id = "imu_link"
        imu.orientation = Quaternion(*q)
        imu.angular_velocity = Vector3(0, 0, self.latest_gz)

        imu.orientation_covariance = [0.01, 0, 0, 0, 0.01, 0, 0, 0, 0.01]
        imu.angular_velocity_covariance = [0.01, 0, 0, 0, 0.01, 0, 0, 0, 0.01]
        imu.linear_acceleration_covariance = [-1, 0, 0, 0, -1, 0, 0, 0, -1]

        self.imu_pub.publish(imu)


def main():
    rospy.init_node('unified_controller')
    UnifiedController()
    rospy.spin()


if __name__ == '__main__':
    main()
