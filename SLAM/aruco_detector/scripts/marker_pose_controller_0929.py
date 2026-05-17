#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
marker_pose_controller.py (UPDATED: robust stop at final 20cm, yaw tol = ±3°, no delays)
- waypoint(=camera_mission) 1,2: VERIFY_ONLY (주행X, 마커ID 일치 확인만)
- waypoint 3: FSM 시퀀스
    (1) SAMPLE: pose_y N개 샘플링→중앙값 y0 계산
        → y_target = clamp(-y0, ±y0_clip_max)  # '반대 부호' 적용
    (2) APPROACH_TO_50: |y - y_target| ≤ 0.03m 되도록 회전 보정하며 x→0.50m 접근
    (3) ALIGN_YAW_90: yaw_b_m을 -90°(±3°)로 정렬
        └ if |y| ≤ 0.06m → GO_TO_20 (최종 접근)
        └ else BACK_TO_70로 이동
    (4) BACK_TO_70: x→0.70m 될 때까지 후진 후 SAMPLE로 복귀
    (5) GO_TO_20: 다른 조건 없이 x→0.20m까지 직진, 도달하면
        └ Twist(0,0) 버스트 전송 + (옵션) 'x' 문자 전송 → 정지 유지 및 종료
"""

import rospy
import time
from std_msgs.msg import Float64, String, Int32, Bool
from geometry_msgs.msg import Twist

def wrap_deg(a):
    return ((a + 180.0) % 360.0) - 180.0

def ang_err_to_target(yaw_deg, target_deg=-90.0):
    return wrap_deg(target_deg - wrap_deg(yaw_deg))

def median(lst):
    if not lst:
        return None
    s = sorted(lst)
    n = len(s)
    mid = n // 2
    return s[mid] if (n % 2 == 1) else 0.5 * (s[mid - 1] + s[mid])

class MarkerAlignFSM:
    # States
    S_SAMPLE       = "SAMPLE"
    S_DONE         = "DONE"
    S_APPROACH50   = "APPROACH_YTARGET_TO_X50"
    S_ALIGN_YAW90  = "ALIGN_YAW_TO_-90"
    S_BACK_TO_70   = "BACK_TO_70_AND_RESAMPLE"
    S_GO_TO_20     = "GO_STRAIGHT_TO_X20"

    # Missions
    M_NONE         = 0
    M_VERIFY_ONLY  = 1
    M_DOCK_FSM     = 2

    def __init__(self):
        rospy.init_node("aruco_wall_align_fsm")

        # ===== Driving (Twist) =====
        self.doc_cmd_topic   = rospy.get_param("~cmd_doc_topic", "/cmd_vel_doc")
        self.doc_lin_speed   = rospy.get_param("~doc_lin_speed", 0.15)   # m/s
        self.doc_ang_speed   = rospy.get_param("~doc_ang_speed", 0.8)    # rad/s
        self.keepalive_ms    = rospy.get_param("~keepalive_ms", 200)
        self.cmd_pub         = rospy.Publisher(self.doc_cmd_topic, Twist, queue_size=10)

        # ===== (Optional) Char-based stop =====
        self.send_stop_char  = rospy.get_param("~send_stop_char", True)
        self.char_cmd_topic  = rospy.get_param("~char_cmd_topic", "/doc_cmd_char")
        self.char_pub        = rospy.Publisher(self.char_cmd_topic, String, queue_size=10) if self.send_stop_char else None

        # Robust stop: zero-twist burst duration
        self.stop_burst_ms   = rospy.get_param("~stop_burst_ms", 400)  # 0.4s 동안 0 트위스트 재전송
        self.stop_burst_until_ms = 0.0

        # ===== Params =====
        self.yaw_target_d     = rospy.get_param("~yaw_target_d", -90.0)
        self.yaw_tol_d        = rospy.get_param("~yaw_tol_d", 3.0)      # ±3°
        self.y_tol_m          = rospy.get_param("~y_tol_m", 0.03)       # 3 cm (접근 단계용)
        self.exit_y_tol_m     = rospy.get_param("~exit_y_tol_m", 0.06)  # 6 cm (yaw 정렬 후 탈출)
        self.x_in_m           = rospy.get_param("~x_in_m", 0.50)        # 0.50 m
        self.x_back_m         = rospy.get_param("~x_back_m", 0.70)      # 0.70 m
        self.x_final_m        = rospy.get_param("~x_final_m", 0.20)     # 최종 0.20 m

        # 샘플링/클램프
        self.y0_samples           = rospy.get_param("~y0_samples", 9)
        self.y0_sample_timeout_ms = rospy.get_param("~y0_sample_timeout_ms", 600)
        self.y0_clip_max          = rospy.get_param("~y0_clip_max", 0.30)

        # 타임아웃
        self.yaw_lost_timeout_ms  = rospy.get_param("~yaw_lost_timeout_ms", 600)
        self.obs_lost_timeout_ms  = rospy.get_param("~obs_lost_timeout_ms", 600)

        # 방향 해석
        self.yaw_left_if_err_pos = rospy.get_param("~yaw_left_if_err_pos", True)
        self.y_left_if_err_pos   = rospy.get_param("~y_left_if_err_pos", True)

        # ===== State vars =====
        self.x, self.y, self.yaw = 999.0, 0.0, None
        self.last_obs_ms, self.last_yaw_ms = 0.0, 0.0

        self.state = self.S_SAMPLE
        self.y_samples = []
        self.sample_start_ms = None
        self.y0 = 0.0
        self.y_target = 0.0

        self.desired_twist = Twist()
        self.last_sent_twist = Twist()
        self.last_send_ms  = 0.0

        # ===== Mission control =====
        self.mission_mode = self.M_NONE
        self.last_mission = None
        self.expected_marker_id = None
        self.current_marker_id  = None
        self.verify_ok_need = rospy.get_param("~verify_ok_need", 5)
        self.verify_ok_cnt  = 0
        self.verify_timeout_s = rospy.get_param("~verify_timeout_s", 30.0)

        # ===== ROS I/O =====
        self.mode_pub   = rospy.Publisher("~mode", String, queue_size=1)
        self.done_pub   = rospy.Publisher("/docking_done", Bool, queue_size=1)
        self.mdone_pub  = rospy.Publisher("/camera_mission_done", Int32, queue_size=1, latch=True)

        topic_pose_z     = rospy.get_param("~topic_pose_z",   "/aruco/pose_z")
        topic_pose_x     = rospy.get_param("~topic_pose_x",   "/aruco/pose_x")
        topic_yaw_b_m    = rospy.get_param("~topic_yaw_b_m",  "/aruco/yaw_b_m")
        topic_marker_id  = rospy.get_param("~topic_marker_id","/aruco/marker_id")
        topic_mission    = rospy.get_param("~topic_mission",  "/camera_mission")

        rospy.Subscriber(topic_pose_z,     Float64, self._cb_x)
        rospy.Subscriber(topic_pose_x,     Float64, self._cb_y)
        rospy.Subscriber(topic_yaw_b_m,    Float64, self._cb_yaw)
        rospy.Subscriber(topic_marker_id,  Int32,   self._cb_marker_id)
        rospy.Subscriber(topic_mission,    Int32,   self._cb_mission)

        rospy.Timer(rospy.Duration(self.keepalive_ms/1000.0), self._tick)

        rospy.loginfo("[FSM] mission 1,2: VERIFY_ONLY | mission 3: SAMPLE → APPROACH50 → ALIGN_YAW90 → (GO_TO_20 or BACK_TO_70→SAMPLE)")
        rospy.spin()

    # ---------- mission / marker id ----------
    def _cb_mission(self, m: Int32):
        val = int(m.data)
        self.last_mission = val
        self.expected_marker_id = val
        self.mission_mode = self.M_VERIFY_ONLY if val in (1, 2) else (self.M_DOCK_FSM if val == 3 else self.M_VERIFY_ONLY)
        self.verify_ok_cnt = 0
        self._go_idle_and_reset(reason=f"NEW MISSION={val} → mode={'VERIFY_ONLY' if self.mission_mode==self.M_VERIFY_ONLY else 'DOCK_FSM'}")

        if self.mission_mode == self.M_VERIFY_ONLY:
            self.verify_start_time = time.time()
            self.mode_pub.publish("VERIFY_ONLY")
        else:
            self.mode_pub.publish("FSM_ACTIVE")  # 시작은 SAMPLE

    def _cb_marker_id(self, m: Int32):
        self.current_marker_id = int(m.data)

    # ---------- helpers ----------
    def _mark_obs(self):
        self.last_obs_ms = time.time()*1000.0

    def _twist(self, lin=0.0, ang=0.0):
        t = Twist()
        t.linear.x  = lin
        t.angular.z = ang
        return t

    def _publish_stop_char(self):
        if self.send_stop_char and self.char_pub is not None:
            try:
                self.char_pub.publish(String(data='x'))
                rospy.loginfo("[DOC_CHAR] 'x' stop sent")
            except Exception as e:
                rospy.logwarn("Failed to publish stop char: %s", e)

    def _start_stop_burst(self):
        self.stop_burst_until_ms = time.time()*1000.0 + self.stop_burst_ms

    def _y_cmd_twist(self, err_positive: bool):
        left = (err_positive and self.y_left_if_err_pos) or ((not err_positive) and (not self.y_left_if_err_pos))
        return self._twist(0.0, +self.doc_ang_speed if left else -self.doc_ang_speed)

    def _yaw_cmd_twist(self, err_positive: bool):
        left = (err_positive and self.yaw_left_if_err_pos) or ((not err_positive) and (not self.yaw_left_if_err_pos))
        return self._twist(0.0, +self.doc_ang_speed if left else -self.doc_ang_speed)

    def _hard_stop(self, reason="STOP"):
        # 0 트위스트 즉시 송신
        self.desired_twist = self._twist(0.0, 0.0)
        self._send_if_due(True, reason)
        # 문자형 정지도 함께(옵션)
        self._publish_stop_char()
        # 짧은 버스트로 0 명령 유지
        self._start_stop_burst()

    def _set_cmd(self, twist: Twist, reason=""):
        self.desired_twist = twist
        self._send_if_due(True, reason)

    def _send_if_due(self, force, reason):
        now = time.time()*1000.0
        due = force or (self._twist_changed(self.desired_twist, self.last_sent_twist)) or (now - self.last_send_ms >= self.keepalive_ms)
        if not due:
            return
        self.cmd_pub.publish(self.desired_twist)
        self.last_sent_twist = self.desired_twist
        self.last_send_ms = now
        rospy.loginfo("[DOC_CMD] lin=%.3f ang=%.3f (%s)", self.desired_twist.linear.x, self.desired_twist.angular.z, reason)

    @staticmethod
    def _twist_changed(a: Twist, b: Twist, eps=1e-6):
        return (abs(a.linear.x - b.linear.x) > eps) or (abs(a.angular.z - b.angular.z) > eps)

    def _reset_sampling(self):
        self.y_samples = []
        self.sample_start_ms = None

    def _goto(self, ns, reason=""):
        prev = self.state
        self._hard_stop(f"{prev}→{ns} | {reason}")
        self.state = ns
        self.mode_pub.publish(self.state)
        rospy.loginfo("[MODE] %s → %s | %s", prev, self.state, reason)

    def _go_idle_and_reset(self, reason=""):
        self._hard_stop("IDLE: " + reason)
        self.state = self.S_SAMPLE
        self._reset_sampling()

    def _finish_and_wait(self, reason=""):
        # 정지 상태 유지 + 완료 신호
        self._hard_stop("DONE: " + reason)
        self.done_pub.publish(Bool(data=True))
        if self.last_mission is not None:
            self.mdone_pub.publish(Int32(data=self.last_mission))
        self.mission_mode = self.M_NONE
        self.expected_marker_id = None
        self.verify_ok_cnt = 0
        self.mode_pub.publish("WAIT_NEXT_WAYPOINT")

    # ---------- callbacks ----------
    def _cb_x(self, m: Float64):
        self.x = float(m.data)
        self._mark_obs()
        self._step()

    def _cb_y(self, m: Float64):
        self.y = float(m.data)
        self._mark_obs()
        if (self.state == self.S_SAMPLE) and (self.mission_mode == self.M_DOCK_FSM):
            if self.sample_start_ms is None:
                self.sample_start_ms = time.time()*1000.0
            self.y_samples.append(self.y)
        self._step()

    def _cb_yaw(self, m: Float64):
        self.yaw = float(m.data)
        self.last_yaw_ms = time.time()*1000.0
        self._mark_obs()
        self._step()

    # ---------- main step ----------
    def _step(self):
        if self.mission_mode == self.M_NONE:
            self._hard_stop("NO_MISSION")
            return

        # VERIFY_ONLY
        if self.mission_mode == self.M_VERIFY_ONLY:
            self._hard_stop("VERIFY_ONLY: stop")
            if (self.expected_marker_id is not None) and (self.current_marker_id is not None):
                if self.current_marker_id == self.expected_marker_id:
                    self.verify_ok_cnt = min(self.verify_ok_need, self.verify_ok_cnt + 1)
                else:
                    self.verify_ok_cnt = 0
            if hasattr(self, "verify_start_time") and (time.time() - self.verify_start_time > self.verify_timeout_s):
                rospy.logwarn("VERIFY_ONLY timeout %.1fs: waiting ID=%s (seen=%s)",
                              self.verify_timeout_s, self.expected_marker_id, self.current_marker_id)
            if self.verify_ok_cnt >= self.verify_ok_need:
                rospy.loginfo("VERIFY_ONLY OK: expected ID=%s", self.expected_marker_id)
                self._finish_and_wait("verify_ok")
            return

        # FSM
        now = time.time()*1000.0
        yaw_stale = (self.yaw is None) or ((now - self.last_yaw_ms) > self.yaw_lost_timeout_ms)
        obs_stale = ((now - self.last_obs_ms) > self.obs_lost_timeout_ms)
        if obs_stale:
            self._hard_stop("OBS_STALE")
            return

        # (1) SAMPLE
        if self.state == self.S_SAMPLE:
            got_n = len(self.y_samples) >= self.y0_samples
            timedout = (self.sample_start_ms is not None) and ((now - self.sample_start_ms) >= self.y0_sample_timeout_ms)
            if got_n or timedout:
                y0_med = median(self.y_samples) if self.y_samples else self.y
                self.y0 = y0_med if y0_med is not None else 0.0
                raw = -self.y0
                self.y_target = max(-self.y0_clip_max, min(self.y0_clip_max, raw))
                rospy.loginfo("[SAMPLE] n=%d, y0_med=%.3f → y_target=%.3f (clip ±%.0fcm)",
                              len(self.y_samples), self.y0, self.y_target, self.y0_clip_max*100.0)
                self._goto(self.S_APPROACH50, "init to approach(→0.50m)")
                self._reset_sampling()
            else:
                self._hard_stop("SAMPLE: collecting y0 median")
            return

        # (2) APPROACH_TO_50
        if self.state == self.S_APPROACH50:
            y_err = self.y - self.y_target
            if abs(y_err) > self.y_tol_m:
                self._set_cmd(self._y_cmd_twist(y_err > 0.0),
                              f"APPROACH50: align y to {self.y_target:+.3f} (err={y_err:+.3f})")
                return
            if self.x > self.x_in_m:
                self._set_cmd(self._twist(self.doc_lin_speed, 0.0),
                              f"APPROACH50: forward to x≤{self.x_in_m:.2f} (x={self.x:.2f})")
                return
            self._hard_stop("APPROACH50: reached 0.50m")
            self._goto(self.S_ALIGN_YAW90, "enter yaw alignment")
            return

        # (3) ALIGN_YAW_90
        if self.state == self.S_ALIGN_YAW90:
            if yaw_stale:
                self._hard_stop("ALIGN_YAW90: yaw stale")
                return

            e = ang_err_to_target(self.yaw, self.yaw_target_d)

            if abs(e) <= self.yaw_tol_d:
                self._hard_stop(f"ALIGN_YAW90: ok (err={e:+.1f}°)")
                if abs(self.y) <= self.exit_y_tol_m:
                    self._goto(self.S_GO_TO_20, "y within 6cm → final 20cm approach")
                else:
                    self._goto(self.S_BACK_TO_70, "y not within 6cm → back out to 0.70m")
            else:
                self._set_cmd(self._yaw_cmd_twist(e < 0.0),
                              f"ALIGN_YAW90: err={e:+.1f}° → turn to {self.yaw_target_d:.0f}° (±{self.yaw_tol_d:.0f}°)")
            return

        # (4) BACK_TO_70
        if self.state == self.S_BACK_TO_70:
            if self.x >= self.x_back_m:
                self._hard_stop("BACK_TO_70: reached 0.70m")
                self._goto(self.S_SAMPLE, "resample y0")
            else:
                self._set_cmd(self._twist(-self.doc_lin_speed, 0.0),
                              f"BACK_TO_70: reversing to x≥{self.x_back_m:.2f} (x={self.x:.2f})")
            return

        # (5) GO_TO_20: 조건 없이 전진 → 0.20m 이내면 확실한 정지
        if self.state == self.S_GO_TO_20:
            if self.x > self.x_final_m:
                self._set_cmd(self._twist(self.doc_lin_speed, 0.0),
                              f"GO_TO_20: forward to x≤{self.x_final_m:.2f} (x={self.x:.2f})")
                return

            # 도달 → 0 트위스트 + (옵션) 'x' 문자 + 버스트 → 종료
            self._hard_stop("GO_TO_20: reached final 0.20m")
            self._finish_and_wait("final_20cm_ok")
            return

        if self.state == self.S_DONE:
            self._finish_and_wait("sequence_done")
            return

    # ---------- timers ----------
    def _tick(self, _evt):
        # stop 버스트 기간에는 0 트위스트를 반복 송신
        now = time.time()*1000.0
        if now < self.stop_burst_until_ms:
            # 이미 desired_twist는 0,0으로 세팅되어 있음
            self._send_if_due(True, "stop-burst")
            return

        if self.mission_mode in (self.M_NONE, self.M_VERIFY_ONLY) or self.state in (self.S_DONE, self.SAMPLE):
            self._hard_stop(f"{'NO_MISSION' if self.mission_mode==self.M_NONE else self.state}: keepalive STOP")
            return

        self._send_if_due(False, "keepalive")

if __name__ == "__main__":
    try:
        MarkerAlignFSM()
    except rospy.ROSInterruptException:
        pass
