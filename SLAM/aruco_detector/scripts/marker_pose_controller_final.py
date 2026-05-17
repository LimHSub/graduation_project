#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
marker_pose_controller.py (STANDALONE-FSM ENABLED)
- camera_mission 값 없이도 마커가 보이면 waypoint3(FSM 도킹)만 수행
- standalone_fsm=True일 때: 마커 감지 → FSM 자동 ARM → 수행 → 완료 시 대기
- 기존 미션 기반 동작도 유지(standalone_fsm=False면 기존 로직 그대로)
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
    # FSM states
    S_SAMPLE       = "SAMPLE"
    S_APPROACH     = "APPROACH_OFFSET"
    S_ALIGN_YAW    = "ALIGN_YAW"
    S_CHECK_Y      = "CHECK_Y"
    S_BACK_OUT     = "BACK_OUT"
    S_FINAL        = "FINAL_APPROACH"
    S_DONE         = "DONE"

    # High-level missions
    M_NONE         = 0
    M_VERIFY_ONLY  = 1  # waypoint 1,2
    M_DOCK_FSM     = 2  # waypoint 3

    def __init__(self):
        rospy.init_node("aruco_wall_align_fsm")

        # ===== Driving (Twist) =====
        self.doc_cmd_topic   = rospy.get_param("~cmd_doc_topic", "/cmd_vel_doc")
        self.doc_lin_speed   = rospy.get_param("~doc_lin_speed", 0.15)   # m/s
        self.doc_ang_speed   = rospy.get_param("~doc_ang_speed", 0.8)    # rad/s
        self.keepalive_ms    = rospy.get_param("~keepalive_ms", 200)
        self.cmd_pub         = rospy.Publisher(self.doc_cmd_topic, Twist, queue_size=10)

        # ===== Original parameters (kept) =====
        self.yaw_target_d     = rospy.get_param("~yaw_target_d", -90.0)
        self.yaw_tol_d        = rospy.get_param("~yaw_tol_d", 1.0)
        self.check_y_tol_m    = rospy.get_param("~check_y_tol_m", 0.05)
        self.final_y_tol_m    = rospy.get_param("~final_y_tol_m", 0.02)
        self.stop_x_m         = rospy.get_param("~stop_x_m", 0.20)
        self.final_y_phase_switch_m = rospy.get_param("~final_y_phase_switch_m", 0.30)
        self.final_y_phase_tol_m    = rospy.get_param("~final_y_phase_tol_m", 0.01)
        self.final_yaw_phase_tol_d  = rospy.get_param("~final_yaw_phase_tol_d", 1.0)
        self.approach_gate_m  = rospy.get_param("~approach_gate_m", 0.50)
        self.backup_target_m  = rospy.get_param("~backup_target_m", 0.70)
        self.back_entry_hold_ms = rospy.get_param("~back_entry_hold_ms", 2000)
        self.y0_samples           = rospy.get_param("~y0_samples", 9)
        self.y0_sample_timeout_ms = rospy.get_param("~y0_sample_timeout_ms", 600)
        self.y0_clip_max          = rospy.get_param("~y0_clip_max", 0.30)
        self.y_tol_m              = rospy.get_param("~y_tol_m", 0.02)
        self.forward_gate_in_ratio  = rospy.get_param("~forward_gate_in_ratio", 0.8)
        self.forward_gate_out_ratio = rospy.get_param("~forward_gate_out_ratio", 1.0)
        if self.forward_gate_in_ratio > self.forward_gate_out_ratio:
            self.forward_gate_in_ratio = self.forward_gate_out_ratio

        # Direction interpretation
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
        self.side_sign = -1

        self.back_start_ms = None
        self.desired_twist = Twist()
        self.last_sent_twist = Twist()
        self.last_send_ms  = 0.0

        # ===== Mission control =====
        self.mission_mode = self.M_NONE
        self.last_mission = None            # 마지막 수신 미션 번호(1/2/3)
        self.expected_marker_id = None
        self.current_marker_id  = None
        self.verify_ok_need = rospy.get_param("~verify_ok_need", 5)
        self.verify_ok_cnt  = 0
        self.verify_timeout_s = rospy.get_param("~verify_timeout_s", 30.0)

        # ===== STANDALONE FSM 모드 =====
        # 카메라 미션 없이 마커만 보이면 FSM(3번)만 수행
        self.standalone_fsm       = rospy.get_param("~standalone_fsm", True)         # ★ STANDALONE
        self.seen_need            = rospy.get_param("~standalone_seen_need", 3)      # ★ 연속 감지 필요 횟수
        self.lost_timeout_ms      = rospy.get_param("~obs_lost_timeout_ms", 600)     # ★ 관측 타임아웃
        self.yaw_lost_timeout_ms  = rospy.get_param("~yaw_lost_timeout_ms", 600)     # ★ yaw 타임아웃
        self._seen_cnt            = 0                                                # ★ 감지 히스테리시스
        self._standalone_active   = False                                            # ★ 현재 FSM이 standalone으로 동작 중인지

        # ===== ROS I/O =====
        self.mode_pub   = rospy.Publisher("~mode", String, queue_size=1)
        self.done_pub   = rospy.Publisher("/docking_done", Bool, queue_size=1)
        self.mdone_pub  = rospy.Publisher("/camera_mission_done", Int32, queue_size=1, latch=True)

        # ---- Topic names ----
        topic_pose_z     = rospy.get_param("~topic_pose_z",   "/aruco/pose_z")      # x-forward(거리)
        topic_pose_x     = rospy.get_param("~topic_pose_x",   "/aruco/pose_x")      # y-lateral
        topic_yaw_b_m    = rospy.get_param("~topic_yaw_b_m",  "/aruco/yaw_b_m")     # yaw(deg)
        topic_marker_id  = rospy.get_param("~topic_marker_id","/aruco/marker_id")
        topic_mission    = rospy.get_param("~topic_mission",  "/camera_mission")

        # ---- Subscribers ----
        rospy.Subscriber(topic_pose_z,     Float64, self._cb_x)
        rospy.Subscriber(topic_pose_x,     Float64, self._cb_y)
        rospy.Subscriber(topic_yaw_b_m,    Float64, self._cb_yaw)
        rospy.Subscriber(topic_marker_id,  Int32,   self._cb_marker_id)
        rospy.Subscriber(topic_mission,    Int32,   self._cb_mission)

        # ---- Keepalive timer ----
        rospy.Timer(rospy.Duration(self.keepalive_ms/1000.0), self._tick)

        rospy.loginfo("[FSM] SAMPLE→APPROACH(%.2fm)→ALIGN_YAW→CHECK_Y→FINAL(0.30/0.20m)→DONE | "
                      "mission 1,2=VERIFY_ONLY / mission 3=DOCK_FSM | standalone=%s",
                      self.approach_gate_m, str(self.standalone_fsm))
        rospy.spin()

    # ---------- mission / marker id ----------
    def _cb_mission(self, m: Int32):
        # standalone 모드면 미션 무시하고 마커 트리거로만 동작
        if self.standalone_fsm:
            rospy.loginfo_throttle(5.0, "[FSM] standalone_fsm=True → /camera_mission 무시")
            return

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
            self.mode_pub.publish("FSM_ACTIVE")

    def _cb_marker_id(self, m: Int32):
        self.current_marker_id = int(m.data)

        # ★ STANDALONE: 마커 감지 히스테리시스 관리
        if self.standalone_fsm:
            if self.current_marker_id is not None and self.current_marker_id >= 0:
                self._seen_cnt = min(self.seen_need, self._seen_cnt + 1)
            else:
                self._seen_cnt = 0

    # ---------- common helpers ----------
    def _mark_obs(self):
        self.last_obs_ms = time.time()*1000.0

    def _twist(self, lin=0.0, ang=0.0):
        t = Twist()
        t.linear.x  = lin
        t.angular.z = ang
        return t

    def _y_cmd_twist(self, err_positive: bool):
        left = (err_positive and self.y_left_if_err_pos) or ((not err_positive) and (not self.y_left_if_err_pos))
        return self._twist(0.0, +self.doc_ang_speed if left else -self.doc_ang_speed)

    def _yaw_cmd_twist(self, err_positive: bool):
        left = (err_positive and self.yaw_left_if_err_pos) or ((not err_positive) and (not self.yaw_left_if_err_pos))
        return self._twist(0.0, +self.doc_ang_speed if left else -self.doc_ang_speed)

    def _hard_stop(self, reason="STOP"):
        self.desired_twist = self._twist(0.0, 0.0)
        self._send_if_due(True, reason)

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

    def _goto(self, ns, reason="", pause_ms=0):
        if pause_ms > 0:
            self._hard_stop(f"{self.state}→{ns}: pause {pause_ms}ms | {reason}")
            rospy.sleep(pause_ms/1000.0)
        prev = self.state
        self.state = ns
        self.mode_pub.publish(self.state)
        rospy.loginfo("[MODE] %s → %s | %s", prev, self.state, reason)
        if ns == self.S_BACK_OUT:
            self.back_start_ms = time.time()*1000.0
        if ns == self.S_SAMPLE:
            self._reset_sampling()

    def _go_idle_and_reset(self, reason=""):
        self._hard_stop("IDLE: " + reason)
        self.state = self.S_SAMPLE
        self._reset_sampling()

    def _finish_and_wait(self, reason=""):
        self._hard_stop("DONE: " + reason)

        # ★ STANDALONE: 완료 시 다음 마커 대기(미션 완료 퍼블리시는 생략)
        if self.standalone_fsm:
            self.mode_pub.publish("WAIT_MARKER")
            self.mission_mode = self.M_NONE
            self._standalone_active = False
            self.expected_marker_id = None
            self.verify_ok_cnt = 0
            return

        # 기존 미션 동작(standalone=False)
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
        if self.state == self.S_SAMPLE and self.mission_mode == self.M_DOCK_FSM:
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
        now = time.time()*1000.0
        yaw_stale = (self.yaw is None) or ((now - self.last_yaw_ms) > self.yaw_lost_timeout_ms)
        obs_stale = ((now - self.last_obs_ms) > self.lost_timeout_ms)

        # ★ STANDALONE 모드: 마커 감지로 자동 ARM/Disarm
        if self.standalone_fsm:
            marker_detected = (self.current_marker_id is not None) and (self.current_marker_id >= 0) and (not obs_stale)
            if self.mission_mode == self.M_NONE:
                # 아직 미션 없고, 마커 연속 감지되면 FSM ARM
                if marker_detected and self._seen_cnt >= self.seen_need:
                    self.mission_mode = self.M_DOCK_FSM
                    self._standalone_active = True
                    self._go_idle_and_reset("STANDALONE: marker seen → ARM FSM")
                    self.mode_pub.publish("FSM_ACTIVE")
                else:
                    self._hard_stop("STANDALONE: waiting marker")
                    return
            else:
                # FSM 진행 중에 관측 끊기면 즉시 정지 대기
                if not marker_detected:
                    self._finish_and_wait("STANDALONE: marker lost → idle")
                    return

        # 미션 없으면 정지(standalone이 아니거나, 위 로직에서 return 안 됐을 때)
        if self.mission_mode == self.M_NONE:
            self._hard_stop("NO_MISSION")
            return

        # VERIFY_ONLY는 standalone 모드에서는 사용하지 않음(혹시 남아있으면 정지)
        if (not self.standalone_fsm) and (self.mission_mode == self.M_VERIFY_ONLY):
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

        # ====== 이하 FSM (3번) ======
        if obs_stale:
            self._hard_stop("OBS_STALE")
            return

        # SAMPLE
        if self.state == self.S_SAMPLE:
            got_n = len(self.y_samples) >= self.y0_samples
            timedout = (self.sample_start_ms is not None) and ((now - self.sample_start_ms) >= self.y0_sample_timeout_ms)
            if got_n or timedout:
                y0_med = median(self.y_samples) if self.y_samples else self.y
                self.y0 = y0_med if y0_med is not None else 0.0
                raw = self.side_sign * self.y0
                self.y_target = max(-self.y0_clip_max, min(self.y0_clip_max, raw))
                rospy.loginfo("[SAMPLE] n=%d, y0_med=%.3f → y_target=%.3f (clip ±%.0fcm)",
                              len(self.y_samples), self.y0, self.y_target, self.y0_clip_max*100.0)
                self._goto(self.S_APPROACH, "init to approach")
            else:
                self._hard_stop("SAMPLE: collecting y0 median")
            return

        # APPROACH
        if self.state == self.S_APPROACH:
            y_err = self.y - self.y_target
            if abs(y_err) > self.y_tol_m:
                self._set_cmd(self._y_cmd_twist(y_err > 0.0),
                              f"APPROACH: align y to {self.y_target:+.3f} (err={y_err:+.3f})")
                return
            gate_in = self.forward_gate_in_ratio * self.y_tol_m
            if abs(y_err) <= gate_in:
                if self.x > self.approach_gate_m:
                    self._set_cmd(self._twist(self.doc_lin_speed, 0.0),
                                  f"APPROACH: forward to x<={self.approach_gate_m:.2f}")
                else:
                    self._hard_stop("APPROACH: at gate")
                    self._goto(self.S_ALIGN_YAW, "enter yaw alignment")
            else:
                self._set_cmd(self._twist(0.0, 0.0), "APPROACH: hys hold")
            return

        # ALIGN_YAW
        if self.state == self.S_ALIGN_YAW:
            if yaw_stale:
                self._goto(self.S_BACK_OUT, "yaw stale → backout", pause_ms=self.back_entry_hold_ms)
                return
            e = ang_err_to_target(self.yaw, self.yaw_target_d)
            if abs(e) <= self.yaw_tol_d:
                self._hard_stop(f"ALIGN_YAW: ok (err={e:+.1f}°)")
                self._goto(self.S_CHECK_Y, "check lateral after yaw")
            else:
                self._set_cmd(self._yaw_cmd_twist(e < 0.0),
                              f"ALIGN_YAW: err={e:+.1f}° → turn")
            return

        # CHECK_Y
        if self.state == self.S_CHECK_Y:
            if abs(self.y) <= self.check_y_tol_m:
                self._goto(self.S_FINAL, "Y within tol → final approach", pause_ms=300)
            else:
                self._goto(self.S_BACK_OUT, "Y out of tol → backout", pause_ms=self.back_entry_hold_ms)
            return

        # BACK_OUT
        if self.state == self.S_BACK_OUT:
            now_ms = time.time()*1000.0
            if (now_ms - (self.back_start_ms or now_ms)) < self.back_entry_hold_ms:
                self._hard_stop("BACK_OUT: entry hold")
                return
            if self.x >= self.backup_target_m:
                self.side_sign *= -1
                self._hard_stop("BACK_OUT: reached target distance")
                self._goto(self.S_SAMPLE, "flip side & resample", pause_ms=self.back_entry_hold_ms)
            else:
                self._set_cmd(self._twist(-self.doc_lin_speed, 0.0),
                              f"BACK_OUT: reversing to {self.backup_target_m:.2f} (x={self.x:.2f})")
            return

        # FINAL
        if self.state == self.S_FINAL:
            # Phase-Y
            if self.x > self.final_y_phase_switch_m:
                y_err = self.y
                if abs(y_err) > self.final_y_phase_tol_m:
                    self._set_cmd(self._y_cmd_twist(y_err > 0.0),
                                  f"FINAL[Y-phase]: trim y={y_err:+.3f}m to ≤{self.final_y_phase_tol_m:.3f}m")
                    return
                else:
                    self._set_cmd(self._twist(self.doc_lin_speed, 0.0),
                                  f"FINAL[Y-phase]: forward to {self.final_y_phase_switch_m:.2f}m")
                    return

            # Phase-Yaw
            if self.x > self.stop_x_m:
                e = None if yaw_stale else ang_err_to_target(self.yaw, self.yaw_target_d)
                if (e is None) or (abs(e) > self.final_yaw_phase_tol_d):
                    if e is None:
                        self._hard_stop("FINAL[Yaw-phase]: yaw stale")
                    else:
                        self._set_cmd(self._yaw_cmd_twist(e < 0.0),
                                      f"FINAL[Yaw-phase]: align yaw err={e:+.1f}° to ≤{self.final_yaw_phase_tol_d:.1f}°")
                    return
                else:
                    self._set_cmd(self._twist(self.doc_lin_speed, 0.0),
                                  f"FINAL[Yaw-phase]: forward to {self.stop_x_m:.2f}m")
                    return

            # 0.20m 도달 → 최종 확인
            e = None if yaw_stale else ang_err_to_target(self.yaw, self.yaw_target_d)
            ok_yaw = (e is not None) and (abs(e) <= self.final_yaw_phase_tol_d)
            ok_y   = abs(self.y) <= self.final_y_phase_tol_m
            if ok_yaw and ok_y:
                self._finish_and_wait("final_ok")
            else:
                self._goto(self.S_BACK_OUT, "FINAL @20cm not within specs", pause_ms=self.back_entry_hold_ms)
            return

        if self.state == self.S_DONE:
            self._hard_stop("DONE: hold")
            return

    # ---------- timers ----------
    def _tick(self, _evt):
        # VERIFY_ONLY/NO_MISSION/샘플링/완료 시에도 정지 keepalive
        if self.mission_mode in (self.M_NONE, self.M_VERIFY_ONLY) or self.state in (self.S_DONE, self.S_SAMPLE):
            self._hard_stop(f"{'NO_MISSION' if self.mission_mode==self.M_NONE else self.state}: keepalive STOP")
            return
        self._send_if_due(False, "keepalive")

if __name__ == "__main__":
    try:
        MarkerAlignFSM()
    except rospy.ROSInterruptException:
        pass

