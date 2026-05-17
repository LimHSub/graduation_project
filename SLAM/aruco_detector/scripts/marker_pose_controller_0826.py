#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy
from std_msgs.msg import Float64, String
import serial
import time

def wrap_deg(a):
    return ((a + 180.0) % 360.0) - 180.0

def ang_err_to_target(yaw_deg, target_deg=-90.0):
    # yaw_b_m(보통 -180~0) → target까지 최단각(부호 포함)
    return wrap_deg(wrap_deg(target_deg) - wrap_deg(yaw_deg))

class MarkerPoseControllerFSM:
    M_FAR   = "FAR"     # 접근: |y| 정렬 우선, 정렬되면 전환 판정
    M_MID   = "MID"     # 임계 내부: 전진 금지, yaw 정렬
    M_FINAL = "FINAL"   # 이미 Y+YAW 모두 정렬 → 정지 유지
    M_BACK  = "BACK"    # (선택) 후진: 필요 시 사용
    M_STOP  = "STOP"    # (확장용)

    def __init__(self):
        rospy.init_node('marker_pose_controller_fsm')

        # --- Serial ---
        port = rospy.get_param("~port", "/dev/ttyACM0")
        baud = rospy.get_param("~baud", 115200)
        self.ser = serial.Serial(port, baud, timeout=1)

        # --- Params ---
        self.far_to_mid   = rospy.get_param("~far_to_mid",   0.40)  # FAR → 전환 후보 x 임계
        self.mid_to_far   = rospy.get_param("~mid_to_far",   0.42)  # MID/FINAL → FAR 복귀 x 임계
        self.y_tol_far    = rospy.get_param("~y_tol_far",    0.03)  # FAR에서의 Y 정렬 기준(|y|)

        # YAW 정렬 기준(중앙이 -90°)
        self.yaw_target   = rospy.get_param("~yaw_target",  -90.0)
        self.yaw_tol_mid  = rospy.get_param("~yaw_tol_mid",  3.0)   # FINAL 판정/ MID 완료 기준(±deg)

        # (권장) yaw 사용 거리 가드: 멀리서는 yaw 무시
        self.yaw_enable_x_m = rospy.get_param("~yaw_enable_x_m", 0.50)

        # BACK 파라미터
        self.backup_target      = rospy.get_param("~backup_target", 1.00)     # 후진 목표 x(m)
        self.back_max_ms        = rospy.get_param("~back_max_ms",   3000)     # 관측 끊겼을 때 최대 후진 시간
        self.back_entry_hold_ms = rospy.get_param("~back_entry_hold_ms", 400) # BACK 진입 추가 정지 시간

        # 유실 판정(ms)
        self.yaw_lost_timeout_ms = rospy.get_param("~yaw_lost_timeout_ms", 400)
        self.obs_lost_timeout_ms = rospy.get_param("~obs_lost_timeout_ms", 400)

        # MID 가드(ms): MID 진입 직후 잠깐은 LOST 판정 금지
        self.mid_guard_ms = rospy.get_param("~mid_guard_ms", 300)

        # 명령 문자
        self.cmd_right = rospy.get_param("~cmd_right", "h")   # 우회전
        self.cmd_left  = rospy.get_param("~cmd_left",  "f")   # 좌회전
        self.cmd_fwd   = rospy.get_param("~cmd_fwd",   "t")   # 전진
        self.cmd_back  = rospy.get_param("~cmd_back",  "g")   # 후진(옵션)
        self.cmd_stop  = rospy.get_param("~cmd_stop",  "x")   # 정지

        # keepalive
        self.keepalive_ms = rospy.get_param("~keepalive_ms", 200)
        self.desired_cmd, self.last_sent_cmd, self.last_send_ms = None, None, 0.0
        rospy.Timer(rospy.Duration(self.keepalive_ms/1000.0), self._tick)

        # State
        self.x, self.y, self.yaw = 999.0, 0.0, None
        self.last_yaw_ms = 0.0
        self.last_obs_ms = 0.0
        self.mode = self.M_FAR
        self.back_start_ms = None
        self.mid_entry_ms = None

        self.mode_pub = rospy.Publisher("~mode", String, queue_size=1)

        # Subs
        rospy.Subscriber('/aruco/pose_z',  Float64, self._cb_x)        # Xb: 전방(+)
        rospy.Subscriber('/aruco/pose_x',  Float64, self._cb_y)        # Yb: 좌/우(+좌/-우)
        rospy.Subscriber('/aruco/yaw_b_m', Float64, self._cb_yaw)      # base 기준 마커 yaw(deg)

        rospy.loginfo("[FSM] FAR: |y| 정렬되면 → (x≤%.2fm일 때 yaw 확인) → FINAL or MID. "
                      "FINAL은 정지 유지, 깨지면 MID. MID/FINAL에서 x>%.2fm이면 FAR 복귀."
                      % (self.far_to_mid, self.mid_to_far))
        rospy.spin()

    # ---------- Callbacks ----------
    def _mark_obs(self):
        self.last_obs_ms = time.time()*1000.0

    def _cb_x(self, m):
        self.x = m.data
        self._mark_obs()
        self._step()

    def _cb_y(self, m):
        self.y = m.data
        self._mark_obs()
        self._step()

    def _cb_yaw(self, m):
        self.yaw = m.data
        self.last_yaw_ms = time.time()*1000.0
        self._mark_obs()
        self._step()

    # ---------- FSM ----------
    def _step(self):
        x, y, yaw = self.x, self.y, self.yaw
        yaw_str = f"{yaw:.1f}" if yaw is not None else "None"
        rospy.loginfo(f"[STATE] mode={self.mode} | x={x:.3f} m, y={y:.3f} m, yaw_b_m={yaw_str}°")

        # --- 모드 전이 규칙 ---
        if self.mode in (self.M_MID, self.M_FINAL):
            # MID/FINAL에서 멀어지면 FAR 복귀
            if x > self.mid_to_far:
                self._transition(self.M_FAR, "MID/FINAL → FAR (x beyond mid_to_far)")
                return

        if self.mode == self.M_FAR:
            # FAR에서는 먼저 |y|를 맞춘다. 정렬된 '그 순간'에만 전환 판정.
            if abs(y) <= self.y_tol_far and x <= self.far_to_mid:
                # 전환 타이밍: yaw 상태 미리 확인해 FINAL or MID 결정
                yaw_ok = False
                # 멀리서는 yaw를 신뢰하지 않음 → x<=yaw_enable_x_m일 때만 체크
                if (yaw is not None) and (x <= self.yaw_enable_x_m):
                    e = ang_err_to_target(yaw, self.yaw_target)
                    yaw_ok = abs(e) <= self.yaw_tol_mid
                if yaw_ok:
                    self._transition(self.M_FINAL, "FAR→FINAL (Y aligned & yaw already OK)")
                else:
                    self._transition(self.M_MID, "FAR→MID (Y aligned; yaw align next)")
                return

        # --- 각 모드 제어 ---
        if self.mode == self.M_FAR:
            # |y|가 크면 회전으로 Y 먼저 맞추기
            if abs(y) > self.y_tol_far:
                self._set_cmd(self.cmd_left if y > 0 else self.cmd_right,
                              f"FAR: align Y (|y|={abs(y):.3f} > {self.y_tol_far:.3f})")
            else:
                # Y가 맞았어도 x가 아직 멀면 전진으로 접근
                if x > self.far_to_mid:
                    self._set_cmd(self.cmd_fwd, "FAR: Y OK → FORWARD to boundary")
                else:
                    # x도 충분히 가까우면 위 전이 로직에서 처리되므로 여기선 대기
                    self._hard_stop("FAR: boundary reached, waiting transition")

        elif self.mode == self.M_FINAL:
            # FINAL: 정지 유지, 조건 깨지면 MID로
            if self._final_broken():
                self._transition(self.M_MID, "FINAL broken (alignment lost) → MID")
            else:
                self._hard_stop("FINAL: hold (aligned)")

        elif self.mode == self.M_MID:
            # MID: 전진 금지, yaw 정렬
            if self._lost_in_mid():
                self._transition(self.M_BACK, "MID: LOST → BACK", pause_ms=1000)
                return
            e = ang_err_to_target(yaw, self.yaw_target)
            if abs(e) <= self.yaw_tol_mid:
                self._hard_stop(f"MID: yaw OK (|err|={abs(e):.1f}°≤{self.yaw_tol_mid:.1f}°) → HOLD")
            else:
                # ✅ 주석과 일치하도록 수정: e>0 → 좌회전
                self._set_cmd(self.cmd_left if e < 0 else self.cmd_right,
                              f"MID: align yaw_b_m (err={e:.1f}°, tol=±{self.yaw_tol_mid:.1f}°)")

        elif self.mode == self.M_BACK:
            now_ms = time.time()*1000.0

            # BACK 진입 직후 추가 정지 시간 유지 (눈에 보이는 완전 정지)
            if self.back_start_ms is not None and (now_ms - self.back_start_ms) < self.back_entry_hold_ms:
                self._hard_stop("BACK: entry hold STOP")
                return

            obs_stale = (now_ms - self.last_obs_ms) > self.obs_lost_timeout_ms

            if not obs_stale:
                # 관측이 살아있으면 목표 거리까지 후진
                if x >= self.backup_target:
                    self._transition(self.M_FAR, f"BACK: reached {self.backup_target:.2f} m → FAR")
                else:
                    self._set_cmd(self.cmd_back, f"BACK: reversing to {self.backup_target:.2f} m (x={x:.2f})")
            else:
                # 관측이 끊겼으면 back_max_ms 한도 내에서 후진
                if self.back_start_ms is None:
                    self.back_start_ms = now_ms
                if (now_ms - self.back_start_ms) < self.back_max_ms:
                    self._set_cmd(self.cmd_back, f"BACK: reversing (no obs, <{self.back_max_ms} ms)")
                else:
                    self._transition(self.M_FAR, "BACK: no obs timeout → FAR")

    # FINAL 유지 조건: |y| ≤ y_tol_far AND |yaw_err| ≤ yaw_tol_mid
    def _final_broken(self):
        if self.yaw is None:
            return True
        e = ang_err_to_target(self.yaw, self.yaw_target)
        return (abs(self.y) > self.y_tol_far) or (abs(e) > self.yaw_tol_mid)

    # ---------- 전이(정지 후 전환) ----------
    def _transition(self, next_mode, reason, pause_ms=1000):
        self._hard_stop(f"{reason} | pause {pause_ms}ms before switch")
        rospy.sleep(pause_ms / 1000.0)
        prev = self.mode
        self.mode = next_mode
        self.mode_pub.publish(self.mode)
        rospy.loginfo(f"[MODE] {prev} → {self.mode}")

        if self.mode == self.M_BACK:
            # BACK 진입 즉시 후진하지 않고, back_entry_hold_ms 동안 STOP 유지
            self.back_start_ms = time.time()*1000.0
            # (중요) 여기서 self._set_cmd(self.cmd_back, ...) 호출하지 않음!
        elif self.mode == self.M_MID:
            # MID 진입 시각 기록(가드 타임 계산용)
            self.mid_entry_ms = time.time()*1000.0

    # ---------- 유실/송신 ----------
    def _tick(self, _evt):
        # FINAL은 계속 STOP keepalive
        if self.mode == self.M_FINAL:
            self._hard_stop("FINAL: keepalive STOP")
            return
        # MID에서 유실되면 즉시 STOP 유지
        if self.mode == self.M_MID and self._lost_in_mid():
            self._hard_stop("MID: LOST (watchdog) → STOP")
            return
        self._send_if_due(False, "keepalive")

    def _lost_in_mid(self):
        now_ms = time.time()*1000.0  # ms 계산
        # MID 진입 후 mid_guard_ms 동안은 LOST 판정 금지
        if self.mode == self.M_MID and self.mid_entry_ms is not None:
            if (now_ms - self.mid_entry_ms) < self.mid_guard_ms:
                return False

        yaw_stale = (self.yaw is None) or ((now_ms - self.last_yaw_ms) > self.yaw_lost_timeout_ms)
        obs_stale = (now_ms - self.last_obs_ms) > self.obs_lost_timeout_ms
        return yaw_stale or obs_stale

    def _set_cmd(self, c, reason):
        self.desired_cmd = c
        self._send_if_due(True, reason)

    def _hard_stop(self, reason):
        try:
            self.ser.write(self.cmd_stop.encode())
            self.ser.write(self.cmd_stop.encode())
        except serial.SerialException as e:
            rospy.logerr(f"[SERIAL] hard_stop write failed: {e}")
        now = time.time()*1000.0
        self.desired_cmd = self.cmd_stop
        self.last_sent_cmd = self.cmd_stop
        self.last_send_ms = now
        rospy.loginfo(f"[CMD] {self.cmd_stop}  ({reason})")

    def _send_if_due(self, force, reason):
        now = time.time()*1000.0
        due = force or (self.desired_cmd != self.last_sent_cmd) or (now - self.last_send_ms >= self.keepalive_ms)
        if not self.desired_cmd or not due:
            return
        try:
            self.ser.write(self.desired_cmd.encode())
            self.last_sent_cmd = self.desired_cmd
            self.last_send_ms = now
            rospy.loginfo(f"[CMD] {self.desired_cmd}  ({reason})")
        except serial.SerialException as e:
            rospy.logerr(f"[SERIAL] write failed: {e}")

if __name__ == '__main__':
    try:
        MarkerPoseControllerFSM()
    except rospy.ROSInterruptException:
        pass

