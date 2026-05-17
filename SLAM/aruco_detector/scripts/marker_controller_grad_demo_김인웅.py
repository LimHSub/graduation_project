#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy
from std_msgs.msg import Float64, Int32, Bool, String, Int16MultiArray
from geometry_msgs.msg import Twist
import time
import statistics


class MarkerPoseControllerGradDemo:
    """
    /camera_mission 규칙
      +N : marker id N 을 보고 접근(approach)
      -N : marker id N 을 보고 후진(retreat)

    유지 기능:
      - /cmd_vel_doc publish
      - /doc_cmd_char publish
      - /doc_pwm_cmd publish
      - /docking_done, /camera_mission_done publish
      - marker id 확인
      - timer 기반 제어

    추가/변경 기능:
      - 구간별 target_y 기준 정렬 주행
      - APPROACH 마지막에 yaw 정렬 단계 추가
      - RETREAT 마지막에 목표 거리 도달 시 종료
      - RETREAT 시 정렬 오차가 크면 후진하면서 정렬하는 PWM 사용

    좌표 해석:
      - pose_x : 좌우 오차(화면/카메라 기준 lateral 오차)
      - pose_z : 마커까지 거리
      - yaw    : 마커 yaw
    """

    MODE_IDLE = 0
    MODE_APPROACH = 1
    MODE_RETREAT = 2
    MODE_SAMPLE_APPROACH = 3

    def __init__(self):
        rospy.init_node('marker_pose_controller_grad_demo')

        # ===== Params =====
        self.cmd_doc_topic = rospy.get_param("~cmd_doc_topic", "/cmd_vel_doc")
        self.char_cmd_topic = rospy.get_param("~char_cmd_topic", "/doc_cmd_char")
        self.doc_pwm_topic = rospy.get_param("~doc_pwm_topic", "/doc_pwm_cmd")

        self.topic_pose_x = rospy.get_param("~topic_pose_x", "/aruco/pose_x")
        self.topic_pose_z = rospy.get_param("~topic_pose_z", "/aruco/pose_z")
        self.topic_marker_id = rospy.get_param("~topic_marker_id", "/aruco/marker_id")
        self.topic_mission = rospy.get_param("~topic_mission", "/camera_mission")
        self.topic_yaw = rospy.get_param("~topic_yaw", "/aruco/yaw_b_m")

        # ===== 정렬 허용 오차 =====
        self.angle_threshold = float(rospy.get_param("~angle_threshold", 0.15))
        self.yaw_threshold = float(rospy.get_param("~yaw_threshold", 1.0))

        # ===== 목표 yaw =====
        # -90도 = -1.57 rad
        self.target_yaw = float(rospy.get_param("~target_yaw", -90.0))

        # ===== 요청 조건 반영 거리 기준 =====
        # 직진 도킹
        self.approach_stage1_dist = float(rospy.get_param("~approach_stage1_dist", 1.85))
        self.approach_stop_dist = float(rospy.get_param("~approach_stop_dist", 0.85))

        # 후진 도킹
        self.retreat_stage1_limit = float(rospy.get_param("~retreat_stage1_limit", 1.20))
        self.retreat_finish_dist = float(rospy.get_param("~retreat_finish_dist", 2.50))

        # ===== 각 구간 목표 Y =====
        self.approach_target_y_far = float(rospy.get_param("~approach_target_y_far", -0.3))
        self.approach_target_y_near = float(rospy.get_param("~approach_target_y_near", 0.2))

        # ===== APPROACH 자동 기준 잡기(FSM SAMPLE 방식) =====
        # /camera_mission +N 수신 후, 현재 pose_x를 여러 번 샘플링해서
        # 그 중앙값을 전진 정렬 기준값으로 사용한다.
        self.use_auto_approach_target_y = bool(rospy.get_param("~use_auto_approach_target_y", True))
        self.approach_sample_count = int(rospy.get_param("~approach_sample_count", 9))
        self.approach_sample_timeout = float(rospy.get_param("~approach_sample_timeout", 0.8))
        self.approach_target_y_offset = float(rospy.get_param("~approach_target_y_offset", 0.0))
        self.approach_auto_target_y = None
        self.approach_y_samples = []
        self.approach_sample_start_time = None

        self.retreat_target_y_close = float(rospy.get_param("~retreat_target_y_close", 0.3))
        self.retreat_target_y_far = float(rospy.get_param("~retreat_target_y_far", -0.15))

        # ===== 기존 Twist 속도 =====
        self.forward_speed = float(rospy.get_param("~forward_speed", 0.10))
        self.backward_speed = float(rospy.get_param("~backward_speed", 0.08))
        self.turn_speed = float(rospy.get_param("~turn_speed", 0.8))

        # yaw 정렬용 회전속도
        self.yaw_turn_speed = float(rospy.get_param("~yaw_turn_speed", 0.5))

        self.marker_timeout_sec = float(rospy.get_param("~marker_timeout_sec", 0.7))
        self.stop_burst_count = int(rospy.get_param("~stop_burst_count", 3))

        # err_positive 일 때 좌회전이면 True
        self.left_if_pos = rospy.get_param("~left_if_pos", True)

        self.control_hz = float(rospy.get_param("~control_hz", 10.0))

        # ===== 고정 PWM 파라미터 =====
        self.use_direct_pwm = bool(rospy.get_param("~use_direct_pwm", True))

        # 전진 직진
        self.pwm_forward_l = int(rospy.get_param("~pwm_forward_l", 30))
        self.pwm_forward_r = int(rospy.get_param("~pwm_forward_r", 30))

        # 전진/일반 회전
        self.pwm_turn_left_l = int(rospy.get_param("~pwm_turn_left_l", 0))
        self.pwm_turn_left_r = int(rospy.get_param("~pwm_turn_left_r", 13))

        self.pwm_turn_right_l = int(rospy.get_param("~pwm_turn_right_l", 13))
        self.pwm_turn_right_r = int(rospy.get_param("~pwm_turn_right_r", 0))

        # 후진 직진
        self.pwm_backward_l = int(rospy.get_param("~pwm_backward_l", -30))
        self.pwm_backward_r = int(rospy.get_param("~pwm_backward_r", -30))

        # ===== 후진하면서 정렬하는 전용 PWM 추가 =====
        # 좌/우 모두 음수이고 크기 차이로 방향을 만듦
        self.pwm_retreat_turn_left_l = int(rospy.get_param("~pwm_retreat_turn_left_l", -10))
        self.pwm_retreat_turn_left_r = int(rospy.get_param("~pwm_retreat_turn_left_r", 0))

        self.pwm_retreat_turn_right_l = int(rospy.get_param("~pwm_retreat_turn_right_l", 0))
        self.pwm_retreat_turn_right_r = int(rospy.get_param("~pwm_retreat_turn_right_r", -10))

        # yaw 정렬용 PWM (제자리 회전)
        self.pwm_yaw_left_l = int(rospy.get_param("~pwm_yaw_left_l", 0))
        self.pwm_yaw_left_r = int(rospy.get_param("~pwm_yaw_left_r", 8))
        self.pwm_yaw_right_l = int(rospy.get_param("~pwm_yaw_right_l", 8))
        self.pwm_yaw_right_r = int(rospy.get_param("~pwm_yaw_right_r", 0))

        # ===== ROS pub/sub =====
        self.cmd_pub = rospy.Publisher(self.cmd_doc_topic, Twist, queue_size=10)
        self.char_pub = rospy.Publisher(self.char_cmd_topic, String, queue_size=10)
        self.pwm_pub = rospy.Publisher(self.doc_pwm_topic, Int16MultiArray, queue_size=10)

        self.topic_done = rospy.get_param("~topic_done", "/marker_mission_done")
        self.docking_done_topic = rospy.get_param("~docking_done_topic", "/marker_docking_done")

        self.done_pub = rospy.Publisher(self.docking_done_topic, Bool, queue_size=1)
        self.mdone_pub = rospy.Publisher(self.topic_done, Int32, queue_size=1, latch=True)
        self.mode_pub = rospy.Publisher("~mode", String, queue_size=1, latch=True)

        rospy.Subscriber(self.topic_pose_x, Float64, self.pose_x_callback, queue_size=1)
        rospy.Subscriber(self.topic_pose_z, Float64, self.pose_z_callback, queue_size=1)
        rospy.Subscriber(self.topic_marker_id, Int32, self.marker_id_callback, queue_size=1)
        rospy.Subscriber(self.topic_mission, Int32, self.mission_callback, queue_size=1)
        rospy.Subscriber(self.topic_yaw, Float64, self.yaw_callback, queue_size=1)

        # ===== State =====
        self.current_pose_x = 0.0
        self.current_distance = float('inf')
        self.current_yaw = 0.0
        self.current_marker_id = None

        self.last_pose_time = 0.0
        self.last_marker_time = 0.0
        self.last_yaw_time = 0.0

        self.mode = self.MODE_IDLE
        self.expected_marker_id = None
        self.last_completed_marker_id = None

        rospy.Timer(rospy.Duration(1.0 / self.control_hz), self._timer_cb)

        rospy.loginfo(
            "[marker_pose_controller_grad_demo] ready "
            "(control_hz=%.1f, angle_threshold=%.3f, yaw_threshold=%.3f, "
            "target_yaw=%.3f, direct_pwm=%s)",
            self.control_hz,
            self.angle_threshold,
            self.yaw_threshold,
            self.target_yaw,
            str(self.use_direct_pwm)
        )
        rospy.spin()

    # =========================================================
    # Callbacks
    # =========================================================
    def pose_x_callback(self, msg):
        self.current_pose_x = float(msg.data)
        self.last_pose_time = time.time()

    def pose_z_callback(self, msg):
        self.current_distance = float(msg.data)
        self.last_pose_time = time.time()

    def yaw_callback(self, msg):
        self.current_yaw = float(msg.data)
        self.last_yaw_time = time.time()

    def marker_id_callback(self, msg):
        self.current_marker_id = int(msg.data)
        self.last_marker_time = time.time()

    def mission_callback(self, msg):
        val = int(msg.data)

        if val == 0:
            self.mode = self.MODE_IDLE
            self.expected_marker_id = None
            self._reset_approach_sampling()
            self.publish_stop("mission=0 -> idle")
            self.mode_pub.publish("IDLE")
            return

        if val > 0:
            self.expected_marker_id = val
            self._reset_approach_sampling()
            if self.use_auto_approach_target_y:
                self.mode = self.MODE_SAMPLE_APPROACH
                self.mode_pub.publish("SAMPLE_APPROACH")
                rospy.loginfo("[marker_ctrl] SAMPLE_APPROACH start for marker id=%d", self.expected_marker_id)
            else:
                self.mode = self.MODE_APPROACH
                self.mode_pub.publish("APPROACH")
                rospy.loginfo("[marker_ctrl] APPROACH start for marker id=%d", self.expected_marker_id)
        else:
            self.mode = self.MODE_RETREAT
            self.expected_marker_id = abs(val)
            self._reset_approach_sampling()
            self.mode_pub.publish("RETREAT")
            rospy.loginfo("[marker_ctrl] RETREAT start for marker id=%d", self.expected_marker_id)

        self.publish_stop("new mission start")

    def _timer_cb(self, _event):
        self.run_control()

    # =========================================================
    # Utils
    # =========================================================
    def _reset_approach_sampling(self):
        self.approach_auto_target_y = None
        self.approach_y_samples = []
        self.approach_sample_start_time = None

    def _sample_approach_target_y(self, pose_x):
        if self.approach_sample_start_time is None:
            self.approach_sample_start_time = time.time()

        self.approach_y_samples.append(float(pose_x))
        elapsed = time.time() - self.approach_sample_start_time

        enough = len(self.approach_y_samples) >= max(1, self.approach_sample_count)
        timeout = elapsed >= self.approach_sample_timeout

        if not (enough or timeout):
            self.publish_stop(
                "SAMPLE_APPROACH collecting pose_x %d/%d" %
                (len(self.approach_y_samples), self.approach_sample_count)
            )
            return False

        med = statistics.median(self.approach_y_samples)
        self.approach_auto_target_y = float(med) + self.approach_target_y_offset
        rospy.loginfo(
            "[marker_ctrl] SAMPLE_APPROACH done: n=%d median_pose_x=%.3f offset=%.3f -> target_y=%.3f",
            len(self.approach_y_samples), med, self.approach_target_y_offset, self.approach_auto_target_y
        )
        self.mode = self.MODE_APPROACH
        self.mode_pub.publish("APPROACH")
        self.publish_stop("SAMPLE_APPROACH done -> APPROACH")
        return True

    def marker_visible_recently(self):
        now = time.time()
        return ((now - self.last_pose_time) < self.marker_timeout_sec) and \
               ((now - self.last_marker_time) < self.marker_timeout_sec)

    def yaw_visible_recently(self):
        now = time.time()
        return (now - self.last_yaw_time) < self.marker_timeout_sec

    def expected_marker_visible(self):
        if self.expected_marker_id is None:
            return False
        if self.current_marker_id is None:
            return False
        return self.current_marker_id == self.expected_marker_id

    def publish_twist(self, lin, ang, reason=""):
        twist = Twist()
        twist.linear.x = lin
        twist.angular.z = ang
        self.cmd_pub.publish(twist)
        rospy.loginfo("[marker_ctrl] cmd_vel lin=%.3f ang=%.3f | %s", lin, ang, reason)

    def publish_pwm(self, left, right, reason=""):
        msg = Int16MultiArray()
        msg.data = [int(left), int(right)]
        self.pwm_pub.publish(msg)
        rospy.loginfo("[marker_ctrl] pwm [%d, %d] | %s", int(left), int(right), reason)

    def publish_stop(self, reason="stop"):
        twist = Twist()
        stop_pwm = Int16MultiArray()
        stop_pwm.data = [0, 0]

        for _ in range(self.stop_burst_count):
            self.cmd_pub.publish(twist)
            self.pwm_pub.publish(stop_pwm)

        self.char_pub.publish(String(data='x'))
        rospy.loginfo("[marker_ctrl] STOP | %s", reason)

    def finish_mission(self, reason="done"):
        marker_id = self.expected_marker_id if self.expected_marker_id is not None else -1

        self.publish_stop(reason)
        self.done_pub.publish(Bool(data=True))
        if marker_id > 0:
            self.mdone_pub.publish(Int32(data=marker_id))

        self.last_completed_marker_id = marker_id
        rospy.loginfo("[marker_ctrl] mission finished marker=%s | %s", str(marker_id), reason)

        self.mode = self.MODE_IDLE
        self.expected_marker_id = None
        self._reset_approach_sampling()
        self.mode_pub.publish("IDLE")

    def _turn_ang(self, err_positive: bool, turn_speed=None) -> float:
        ts = self.turn_speed if turn_speed is None else float(turn_speed)
        left = (err_positive and self.left_if_pos) or \
               ((not err_positive) and (not self.left_if_pos))
        return +ts if left else -ts

    def _turn_pwm(self, err_positive: bool):
        left = (err_positive and self.left_if_pos) or \
               ((not err_positive) and (not self.left_if_pos))
        if left:
            return self.pwm_turn_left_l, self.pwm_turn_left_r
        else:
            return self.pwm_turn_right_l, self.pwm_turn_right_r

    def _retreat_turn_pwm(self, err_positive: bool):
        left = (err_positive and self.left_if_pos) or \
               ((not err_positive) and (not self.left_if_pos))
        if left:
            return self.pwm_retreat_turn_left_l, self.pwm_retreat_turn_left_r
        else:
            return self.pwm_retreat_turn_right_l, self.pwm_retreat_turn_right_r

    def _yaw_turn_pwm(self, err_positive: bool):
        left = (err_positive and self.left_if_pos) or \
               ((not err_positive) and (not self.left_if_pos))
        if left:
            return self.pwm_yaw_left_l, self.pwm_yaw_left_r
        else:
            return self.pwm_yaw_right_l, self.pwm_yaw_right_r

    def drive_with_target_y(self, dist, pose_x, target_y, forward_motion=True, phase=""):
        err = pose_x - target_y

        if abs(err) <= self.angle_threshold:
            if forward_motion:
                if self.use_direct_pwm:
                    self.publish_pwm(
                        self.pwm_forward_l, self.pwm_forward_r,
                        "%s forward dist=%.3f pose_x=%.3f target_y=%.3f err=%.3f" %
                        (phase, dist, pose_x, target_y, err)
                    )
                else:
                    self.publish_twist(
                        self.forward_speed, 0.0,
                        "%s forward dist=%.3f pose_x=%.3f target_y=%.3f err=%.3f" %
                        (phase, dist, pose_x, target_y, err)
                    )
            else:
                if self.use_direct_pwm:
                    self.publish_pwm(
                        self.pwm_backward_l, self.pwm_backward_r,
                        "%s backward dist=%.3f pose_x=%.3f target_y=%.3f err=%.3f" %
                        (phase, dist, pose_x, target_y, err)
                    )
                else:
                    self.publish_twist(
                        -self.backward_speed, 0.0,
                        "%s backward dist=%.3f pose_x=%.3f target_y=%.3f err=%.3f" %
                        (phase, dist, pose_x, target_y, err)
                    )
        else:
            if self.use_direct_pwm:
                if forward_motion:
                    l, r = self._turn_pwm(err > 0.0)
                    self.publish_pwm(
                        l, r,
                        "%s turn dist=%.3f pose_x=%.3f target_y=%.3f err=%.3f" %
                        (phase, dist, pose_x, target_y, err)
                    )
                else:
                    l, r = self._retreat_turn_pwm(err > 0.0)
                    self.publish_pwm(
                        l, r,
                        "%s retreat_turn dist=%.3f pose_x=%.3f target_y=%.3f err=%.3f" %
                        (phase, dist, pose_x, target_y, err)
                    )
            else:
                if forward_motion:
                    ang = self._turn_ang(err > 0.0)
                    self.publish_twist(
                        0.0, ang,
                        "%s turn dist=%.3f pose_x=%.3f target_y=%.3f err=%.3f ang=%.2f" %
                        (phase, dist, pose_x, target_y, err, ang)
                    )
                else:
                    # twist 사용 시에는 backward + angular.z 로 후진하면서 정렬
                    ang = self._turn_ang(err > 0.0)
                    self.publish_twist(
                        -self.backward_speed, ang,
                        "%s backward_turn dist=%.3f pose_x=%.3f target_y=%.3f err=%.3f ang=%.2f" %
                        (phase, dist, pose_x, target_y, err, ang)
                    )

    def align_yaw_only(self, yaw, phase="YAW_ALIGN"):
        err = yaw - self.target_yaw

        if abs(err) <= self.yaw_threshold:
            self.finish_mission(
                "%s complete yaw=%.3f target=%.3f err=%.3f" %
                (phase, yaw, self.target_yaw, err)
            )
            return

        if self.use_direct_pwm:
            l, r = self._yaw_turn_pwm(err > 0.0)
            self.publish_pwm(
                l, r,
                "%s turning yaw=%.3f target=%.3f err=%.3f threshold=%.3f" %
                (phase, yaw, self.target_yaw, err, self.yaw_threshold)
            )
        else:
            ang = self._turn_ang(err > 0.0, self.yaw_turn_speed)
            self.publish_twist(
                0.0, ang,
                "%s turning yaw=%.3f target=%.3f err=%.3f threshold=%.3f ang=%.2f" %
                (phase, yaw, self.target_yaw, err, self.yaw_threshold, ang)
            )

    # =========================================================
    # Main Control
    # =========================================================
    def run_control(self):
        if self.mode == self.MODE_IDLE:
            return

        if not self.marker_visible_recently():
            self.publish_stop("marker stale")
            return

        if not self.expected_marker_visible():
            self.publish_stop(
                "waiting expected marker id=%s, seen=%s" %
                (str(self.expected_marker_id), str(self.current_marker_id))
            )
            return

        pose_x = self.current_pose_x
        dist = self.current_distance
        yaw = self.current_yaw

        # -------------------------
        # SAMPLE_APPROACH
        # -------------------------
        if self.mode == self.MODE_SAMPLE_APPROACH:
            self._sample_approach_target_y(pose_x)
            return

        # -------------------------
        # APPROACH
        # -------------------------
        if self.mode == self.MODE_APPROACH:
            # 1) dist > 1.85 : Y=-0.3 기준 정렬 주행
            if dist > self.approach_stage1_dist:
                self.drive_with_target_y(
                    dist=dist,
                    pose_x=pose_x,
                    target_y=(self.approach_auto_target_y if self.approach_auto_target_y is not None else self.approach_target_y_far),
                    forward_motion=True,
                    phase="APPROACH_STAGE1"
                )
                return

            # 2) 0.60 < dist <= 1.85 : Y=0 기준 정렬 주행
            if dist > self.approach_stop_dist:
                self.drive_with_target_y(
                    dist=dist,
                    pose_x=pose_x,
                    target_y=(self.approach_auto_target_y if self.approach_auto_target_y is not None else self.approach_target_y_near),
                    forward_motion=True,
                    phase="APPROACH_STAGE2"
                )
                return

            # 3) dist <= 0.60 : 정지 후 yaw 정렬
            self.publish_stop("APPROACH reached stop point dist=%.3f -> yaw align" % dist)

            if not self.yaw_visible_recently():
                rospy.logwarn("[marker_ctrl] yaw not updated recently, cannot align yaw")
                return

            self.align_yaw_only(yaw, phase="APPROACH_YAW_ALIGN")
            return

        # -------------------------
        # RETREAT
        # -------------------------
        if self.mode == self.MODE_RETREAT:
            # 1) dist < 0.60 : Y=+0.3 기준으로 후진 정렬
            if dist < self.retreat_stage1_limit:
                self.drive_with_target_y(
                    dist=dist,
                    pose_x=pose_x,
                    target_y=self.retreat_target_y_close,
                    forward_motion=False,
                    phase="RETREAT_STAGE1"
                )
                return

            # 2) 0.60 <= dist < 2.00 : Y=0 기준으로 후진 정렬
            if dist < self.retreat_finish_dist:
                self.drive_with_target_y(
                    dist=dist,
                    pose_x=pose_x,
                    target_y=self.retreat_target_y_far,
                    forward_motion=False,
                    phase="RETREAT_STAGE2"
                )
                return

            # 3) dist >= 2.00 : 정지 후 종료
            self.finish_mission("RETREAT reached finish distance dist=%.3f" % dist)
            return


if __name__ == '__main__':
    try:
        MarkerPoseControllerGradDemo()
    except rospy.ROSInterruptException:
        pass
