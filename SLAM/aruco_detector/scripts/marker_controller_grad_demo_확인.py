#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import time
import logging

import cv2
import rospy

from ultralytics import YOLO
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from std_msgs.msg import Float64, Int32, Bool, String, Int16MultiArray
from geometry_msgs.msg import Twist


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


class MarkerPoseControllerGradDemo:
    """
    /camera_mission 규칙
      +N : marker id N 을 보고 접근(approach)
      -N : marker id N 을 보고 후진(retreat)

    기능:
      - /cmd_vel_doc publish
      - /doc_cmd_char publish
      - /doc_pwm_cmd publish
      - /marker_docking_done, /marker_mission_done publish
      - /marker_mission_fail publish
      - marker id 확인
      - timer 기반 제어
      - APPROACH 구간별 pose_x 목표값 기반 정렬 주행
      - APPROACH 마지막 yaw 정렬
      - RETREAT 목표 거리 도달 시 종료
      - RETREAT 시 정렬 오차가 크면 후진하면서 정렬하는 PWM 사용
      - APPROACH 시작 전 사전 후진 기능

    추가 버튼 검증 기능:
      1) APPROACH 시작 전 기존 3초 사전 후진을 그대로 수행
      2) 사전 후진이 완전히 끝난 뒤 로봇 정지
      3) 후진 완료 후 목표 마커가 보이면 버튼 상태 검사를 생략하고 APPROACH
      4) 마커가 보이지 않으면 첫 번째 YOLO 모델로 UP 버튼 위치를 한 번만 검출
      5) 이후 새 카메라 프레임마다 같은 버튼 좌표를 잘라 두 번째 모델로 ON/OFF 분류
      6) 유효한 ON/OFF 판정 5회 중 같은 상태가 3회 이상이면 상태 확정
      7) ON 확정:
           마커가 나타날 때까지 계속 대기한 뒤 APPROACH
      8) OFF 확정:
           일정 시간 동안 마커를 기다림
           - 시간 안에 마커가 보이면 APPROACH
           - 시간 안에 마커가 안 보이면 /marker_mission_fail 발행

    좌표 해석:
      - pose_x : 좌우 오차 또는 base_link 기준 lateral 값
      - pose_z : 마커까지 전방 거리
      - yaw    : 마커 yaw(deg)
    """

    MODE_IDLE = 0
    MODE_APPROACH = 1
    MODE_RETREAT = 2
    MODE_PRE_BACKWARD = 3

    # 추가 상태
    MODE_CHECK_ENTRY = 4
    MODE_CHECK_BUTTON = 5
    MODE_WAIT_MARKER_ON = 6
    MODE_WAIT_MARKER_OFF = 7

    def __init__(self):
        rospy.init_node('marker_pose_controller_grad_demo')

        # =====================================================
        # Params
        # 기존 파라미터 부분은 수정하지 않음
        # =====================================================
        self.cmd_doc_topic = rospy.get_param("~cmd_doc_topic", "/cmd_vel_doc")
        self.char_cmd_topic = rospy.get_param("~char_cmd_topic", "/doc_cmd_char")
        self.doc_pwm_topic = rospy.get_param("~doc_pwm_topic", "/doc_pwm_cmd")

        self.topic_pose_x = rospy.get_param("~topic_pose_x", "/aruco/pose_x")
        self.topic_pose_z = rospy.get_param("~topic_pose_z", "/aruco/pose_z")
        self.topic_marker_id = rospy.get_param("~topic_marker_id", "/aruco/marker_id")
        self.topic_mission = rospy.get_param("~topic_mission", "/camera_mission")
        self.topic_yaw = rospy.get_param("~topic_yaw", "/aruco/yaw_b_m")

        # 정렬 허용 오차
        self.angle_threshold = float(rospy.get_param("~angle_threshold", 0.15))
        self.yaw_threshold = float(rospy.get_param("~yaw_threshold", 1.0))

        # 목표 yaw
        self.target_yaw = float(rospy.get_param("~target_yaw", -83.0))

        # =====================================================
        # APPROACH 구간 기준
        # =====================================================
        # 기존 조건:
        #   dist > 1.85            -> pose_x = 0.0 기준
        #   0.85 < dist <= 1.85    -> pose_x = -0.3 기준
        #   0.6 < dist <= 0.85     -> pose_x = 0.2 기준
        #   dist <= 0.6            -> stop + yaw 정렬
        #
        # 수정 조건:
        #   dist > 1.85            -> pose_x = 0.0 기준
        #   1.20 < dist <= 1.85    -> pose_x = -0.3 기준
        #   0.85 < dist <= 1.20    -> pose_x = -0.1 기준
        #   0.6 < dist <= 0.85     -> pose_x = 0.2 기준
        #   dist <= 0.6            -> stop + yaw 정렬
        self.approach_stage1_dist = float(
            rospy.get_param("~approach_stage1_dist", 1.85)
        )
        self.approach_stage1_5_dist = float(
            rospy.get_param("~approach_stage1_5_dist", 1.00)
        )
        self.approach_stage2_dist = float(
            rospy.get_param("~approach_stage2_dist", 0.85)
        )
        self.approach_stop_dist = float(
            rospy.get_param("~approach_stop_dist", 0.55)
        )  # 0.45

        self.approach_target_y_far = float(
            rospy.get_param("~approach_target_y_far", 0.0)
        )
        self.approach_target_y_mid = float(
            rospy.get_param("~approach_target_y_mid", -0.3)
        )  # -0.5
        self.approach_target_y_mid2 = float(
            rospy.get_param("~approach_target_y_mid2", -0.2)
        )  # -0.2
        self.approach_target_y_near = float(
            rospy.get_param("~approach_target_y_near", 0.3)
        )  # 0.3

        # 후진 도킹
        self.retreat_stage1_limit = float(
            rospy.get_param("~retreat_stage1_limit", 0.85)
        )
        self.retreat_finish_dist = float(
            rospy.get_param("~retreat_finish_dist", 1.70)
        )

        self.retreat_target_y_close = float(
            rospy.get_param("~retreat_target_y_close", 0.20)
        )
        self.retreat_target_y_far = float(
            rospy.get_param("~retreat_target_y_far", -0.25)
        )

        # Twist 속도
        self.forward_speed = float(rospy.get_param("~forward_speed", 0.10))
        self.backward_speed = float(rospy.get_param("~backward_speed", 0.08))
        self.turn_speed = float(rospy.get_param("~turn_speed", 0.8))
        self.yaw_turn_speed = float(rospy.get_param("~yaw_turn_speed", 0.5))

        self.marker_timeout_sec = float(
            rospy.get_param("~marker_timeout_sec", 0.7)
        )
        self.stop_burst_count = int(
            rospy.get_param("~stop_burst_count", 3)
        )

        # err_positive일 때 좌회전이면 True
        self.left_if_pos = rospy.get_param("~left_if_pos", True)

        self.control_hz = float(rospy.get_param("~control_hz", 10.0))

        # =====================================================
        # Direct PWM
        # 기존 파라미터 부분은 수정하지 않음
        # =====================================================
        self.use_direct_pwm = bool(
            rospy.get_param("~use_direct_pwm", True)
        )

        # 전진 직진
        self.pwm_forward_l = int(
            rospy.get_param("~pwm_forward_l", 30)
        )
        self.pwm_forward_r = int(
            rospy.get_param("~pwm_forward_r", 30)
        )

        # 전진/일반 회전
        self.pwm_turn_left_l = int(
            rospy.get_param("~pwm_turn_left_l", 0)
        )
        self.pwm_turn_left_r = int(
            rospy.get_param("~pwm_turn_left_r", 13)
        )

        self.pwm_turn_right_l = int(
            rospy.get_param("~pwm_turn_right_l", 13)
        )
        self.pwm_turn_right_r = int(
            rospy.get_param("~pwm_turn_right_r", 0)
        )

        # 후진 직진
        self.pwm_backward_l = int(
            rospy.get_param("~pwm_backward_l", -30)
        )
        self.pwm_backward_r = int(
            rospy.get_param("~pwm_backward_r", -30)
        )

        # 후진하면서 정렬하는 PWM
        self.pwm_retreat_turn_left_l = int(
            rospy.get_param("~pwm_retreat_turn_left_l", -10)
        )
        self.pwm_retreat_turn_left_r = int(
            rospy.get_param("~pwm_retreat_turn_left_r", 0)
        )

        self.pwm_retreat_turn_right_l = int(
            rospy.get_param("~pwm_retreat_turn_right_l", 0)
        )
        self.pwm_retreat_turn_right_r = int(
            rospy.get_param("~pwm_retreat_turn_right_r", -10)
        )

        # yaw 정렬용 PWM
        self.pwm_yaw_left_l = int(
            rospy.get_param("~pwm_yaw_left_l", 0)
        )
        self.pwm_yaw_left_r = int(
            rospy.get_param("~pwm_yaw_left_r", 8)
        )
        self.pwm_yaw_right_l = int(
            rospy.get_param("~pwm_yaw_right_l", 8)
        )
        self.pwm_yaw_right_r = int(
            rospy.get_param("~pwm_yaw_right_r", 0)
        )

        # APPROACH 시작 전 사전 후진
        self.use_pre_backward_on_approach = bool(
            rospy.get_param("~use_pre_backward_on_approach", True)
        )
        self.pre_backward_duration = float(
            rospy.get_param("~pre_backward_duration", 3.0)
        )
        self.pre_backward_l = int(
            rospy.get_param("~pre_backward_l", -30)
        )
        self.pre_backward_r = int(
            rospy.get_param("~pre_backward_r", -30)
        )
        self.pre_backward_start_time = None
        self.pre_backward_next_mode = self.MODE_IDLE

        # =====================================================
        # 버튼 검증 설정
        # 아래 두 모델 경로만 직접 수정
        # ROS 파라미터로 만들지 않고 코드에서 직접 지정
        # =====================================================
        self.button_detect_model_path = "/직접/지정/button_detect.pt"
        self.button_state_model_path = "/직접/지정/button_state.pt"

        # 로봇팔 파일과 동일한 카메라 컬러 토픽 사용
        self.color_topic = "/camera/color/image_raw"

        # 첫 번째 모델: UP 버튼 검출
        self.button_detect_conf_thresh = 0.50
        self.button_detect_iou_thresh = 0.45

        # 두 번째 모델: 잘라낸 버튼 영역 ON/OFF 분류
        self.button_state_imgsz = 224
        self.button_state_min_conf = 0.50

        # 검출 모델에서 사용할 UP 버튼 클래스 이름
        # 실제 모델 클래스명이 다르면 이 문자열만 수정
        self.button_target_class = "UP"

        # 5회의 유효 결과 중 3회 이상 같은 상태면 확정
        self.button_required_samples = 5
        self.button_required_agreement = 3

        # YOLO 추론 간격
        # control timer가 실행될 때마다 무조건 추론하지 않도록 제한
        self.button_inference_interval = 0.20

        # OFF 확정 후 마커 대기시간
        # 필요에 따라 코드에서 직접 수정
        self.button_off_marker_timeout = 5.0

        # 후진 완료 직후 정지 명령을 보낸 다음
        # 마커/버튼 검증을 시작하기 전 짧은 안정화 시간
        self.post_backward_settle_time = 0.20

        # =====================================================
        # YOLO 모델 로드
        # 로봇팔 제어 파일의 모델 로드 방식 참고
        # =====================================================
        if not os.path.exists(self.button_detect_model_path):
            raise FileNotFoundError(
                "Button detect model not found: %s"
                % self.button_detect_model_path
            )

        if not os.path.exists(self.button_state_model_path):
            raise FileNotFoundError(
                "Button state model not found: %s"
                % self.button_state_model_path
            )

        logging.getLogger("ultralytics").setLevel(logging.ERROR)

        rospy.loginfo(
            "[marker_ctrl] loading button detect model: %s",
            self.button_detect_model_path
        )
        self.button_detect_model = YOLO(
            self.button_detect_model_path
        )

        rospy.loginfo(
            "[marker_ctrl] loading button state model: %s",
            self.button_state_model_path
        )
        self.button_state_model = YOLO(
            self.button_state_model_path
        )

        # =====================================================
        # ROS pub/sub
        # =====================================================
        self.cmd_pub = rospy.Publisher(
            self.cmd_doc_topic,
            Twist,
            queue_size=10
        )
        self.char_pub = rospy.Publisher(
            self.char_cmd_topic,
            String,
            queue_size=10
        )
        self.pwm_pub = rospy.Publisher(
            self.doc_pwm_topic,
            Int16MultiArray,
            queue_size=10
        )

        self.topic_done = rospy.get_param(
            "~topic_done",
            "/marker_mission_done"
        )
        self.docking_done_topic = rospy.get_param(
            "~docking_done_topic",
            "/marker_docking_done"
        )

        self.done_pub = rospy.Publisher(
            self.docking_done_topic,
            Bool,
            queue_size=1
        )
        self.mdone_pub = rospy.Publisher(
            self.topic_done,
            Int32,
            queue_size=1,
            latch=True
        )

        # 추가 실패 신호
        self.mission_fail_topic = "/marker_mission_fail"
        self.mfail_pub = rospy.Publisher(
            self.mission_fail_topic,
            Int32,
            queue_size=1,
            latch=True
        )

        self.mode_pub = rospy.Publisher(
            "~mode",
            String,
            queue_size=1,
            latch=True
        )

        rospy.Subscriber(
            self.topic_pose_x,
            Float64,
            self.pose_x_callback,
            queue_size=1
        )
        rospy.Subscriber(
            self.topic_pose_z,
            Float64,
            self.pose_z_callback,
            queue_size=1
        )
        rospy.Subscriber(
            self.topic_marker_id,
            Int32,
            self.marker_id_callback,
            queue_size=1
        )
        rospy.Subscriber(
            self.topic_mission,
            Int32,
            self.mission_callback,
            queue_size=1
        )
        rospy.Subscriber(
            self.topic_yaw,
            Float64,
            self.yaw_callback,
            queue_size=1
        )

        # 버튼 검증용 컬러 영상
        self.bridge = CvBridge()
        self.latest_color_image = None
        self.latest_color_time = 0.0
        self.latest_color_seq = 0

        rospy.Subscriber(
            self.color_topic,
            Image,
            self.color_callback,
            queue_size=1,
            buff_size=2**24
        )

        # =====================================================
        # State
        # =====================================================
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

        # 버튼 검증 상태
        self.button_state_samples = []
        self.confirmed_button_state = None
        self.last_button_inference_time = 0.0

        # UP 버튼 위치는 최초 1회만 검출한 뒤 고정해서 사용
        self.cached_up_button_box = None
        self.cached_up_button_detect_conf = 0.0

        # 같은 카메라 프레임을 중복 분류하지 않기 위한 시퀀스 번호
        self.last_classified_color_seq = -1

        # 후진이 완전히 끝난 시점
        self.pre_backward_completed_time = None

        # OFF 확정 후 마커 대기 시작 시간
        self.off_marker_wait_start_time = None

        rospy.Timer(
            rospy.Duration(1.0 / self.control_hz),
            self._timer_cb
        )

        rospy.loginfo(
            "[marker_pose_controller_grad_demo] ready "
            "(control_hz=%.1f, angle_threshold=%.3f, "
            "yaw_threshold=%.3f, target_yaw=%.3f, direct_pwm=%s)",
            self.control_hz,
            self.angle_threshold,
            self.yaw_threshold,
            self.target_yaw,
            str(self.use_direct_pwm)
        )

        rospy.loginfo(
            "[marker_ctrl] APPROACH logic: "
            "dist>%.2f -> x=%.2f | "
            "%.2f<dist<=%.2f -> x=%.2f | "
            "%.2f<dist<=%.2f -> x=%.2f | "
            "%.2f<dist<=%.2f -> x=%.2f | "
            "dist<=%.2f -> yaw",
            self.approach_stage1_dist,
            self.approach_target_y_far,
            self.approach_stage1_5_dist,
            self.approach_stage1_dist,
            self.approach_target_y_mid,
            self.approach_stage2_dist,
            self.approach_stage1_5_dist,
            self.approach_target_y_mid2,
            self.approach_stop_dist,
            self.approach_stage2_dist,
            self.approach_target_y_near,
            self.approach_stop_dist
        )

        rospy.loginfo(
            "[marker_ctrl] button validation: "
            "samples=%d agreement=%d off_marker_timeout=%.2fs "
            "target_class=%s color_topic=%s",
            self.button_required_samples,
            self.button_required_agreement,
            self.button_off_marker_timeout,
            self.button_target_class,
            self.color_topic
        )

        rospy.spin()

    # =========================================================
    # Callbacks
    # =========================================================
    def color_callback(self, msg):
        try:
            self.latest_color_image = self.bridge.imgmsg_to_cv2(
                msg,
                desired_encoding="bgr8"
            )
            self.latest_color_time = time.time()
            self.latest_color_seq += 1

        except Exception as e:
            rospy.logerr_throttle(
                1.0,
                "[marker_ctrl] color callback failed: %s",
                str(e)
            )

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
            self.pre_backward_start_time = None
            self.pre_backward_next_mode = self.MODE_IDLE

            self.reset_button_validation_state()

            self.publish_stop("mission=0 -> idle")
            self.mode_pub.publish("IDLE")
            return

        if val > 0:
            self.expected_marker_id = val

            self.reset_button_validation_state()

            if self.use_pre_backward_on_approach:
                self.mode = self.MODE_PRE_BACKWARD
                self.pre_backward_start_time = time.time()

                # 기존에는 APPROACH로 바로 전환했지만,
                # 이제 후진 완료 후 검증 상태로 전환
                self.pre_backward_next_mode = self.MODE_CHECK_ENTRY

                self.mode_pub.publish("PRE_BACKWARD")

                rospy.loginfo(
                    "[marker_ctrl] PRE_BACKWARD start before APPROACH "
                    "marker id=%d, duration=%.2fs, pwm=[%d,%d]",
                    self.expected_marker_id,
                    self.pre_backward_duration,
                    self.pre_backward_l,
                    self.pre_backward_r
                )

            else:
                # 사전 후진 기능을 사용하지 않는 경우는
                # 기존 동작대로 바로 APPROACH
                self.mode = self.MODE_APPROACH
                self.pre_backward_start_time = None
                self.pre_backward_next_mode = self.MODE_IDLE
                self.mode_pub.publish("APPROACH")

                rospy.loginfo(
                    "[marker_ctrl] APPROACH start for marker id=%d",
                    self.expected_marker_id
                )

        else:
            # 기존 RETREAT 로직 유지
            self.mode = self.MODE_RETREAT
            self.expected_marker_id = abs(val)
            self.pre_backward_start_time = None
            self.pre_backward_next_mode = self.MODE_IDLE

            self.reset_button_validation_state()

            self.mode_pub.publish("RETREAT")

            rospy.loginfo(
                "[marker_ctrl] RETREAT start for marker id=%d",
                self.expected_marker_id
            )

        self.publish_stop("new mission start")

    def _timer_cb(self, _event):
        self.run_control()

    # =========================================================
    # 버튼 검증 관련 함수
    # =========================================================
    def reset_button_validation_state(self):
        self.button_state_samples = []
        self.confirmed_button_state = None
        self.last_button_inference_time = 0.0

        # 새 검증 미션마다 UP 위치를 다시 한 번 검출
        self.cached_up_button_box = None
        self.cached_up_button_detect_conf = 0.0
        self.last_classified_color_seq = -1

        self.pre_backward_completed_time = None
        self.off_marker_wait_start_time = None

    def normalize_button_state(self, state):
        """
        분류 모델의 클래스 이름을 ON/OFF로 통일한다.
        """
        raw = (
            str(state)
            .strip()
            .lower()
            .replace("-", "_")
            .replace(" ", "_")
        )

        on_aliases = {
            "on",
            "pressed",
            "press",
            "active",
            "selected",
            "button_on",
            "up_on",
            "1",
            "true"
        }

        off_aliases = {
            "off",
            "not_pressed",
            "unpressed",
            "released",
            "inactive",
            "button_off",
            "up_off",
            "0",
            "false"
        }

        if raw in on_aliases:
            return "on"

        if raw in off_aliases:
            return "off"

        return "unknown"

    def latest_color_valid(self):
        if self.latest_color_image is None:
            return False

        # 너무 오래된 프레임은 사용하지 않음
        if time.time() - self.latest_color_time > 1.0:
            return False

        return True

    def detect_up_button_box(self, bgr):
        """
        첫 번째 YOLO 모델로 UP 버튼 위치를 검출한다.

        반환:
          (x1, y1, x2, y2, confidence)
          검출 실패 시 None
        """
        try:
            result = self.button_detect_model.predict(
                source=bgr,
                conf=self.button_detect_conf_thresh,
                iou=self.button_detect_iou_thresh,
                verbose=False
            )[0]

            if (
                result is None
                or result.boxes is None
                or len(result.boxes) == 0
            ):
                return None

            names = result.names
            best_box = None
            best_conf = -1.0

            for box in result.boxes:
                cls_id = int(box.cls[0].item())
                label = str(names.get(cls_id, str(cls_id))).strip()
                conf = float(box.conf[0].item())

                if (
                    self.button_target_class
                    and label.upper() != self.button_target_class.upper()
                ):
                    continue

                if conf > best_conf:
                    x1, y1, x2, y2 = (
                        box.xyxy[0]
                        .cpu()
                        .numpy()
                        .astype(int)
                    )

                    best_box = (
                        int(x1),
                        int(y1),
                        int(x2),
                        int(y2),
                        conf
                    )
                    best_conf = conf

            return best_box

        except Exception as e:
            rospy.logwarn_throttle(
                1.0,
                "[marker_ctrl] UP button detection failed: %s",
                str(e)
            )
            return None

    def classify_button_state(self, bgr, x1, y1, x2, y2):
        """
        두 번째 YOLO 분류 모델로 잘라낸 UP 버튼 영역의
        ON/OFF 상태를 판단한다.

        반환:
          ("on", confidence)
          ("off", confidence)
          ("unknown", confidence)
        """
        h, w = bgr.shape[:2]

        x1 = clamp(int(x1), 0, w - 1)
        x2 = clamp(int(x2), 0, w)
        y1 = clamp(int(y1), 0, h - 1)
        y2 = clamp(int(y2), 0, h)

        if x2 <= x1 or y2 <= y1:
            return "unknown", 0.0

        crop = bgr[y1:y2, x1:x2]

        if crop.size == 0:
            return "unknown", 0.0

        try:
            result = self.button_state_model.predict(
                source=crop,
                imgsz=self.button_state_imgsz,
                verbose=False
            )[0]

            if getattr(result, "probs", None) is None:
                return "unknown", 0.0

            names = result.names
            class_id = int(result.probs.top1)
            confidence = float(result.probs.top1conf)

            raw_label = str(
                names.get(class_id, str(class_id))
                if isinstance(names, dict)
                else names[class_id]
            )

            normalized = self.normalize_button_state(raw_label)

            if confidence < self.button_state_min_conf:
                return "unknown", confidence

            return normalized, confidence

        except Exception as e:
            rospy.logwarn_throttle(
                1.0,
                "[marker_ctrl] button state classification failed: %s",
                str(e)
            )
            return "unknown", 0.0

    def infer_button_state_once(self):
        """
        UP 버튼 상태를 한 번 판정한다.

        동작 방식:
          1) cached_up_button_box가 없을 때만 첫 번째 YOLO 모델로
             전체 화면에서 UP 버튼 위치를 한 번 검출한다.
          2) UP 위치가 한 번 검출되면 이후에는 첫 번째 모델을 다시 실행하지 않는다.
          3) 새 카메라 프레임이 들어올 때마다 저장된 동일 좌표를 crop하여
             두 번째 YOLO 모델로 ON/OFF 상태만 분류한다.

        반환:
          ("on", confidence)
          ("off", confidence)
          ("unknown", confidence)
        """
        if not self.latest_color_valid():
            rospy.logwarn_throttle(
                1.0,
                "[marker_ctrl] waiting recent color image for button check"
            )
            return "unknown", 0.0

        # 같은 카메라 프레임을 두 번 분류하지 않음
        current_color_seq = self.latest_color_seq
        if (
            self.cached_up_button_box is not None
            and current_color_seq == self.last_classified_color_seq
        ):
            rospy.loginfo_throttle(
                1.0,
                "[marker_ctrl] waiting new color frame for next ON/OFF sample"
            )
            return "unknown", 0.0

        try:
            image = self.latest_color_image.copy()
        except Exception:
            return "unknown", 0.0

        # -----------------------------------------------------
        # UP 위치 검출은 최초 성공할 때까지만 실행
        # -----------------------------------------------------
        if self.cached_up_button_box is None:
            detected = self.detect_up_button_box(image)

            if detected is None:
                rospy.loginfo_throttle(
                    1.0,
                    "[marker_ctrl] UP button not detected"
                )
                return "unknown", 0.0

            x1, y1, x2, y2, detect_conf = detected

            self.cached_up_button_box = (
                int(x1),
                int(y1),
                int(x2),
                int(y2)
            )
            self.cached_up_button_detect_conf = float(detect_conf)

            rospy.loginfo(
                "[marker_ctrl] UP button position cached once: "
                "box=(%d,%d,%d,%d) detect_conf=%.3f",
                int(x1),
                int(y1),
                int(x2),
                int(y2),
                float(detect_conf)
            )

        x1, y1, x2, y2 = self.cached_up_button_box

        # 현재 새 프레임의 동일 좌표 영역만 ON/OFF 분류
        state, state_conf = self.classify_button_state(
            image,
            x1,
            y1,
            x2,
            y2
        )

        # 분류를 시도한 프레임 번호 저장
        self.last_classified_color_seq = current_color_seq

        rospy.loginfo(
            "[marker_ctrl] button state only inference: "
            "cached_detect_conf=%.3f state=%s state_conf=%.3f "
            "color_seq=%d",
            self.cached_up_button_detect_conf,
            state,
            state_conf,
            current_color_seq
        )

        return state, state_conf

    def add_button_state_sample(self, state):
        """
        유효한 ON/OFF 결과만 샘플 목록에 추가한다.

        5개의 유효 결과가 쌓인 뒤,
        ON 또는 OFF가 3회 이상이면 상태를 확정한다.
        """
        if state not in ["on", "off"]:
            return None

        self.button_state_samples.append(state)

        # 최근 5개의 유효 결과만 유지
        if len(self.button_state_samples) > self.button_required_samples:
            self.button_state_samples = self.button_state_samples[
                -self.button_required_samples:
            ]

        on_count = self.button_state_samples.count("on")
        off_count = self.button_state_samples.count("off")

        rospy.loginfo(
            "[marker_ctrl] button samples=%s "
            "valid=%d/%d on=%d off=%d",
            str(self.button_state_samples),
            len(self.button_state_samples),
            self.button_required_samples,
            on_count,
            off_count
        )

        # 유효 결과 5회가 모이기 전에는 확정하지 않음
        if len(self.button_state_samples) < self.button_required_samples:
            return None

        if on_count >= self.button_required_agreement:
            return "on"

        if off_count >= self.button_required_agreement:
            return "off"

        # 5개가 쌓였지만 3회 이상 같은 결과가 없는 경우
        # 이론상 ON/OFF 두 상태만 있으면 발생하지 않지만,
        # 안전하게 다시 수집
        rospy.logwarn(
            "[marker_ctrl] no majority in button samples -> reset samples"
        )
        self.button_state_samples = []
        return None

    def transition_to_approach(self, reason):
        """
        검증이 끝난 뒤 기존 APPROACH 로직으로 진입한다.
        """
        self.mode = self.MODE_APPROACH
        self.confirmed_button_state = None
        self.off_marker_wait_start_time = None

        self.mode_pub.publish("APPROACH")

        rospy.loginfo(
            "[marker_ctrl] validation complete -> APPROACH "
            "marker id=%s | %s",
            str(self.expected_marker_id),
            reason
        )

    def fail_mission(self, reason="failed"):
        """
        마커 전진 미션 실패 처리.

        성공 토픽은 발행하지 않고
        /marker_mission_fail에 현재 마커 ID를 발행한다.
        """
        marker_id = (
            self.expected_marker_id
            if self.expected_marker_id is not None
            else -1
        )

        self.publish_stop(reason)

        if marker_id > 0:
            self.mfail_pub.publish(Int32(data=marker_id))

        rospy.logerr(
            "[marker_ctrl] mission failed marker=%s | %s",
            str(marker_id),
            reason
        )

        self.mode = self.MODE_IDLE
        self.expected_marker_id = None
        self.pre_backward_start_time = None
        self.pre_backward_next_mode = self.MODE_IDLE

        self.reset_button_validation_state()

        self.mode_pub.publish("IDLE")

    # =========================================================
    # Utils
    # 기존 함수 유지
    # =========================================================
    def marker_visible_recently(self):
        now = time.time()

        return (
            ((now - self.last_pose_time) < self.marker_timeout_sec)
            and
            ((now - self.last_marker_time) < self.marker_timeout_sec)
        )

    def yaw_visible_recently(self):
        now = time.time()
        return (
            (now - self.last_yaw_time)
            < self.marker_timeout_sec
        )

    def expected_marker_visible(self):
        if self.expected_marker_id is None:
            return False

        if self.current_marker_id is None:
            return False

        return (
            self.current_marker_id
            == self.expected_marker_id
        )

    def target_marker_visible_now(self):
        """
        최근 마커 데이터가 유효하고,
        현재 마커 ID가 목표 마커 ID인지 함께 확인한다.
        """
        return (
            self.marker_visible_recently()
            and
            self.expected_marker_visible()
        )

    def publish_twist(self, lin, ang, reason=""):
        twist = Twist()
        twist.linear.x = lin
        twist.angular.z = ang

        self.cmd_pub.publish(twist)

        rospy.loginfo(
            "[marker_ctrl] cmd_vel lin=%.3f ang=%.3f | %s",
            lin,
            ang,
            reason
        )

    def publish_pwm(self, left, right, reason=""):
        msg = Int16MultiArray()
        msg.data = [int(left), int(right)]

        self.pwm_pub.publish(msg)

        rospy.loginfo(
            "[marker_ctrl] pwm [%d, %d] | %s",
            int(left),
            int(right),
            reason
        )

    def publish_stop(self, reason="stop"):
        twist = Twist()

        stop_pwm = Int16MultiArray()
        stop_pwm.data = [0, 0]

        for _ in range(self.stop_burst_count):
            self.cmd_pub.publish(twist)
            self.pwm_pub.publish(stop_pwm)

        self.char_pub.publish(String(data='x'))

        rospy.loginfo(
            "[marker_ctrl] STOP | %s",
            reason
        )

    def finish_mission(self, reason="done"):
        marker_id = (
            self.expected_marker_id
            if self.expected_marker_id is not None
            else -1
        )

        self.publish_stop(reason)

        # 기존 성공 신호 유지
        self.done_pub.publish(Bool(data=True))

        if marker_id > 0:
            self.mdone_pub.publish(
                Int32(data=marker_id)
            )

        self.last_completed_marker_id = marker_id

        rospy.loginfo(
            "[marker_ctrl] mission finished marker=%s | %s",
            str(marker_id),
            reason
        )

        self.mode = self.MODE_IDLE
        self.expected_marker_id = None
        self.pre_backward_start_time = None
        self.pre_backward_next_mode = self.MODE_IDLE

        self.reset_button_validation_state()

        self.mode_pub.publish("IDLE")

    def _turn_ang(self, err_positive, turn_speed=None):
        ts = (
            self.turn_speed
            if turn_speed is None
            else float(turn_speed)
        )

        left = (
            (err_positive and self.left_if_pos)
            or
            ((not err_positive) and (not self.left_if_pos))
        )

        return +ts if left else -ts

    def _turn_pwm(self, err_positive):
        left = (
            (err_positive and self.left_if_pos)
            or
            ((not err_positive) and (not self.left_if_pos))
        )

        if left:
            return (
                self.pwm_turn_left_l,
                self.pwm_turn_left_r
            )

        return (
            self.pwm_turn_right_l,
            self.pwm_turn_right_r
        )

    def _retreat_turn_pwm(self, err_positive):
        left = (
            (err_positive and self.left_if_pos)
            or
            ((not err_positive) and (not self.left_if_pos))
        )

        if left:
            return (
                self.pwm_retreat_turn_left_l,
                self.pwm_retreat_turn_left_r
            )

        return (
            self.pwm_retreat_turn_right_l,
            self.pwm_retreat_turn_right_r
        )

    def _yaw_turn_pwm(self, err_positive):
        left = (
            (err_positive and self.left_if_pos)
            or
            ((not err_positive) and (not self.left_if_pos))
        )

        if left:
            return (
                self.pwm_yaw_left_l,
                self.pwm_yaw_left_r
            )

        return (
            self.pwm_yaw_right_l,
            self.pwm_yaw_right_r
        )

    def drive_with_target_y(
        self,
        dist,
        pose_x,
        target_y,
        forward_motion=True,
        phase=""
    ):
        """
        기존 APPROACH/RETREAT 주행 로직 유지
        """
        err = pose_x - target_y

        if abs(err) <= self.angle_threshold:
            if forward_motion:
                if self.use_direct_pwm:
                    self.publish_pwm(
                        self.pwm_forward_l,
                        self.pwm_forward_r,
                        "%s forward dist=%.3f "
                        "pose_x=%.3f target_y=%.3f err=%.3f"
                        % (
                            phase,
                            dist,
                            pose_x,
                            target_y,
                            err
                        )
                    )

                else:
                    self.publish_twist(
                        self.forward_speed,
                        0.0,
                        "%s forward dist=%.3f "
                        "pose_x=%.3f target_y=%.3f err=%.3f"
                        % (
                            phase,
                            dist,
                            pose_x,
                            target_y,
                            err
                        )
                    )

            else:
                if self.use_direct_pwm:
                    self.publish_pwm(
                        self.pwm_backward_l,
                        self.pwm_backward_r,
                        "%s backward dist=%.3f "
                        "pose_x=%.3f target_y=%.3f err=%.3f"
                        % (
                            phase,
                            dist,
                            pose_x,
                            target_y,
                            err
                        )
                    )

                else:
                    self.publish_twist(
                        -self.backward_speed,
                        0.0,
                        "%s backward dist=%.3f "
                        "pose_x=%.3f target_y=%.3f err=%.3f"
                        % (
                            phase,
                            dist,
                            pose_x,
                            target_y,
                            err
                        )
                    )

        else:
            if self.use_direct_pwm:
                if forward_motion:
                    left_pwm, right_pwm = self._turn_pwm(
                        err > 0.0
                    )

                    self.publish_pwm(
                        left_pwm,
                        right_pwm,
                        "%s turn dist=%.3f "
                        "pose_x=%.3f target_y=%.3f err=%.3f"
                        % (
                            phase,
                            dist,
                            pose_x,
                            target_y,
                            err
                        )
                    )

                else:
                    left_pwm, right_pwm = (
                        self._retreat_turn_pwm(
                            err > 0.0
                        )
                    )

                    self.publish_pwm(
                        left_pwm,
                        right_pwm,
                        "%s retreat_turn dist=%.3f "
                        "pose_x=%.3f target_y=%.3f err=%.3f"
                        % (
                            phase,
                            dist,
                            pose_x,
                            target_y,
                            err
                        )
                    )

            else:
                if forward_motion:
                    ang = self._turn_ang(
                        err > 0.0
                    )

                    self.publish_twist(
                        0.0,
                        ang,
                        "%s turn dist=%.3f "
                        "pose_x=%.3f target_y=%.3f "
                        "err=%.3f ang=%.2f"
                        % (
                            phase,
                            dist,
                            pose_x,
                            target_y,
                            err,
                            ang
                        )
                    )

                else:
                    ang = self._turn_ang(
                        err > 0.0
                    )

                    self.publish_twist(
                        -self.backward_speed,
                        ang,
                        "%s backward_turn dist=%.3f "
                        "pose_x=%.3f target_y=%.3f "
                        "err=%.3f ang=%.2f"
                        % (
                            phase,
                            dist,
                            pose_x,
                            target_y,
                            err,
                            ang
                        )
                    )

    def align_yaw_only(self, yaw, phase="YAW_ALIGN"):
        """
        기존 yaw 정렬 로직 유지
        """
        err = yaw - self.target_yaw

        if abs(err) <= self.yaw_threshold:
            self.finish_mission(
                "%s complete yaw=%.3f "
                "target=%.3f err=%.3f"
                % (
                    phase,
                    yaw,
                    self.target_yaw,
                    err
                )
            )
            return

        if self.use_direct_pwm:
            left_pwm, right_pwm = self._yaw_turn_pwm(
                err > 0.0
            )

            self.publish_pwm(
                left_pwm,
                right_pwm,
                "%s turning yaw=%.3f "
                "target=%.3f err=%.3f threshold=%.3f"
                % (
                    phase,
                    yaw,
                    self.target_yaw,
                    err,
                    self.yaw_threshold
                )
            )

        else:
            ang = self._turn_ang(
                err > 0.0,
                self.yaw_turn_speed
            )

            self.publish_twist(
                0.0,
                ang,
                "%s turning yaw=%.3f "
                "target=%.3f err=%.3f "
                "threshold=%.3f ang=%.2f"
                % (
                    phase,
                    yaw,
                    self.target_yaw,
                    err,
                    self.yaw_threshold,
                    ang
                )
            )

    # =========================================================
    # Main Control
    # =========================================================
    def run_control(self):
        if self.mode == self.MODE_IDLE:
            return

        # -----------------------------------------------------
        # PRE_BACKWARD
        # 기존 3초 후진 PWM과 시간 제어는 그대로 유지
        # -----------------------------------------------------
        if self.mode == self.MODE_PRE_BACKWARD:
            if self.pre_backward_start_time is None:
                self.pre_backward_start_time = time.time()

            elapsed = (
                time.time()
                - self.pre_backward_start_time
            )

            if elapsed < self.pre_backward_duration:
                self.publish_pwm(
                    self.pre_backward_l,
                    self.pre_backward_r,
                    "PRE_BACKWARD %.2f/%.2fs "
                    "before APPROACH marker id=%s"
                    % (
                        elapsed,
                        self.pre_backward_duration,
                        str(self.expected_marker_id)
                    )
                )
                return

            # 후진이 완전히 끝난 후 정지
            self.publish_stop(
                "PRE_BACKWARD done -> entry validation"
            )

            self.mode = self.pre_backward_next_mode
            self.pre_backward_start_time = None
            self.pre_backward_next_mode = self.MODE_IDLE

            # 후진 중 들어온 버튼 판정 기록은 사용하지 않음
            self.button_state_samples = []
            self.confirmed_button_state = None
            self.last_button_inference_time = 0.0
            self.cached_up_button_box = None
            self.cached_up_button_detect_conf = 0.0
            self.last_classified_color_seq = -1

            # 후진 완료 시점 저장
            self.pre_backward_completed_time = time.time()
            self.off_marker_wait_start_time = None

            self.mode_pub.publish("CHECK_ENTRY")

            rospy.loginfo(
                "[marker_ctrl] PRE_BACKWARD fully completed "
                "-> CHECK_ENTRY marker id=%s",
                str(self.expected_marker_id)
            )
            return

        # -----------------------------------------------------
        # CHECK_ENTRY
        # 후진 완료 후 잠깐 안정화한 다음
        # 목표 마커가 보이는지 먼저 확인
        # -----------------------------------------------------
        if self.mode == self.MODE_CHECK_ENTRY:
            self.publish_stop(
                "CHECK_ENTRY waiting robot settle"
            )

            if self.pre_backward_completed_time is None:
                self.pre_backward_completed_time = time.time()
                return

            settle_elapsed = (
                time.time()
                - self.pre_backward_completed_time
            )

            if settle_elapsed < self.post_backward_settle_time:
                return

            # 후진 완료 후 목표 마커가 보이면
            # 버튼 ON/OFF 검사를 생략하고 바로 탑승
            if self.target_marker_visible_now():
                self.transition_to_approach(
                    "target marker already visible after backward"
                )
                return

            # 마커가 안 보일 때만 버튼 검증 시작
            self.mode = self.MODE_CHECK_BUTTON
            self.button_state_samples = []
            self.confirmed_button_state = None
            self.last_button_inference_time = 0.0
            self.cached_up_button_box = None
            self.cached_up_button_detect_conf = 0.0
            self.last_classified_color_seq = -1

            self.mode_pub.publish("CHECK_BUTTON")

            rospy.loginfo(
                "[marker_ctrl] marker not visible after backward "
                "-> start UP button validation"
            )
            return

        # -----------------------------------------------------
        # CHECK_BUTTON
        # 두 모델로 UP 버튼 상태를 직접 확인
        # -----------------------------------------------------
        if self.mode == self.MODE_CHECK_BUTTON:
            self.publish_stop(
                "CHECK_BUTTON robot stopped"
            )

            # 버튼 검증 중이라도 목표 마커가 나타나면
            # 더 이상 버튼 검증하지 않고 바로 탑승
            if self.target_marker_visible_now():
                self.transition_to_approach(
                    "target marker appeared during button validation"
                )
                return

            now = time.time()

            if (
                now - self.last_button_inference_time
                < self.button_inference_interval
            ):
                return

            self.last_button_inference_time = now

            state, state_conf = self.infer_button_state_once()

            if state not in ["on", "off"]:
                rospy.loginfo_throttle(
                    1.0,
                    "[marker_ctrl] button state UNKNOWN "
                    "-> not counted"
                )
                return

            rospy.loginfo(
                "[marker_ctrl] valid button state sample: "
                "%s conf=%.3f",
                state,
                state_conf
            )

            confirmed = self.add_button_state_sample(
                state
            )

            if confirmed is None:
                return

            self.confirmed_button_state = confirmed

            if confirmed == "on":
                # ON이면 마커가 보일 때까지 계속 대기
                self.mode = self.MODE_WAIT_MARKER_ON
                self.mode_pub.publish("WAIT_MARKER_ON")

                rospy.loginfo(
                    "[marker_ctrl] UP button confirmed ON "
                    "-> wait marker without failure timeout"
                )
                return

            # OFF이면 이 시점부터 제한시간 측정
            self.mode = self.MODE_WAIT_MARKER_OFF
            self.off_marker_wait_start_time = time.time()
            self.mode_pub.publish("WAIT_MARKER_OFF")

            rospy.loginfo(
                "[marker_ctrl] UP button confirmed OFF "
                "-> wait marker for %.2fs",
                self.button_off_marker_timeout
            )
            return

        # -----------------------------------------------------
        # WAIT_MARKER_ON
        # ON 확정이면 마커가 나타날 때까지 계속 대기
        # -----------------------------------------------------
        if self.mode == self.MODE_WAIT_MARKER_ON:
            self.publish_stop(
                "UP ON confirmed -> waiting target marker"
            )

            if self.target_marker_visible_now():
                self.transition_to_approach(
                    "UP ON and target marker detected"
                )
                return

            rospy.loginfo_throttle(
                1.0,
                "[marker_ctrl] UP is ON, waiting marker id=%s",
                str(self.expected_marker_id)
            )
            return

        # -----------------------------------------------------
        # WAIT_MARKER_OFF
        # OFF 확정이면 제한시간 동안 마커 대기
        # -----------------------------------------------------
        if self.mode == self.MODE_WAIT_MARKER_OFF:
            self.publish_stop(
                "UP OFF confirmed -> waiting target marker"
            )

            if self.target_marker_visible_now():
                self.transition_to_approach(
                    "UP OFF but target marker detected "
                    "within timeout"
                )
                return

            if self.off_marker_wait_start_time is None:
                self.off_marker_wait_start_time = time.time()

            elapsed = (
                time.time()
                - self.off_marker_wait_start_time
            )

            if elapsed >= self.button_off_marker_timeout:
                self.fail_mission(
                    "UP OFF confirmed and target marker "
                    "not detected for %.2fs"
                    % self.button_off_marker_timeout
                )
                return

            rospy.loginfo_throttle(
                1.0,
                "[marker_ctrl] UP OFF, waiting marker "
                "%.2f/%.2fs id=%s",
                elapsed,
                self.button_off_marker_timeout,
                str(self.expected_marker_id)
            )
            return

        # -----------------------------------------------------
        # 아래부터 기존 마커 데이터 유효성 확인 로직 유지
        # -----------------------------------------------------
        if not self.marker_visible_recently():
            self.publish_stop("marker stale")
            return

        if not self.expected_marker_visible():
            self.publish_stop(
                "waiting expected marker id=%s, seen=%s"
                % (
                    str(self.expected_marker_id),
                    str(self.current_marker_id)
                )
            )
            return

        pose_x = self.current_pose_x
        dist = self.current_distance
        yaw = self.current_yaw

        # -----------------------------------------------------
        # APPROACH
        # 기존 마커 기반 전진 로직 그대로 유지
        # -----------------------------------------------------
        if self.mode == self.MODE_APPROACH:
            if dist > self.approach_stage1_dist:
                self.drive_with_target_y(
                    dist=dist,
                    pose_x=pose_x,
                    target_y=self.approach_target_y_far,
                    forward_motion=True,
                    phase="APPROACH_STAGE1_X0"
                )
                return

            if dist > self.approach_stage1_5_dist:
                self.drive_with_target_y(
                    dist=dist,
                    pose_x=pose_x,
                    target_y=self.approach_target_y_mid,
                    forward_motion=True,
                    phase="APPROACH_STAGE2_X_NEG03"
                )
                return

            if dist > self.approach_stage2_dist:
                self.drive_with_target_y(
                    dist=dist,
                    pose_x=pose_x,
                    target_y=self.approach_target_y_mid2,
                    forward_motion=True,
                    phase="APPROACH_STAGE3_X_NEG01"
                )
                return

            if dist > self.approach_stop_dist:
                self.drive_with_target_y(
                    dist=dist,
                    pose_x=pose_x,
                    target_y=self.approach_target_y_near,
                    forward_motion=True,
                    phase="APPROACH_STAGE4_X_POS02"
                )
                return

            self.publish_stop(
                "APPROACH reached stop point "
                "dist=%.3f -> yaw align"
                % dist
            )

            if not self.yaw_visible_recently():
                rospy.logwarn(
                    "[marker_ctrl] yaw not updated recently, "
                    "cannot align yaw"
                )
                return

            self.align_yaw_only(
                yaw,
                phase="APPROACH_YAW_ALIGN"
            )
            return

        # -----------------------------------------------------
        # RETREAT
        # 기존 마커 기반 후진 로직 그대로 유지
        # -----------------------------------------------------
        if self.mode == self.MODE_RETREAT:
            if dist < self.retreat_stage1_limit:
                self.drive_with_target_y(
                    dist=dist,
                    pose_x=pose_x,
                    target_y=self.retreat_target_y_close,
                    forward_motion=False,
                    phase="RETREAT_STAGE1"
                )
                return

            if dist < self.retreat_finish_dist:
                self.drive_with_target_y(
                    dist=dist,
                    pose_x=pose_x,
                    target_y=self.retreat_target_y_far,
                    forward_motion=False,
                    phase="RETREAT_STAGE2"
                )
                return

            self.finish_mission(
                "RETREAT reached finish distance "
                "dist=%.3f"
                % dist
            )
            return


if __name__ == '__main__':
    try:
        MarkerPoseControllerGradDemo()

    except rospy.ROSInterruptException:
        pass
