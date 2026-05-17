#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import cv2
import numpy as np
import rospy
from cv_bridge import CvBridge
from sensor_msgs.msg import Image, CameraInfo
from std_msgs.msg import Int16MultiArray, Int32, Bool, String
from ultralytics import YOLO


class MarkerPoseControllerPanel:
    def __init__(self):
        rospy.init_node("marker_pose_controller", anonymous=False)

        default_model = os.path.expanduser("/home/inwoong/panel_best.pt")

        # =========================================================
        # Params
        # =========================================================
        self.model_path = rospy.get_param("~model_path", default_model)

        self.rgb_topic = rospy.get_param("~rgb_topic", "/camera/color/image_raw")
        self.depth_topic = rospy.get_param("~depth_topic", "/camera/aligned_depth_to_color/image_raw")
        self.camera_info_topic = rospy.get_param("~camera_info_topic", "/camera/color/camera_info")

        # 기존 /panel_mission 대신 현재 구조에 맞춤
        self.topic_mission = rospy.get_param("~topic_mission", "/camera_mission_panel")

        # 완료 토픽은 기존 호환 유지
        self.topic_done = rospy.get_param("~topic_done", "/panel_mission_done")
        self.docking_done_topic = rospy.get_param("~docking_done_topic", "/panel_docking_done")

        # 출력 토픽도 launch 파라미터 반영 가능하게 정리
        self.cmd_doc_topic = rospy.get_param("~cmd_doc_topic", "/cmd_vel_doc")
        self.char_cmd_topic = rospy.get_param("~char_cmd_topic", "/doc_cmd_char")
        self.doc_pwm_topic = rospy.get_param("~doc_pwm_topic", "/doc_pwm_cmd")

        # left_panel만 추종
        self.target_class_name = rospy.get_param("~target_class_name", "left_panel")

        # launch와 기존 코드 둘 다 호환
        self.conf_thresh = float(
            rospy.get_param("~conf_threshold",
                            rospy.get_param("~conf_thresh", 0.50))
        )

        self.depth_patch = int(
            rospy.get_param("~depth_window",
                            rospy.get_param("~depth_patch", 5))
        )

        self.view_image = bool(rospy.get_param("~view_image", False))
        # 카메라 시각화/디버그 이미지 publish 부하 방지용
        # False이면 OpenCV 창 표시와 /panel/debug_image publish를 모두 끈다.
        self.publish_debug_image_enabled = bool(rospy.get_param("~publish_debug_image", False))

        # 접근 제어 파라미터
        self.stop_distance_m = float(
            rospy.get_param("~target_stop_z",
                            rospy.get_param("~stop_distance_m", 0.50))
        )

        # 기존 픽셀 단위 tolerance 유지, launch에서 align_x_threshold 주면 우선 적용
        align_x_threshold = rospy.get_param("~align_x_threshold", None)
        if align_x_threshold is not None:
            try:
                # 사용자가 px 단위로 넘기면 그대로 사용, 0~1 비율이면 화면폭 기준으로 변환
                ax = float(align_x_threshold)
                if 0.0 < abs(ax) < 1.0:
                    # color_cb에서 실제 frame width 기준으로 사용하기 위해 저장
                    self.center_tolerance_ratio = abs(ax)
                    self.center_tolerance_px = None
                else:
                    self.center_tolerance_ratio = None
                    self.center_tolerance_px = int(abs(ax))
            except Exception:
                self.center_tolerance_ratio = None
                self.center_tolerance_px = int(rospy.get_param("~center_tolerance_px", 40))
        else:
            self.center_tolerance_ratio = None
            self.center_tolerance_px = int(rospy.get_param("~center_tolerance_px", 40))

        self.search_timeout_sec = float(rospy.get_param("~search_timeout_sec", 2.0))
        self.control_hz = float(rospy.get_param("~control_hz", 10.0))

        # depth valid range
        self.max_valid_depth_m = float(rospy.get_param("~max_valid_depth_m", 5.0))
        self.min_valid_depth_m = float(rospy.get_param("~min_valid_depth_m", 0.05))

        # 기타 launch 호환 파라미터
        self.input_size = int(rospy.get_param("~input_size", 640))
        self.marker_timeout_sec = float(rospy.get_param("~marker_timeout_sec", 0.7))
        self.stop_burst_count = int(rospy.get_param("~stop_burst_count", 3))
        self.left_if_pos = bool(rospy.get_param("~left_if_pos", True))
        self.use_direct_pwm = bool(rospy.get_param("~use_direct_pwm", True))

        # PWM
        self.turn_pwm = int(rospy.get_param("~turn_pwm", 8))
        self.forward_pwm = int(rospy.get_param("~forward_pwm", 25))
        self.forward_slow_pwm = int(rospy.get_param("~forward_slow_pwm", 5))

        # launch에서 개별 pwm 주는 경우 우선 사용
        self.pwm_forward_l = int(rospy.get_param("~pwm_forward_l", self.forward_pwm))
        self.pwm_forward_r = int(rospy.get_param("~pwm_forward_r", self.forward_pwm))
        self.pwm_turn_left_l = int(rospy.get_param("~pwm_turn_left_l", -self.turn_pwm))
        self.pwm_turn_left_r = int(rospy.get_param("~pwm_turn_left_r", self.turn_pwm))
        self.pwm_turn_right_l = int(rospy.get_param("~pwm_turn_right_l", self.turn_pwm))
        self.pwm_turn_right_r = int(rospy.get_param("~pwm_turn_right_r", -self.turn_pwm))
        self.pwm_backward_l = int(rospy.get_param("~pwm_backward_l", 0))
        self.pwm_backward_r = int(rospy.get_param("~pwm_backward_r", 0))

        self.debug_image_topic = rospy.get_param("~debug_image_topic", "/panel/debug_image")
        self.mode_topic = rospy.get_param("~mode_topic", "/marker_pose_controller/mode")

        # =========================================================
        # Runtime State
        # =========================================================
        self.bridge = CvBridge()
        self.model = YOLO(self.model_path)
        self.class_names = self.model.names

        self.latest_depth = None
        self.latest_depth_stamp = None
        self.latest_color_stamp = None

        self.cam_info = None
        self.fx = None
        self.fy = None
        self.cx0 = None
        self.cy0 = None

        self.active_mission = 0
        self.enabled = False
        self.mode = "IDLE"
        self.last_detect_time = rospy.Time(0)

        self.last_control_time = rospy.Time(0)

        rospy.loginfo("==== marker_pose_controller_panel ====")
        rospy.loginfo("model_path           : %s", self.model_path)
        rospy.loginfo("target_class_name    : %s", self.target_class_name)
        rospy.loginfo("topic_mission        : %s", self.topic_mission)
        rospy.loginfo("topic_done           : %s", self.topic_done)
        rospy.loginfo("docking_done_topic   : %s", self.docking_done_topic)
        rospy.loginfo("debug_image_topic    : %s", self.debug_image_topic)
        rospy.loginfo("view_image           : %s", str(self.view_image))
        rospy.loginfo("publish_debug_image  : %s", str(self.publish_debug_image_enabled))
        rospy.loginfo("stop_distance_m      : %.3f", self.stop_distance_m)
        rospy.loginfo("conf_thresh          : %.3f", self.conf_thresh)
        rospy.loginfo("depth_patch          : %d", self.depth_patch)
        rospy.loginfo("control_hz           : %.1f", self.control_hz)
        rospy.loginfo("class_names          : %s", str(self.class_names))

        # =========================================================
        # ROS pub/sub
        # =========================================================
        self.pub_debug_image = rospy.Publisher(self.debug_image_topic, Image, queue_size=1)
        self.pub_mode = rospy.Publisher(self.mode_topic, String, queue_size=1)

        self.pub_doc_pwm = rospy.Publisher(self.doc_pwm_topic, Int16MultiArray, queue_size=1)
        self.pub_doc_cmd_char = rospy.Publisher(self.char_cmd_topic, String, queue_size=1)
        self.pub_done = rospy.Publisher(self.topic_done, Int32, queue_size=1, latch=True)
        self.pub_docking_done = rospy.Publisher(self.docking_done_topic, Bool, queue_size=1)

        rospy.Subscriber(self.topic_mission, Int32, self.mission_cb, queue_size=1)
        rospy.Subscriber(self.camera_info_topic, CameraInfo, self.camera_info_cb, queue_size=1)
        rospy.Subscriber(self.depth_topic, Image, self.depth_cb, queue_size=1, buff_size=2**24)
        rospy.Subscriber(self.rgb_topic, Image, self.color_cb, queue_size=1, buff_size=2**24)

        self.publish_mode("IDLE")

    # =========================================================
    # Callbacks
    # =========================================================
    def mission_cb(self, msg):
        self.active_mission = int(msg.data)

        if self.active_mission > 0:
            self.enabled = True
            self.last_detect_time = rospy.Time.now()
            self.publish_mode("APPROACH")
            rospy.loginfo("Panel mission started: %d", self.active_mission)
        else:
            self.enabled = False
            self.stop_robot()
            self.publish_mode("IDLE")
            rospy.loginfo("Panel mission reset: %d", self.active_mission)

    def camera_info_cb(self, msg):
        self.cam_info = msg
        if len(msg.K) >= 9:
            self.fx = float(msg.K[0])
            self.fy = float(msg.K[4])
            self.cx0 = float(msg.K[2])
            self.cy0 = float(msg.K[5])

    def depth_cb(self, msg):
        try:
            if msg.encoding in ["16UC1", "mono16", "passthrough", "32FC1"]:
                depth = self.bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
            else:
                rospy.logwarn_throttle(3.0, "Unsupported depth encoding: %s", msg.encoding)
                return
            self.latest_depth = depth
            self.latest_depth_stamp = msg.header.stamp
        except Exception as e:
            rospy.logerr_throttle(1.0, "depth_cb error: %s", str(e))

    def color_cb(self, msg):
        self.latest_color_stamp = msg.header.stamp

        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception as e:
            rospy.logerr_throttle(1.0, "color_cb bridge error: %s", str(e))
            return

        annotated = frame.copy()
        annotated = self.draw_center_lines(annotated)

        if not self.enabled:
            self.publish_debug_image(annotated, msg.header)
            return

        # 너무 자주 제어 안 하도록 제한
        now = rospy.Time.now()
        if (now - self.last_control_time).to_sec() < (1.0 / max(1.0, self.control_hz)):
            self.publish_debug_image(annotated, msg.header)
            return
        self.last_control_time = now

        target_det, candidates = self.detect_target(frame)
        annotated = self.draw_detections(annotated, candidates, target_det)

        h, w = frame.shape[:2]
        img_cx = w // 2
        img_cy = h // 2

        # tolerance 계산
        if self.center_tolerance_px is not None:
            center_tol_px = self.center_tolerance_px
        elif self.center_tolerance_ratio is not None:
            center_tol_px = int(max(1, abs(self.center_tolerance_ratio) * w))
        else:
            center_tol_px = 40

        if target_det is None:
            self.publish_mode("SEARCH")
            cv2.putText(annotated, "LEFT_PANEL NOT FOUND", (20, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            rospy.logwarn_throttle(1.0, "LEFT_PANEL NOT FOUND")
            self.stop_robot()
            self.publish_debug_image(annotated, msg.header)
            return

        self.last_detect_time = rospy.Time.now()
        self.publish_mode("APPROACH")

        bx = target_det["cx"]
        by = target_det["cy"]
        x1 = target_det["x1"]
        y1 = target_det["y1"]
        x2 = target_det["x2"]
        y2 = target_det["y2"]
        conf = target_det["conf"]
        cls_name = target_det["cls_name"]

        err_x = bx - img_cx
        err_y = by - img_cy
        depth_m = self.get_depth_m(bx, by)
        yaw_deg = self.estimate_yaw_deg(x1, y1, x2, y2)

        text = "TARGET {} conf={:.2f} x={} y={} z={:.3f}m yaw={:.1f} err=({}, {})".format(
            cls_name, conf, bx, by, depth_m, yaw_deg, err_x, err_y
        )
        cv2.putText(annotated, text, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 0, 0), 2)
        cv2.circle(annotated, (bx, by), 5, (255, 0, 255), -1)
        cv2.putText(annotated,
                    "x:{} y:{} z:{:.3f} yaw:{:.1f}".format(bx, by, depth_m, yaw_deg),
                    (x1, max(20, y1 - 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
        cv2.putText(annotated,
                    "tol_px:{}".format(center_tol_px),
                    (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 0), 2)

        rospy.loginfo_throttle(0.5, text)

        self.control_approach(err_x, depth_m, center_tol_px)

        self.publish_debug_image(annotated, msg.header)

        if self.view_image:
            cv2.imshow("marker_pose_controller_left_panel", annotated)
            cv2.waitKey(1)

    # =========================================================
    # Detection
    # =========================================================
    def detect_target(self, frame):
        try:
            results = self.model.predict(
                source=frame,
                conf=self.conf_thresh,
                imgsz=self.input_size,
                verbose=False
            )
        except Exception as e:
            rospy.logerr_throttle(1.0, "YOLO inference error: %s", str(e))
            return None, []

        target_det = None
        candidates = []

        if len(results) == 0:
            return None, []

        r = results[0]
        if r.boxes is None or len(r.boxes) == 0:
            return None, []

        boxes = r.boxes
        for i in range(len(boxes)):
            try:
                box_xyxy = boxes.xyxy[i].cpu().numpy().astype(int)
                conf = float(boxes.conf[i].cpu().numpy())
                cls_id = int(boxes.cls[i].cpu().numpy())
                cls_name = self.class_names.get(cls_id, str(cls_id))

                x1, y1, x2, y2 = box_xyxy
                cx = int((x1 + x2) * 0.5)
                cy = int((y1 + y2) * 0.5)
                area = max(0, x2 - x1) * max(0, y2 - y1)
                depth_m = self.get_depth_m(cx, cy)

                candidates.append({
                    "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                    "cx": cx, "cy": cy,
                    "conf": conf,
                    "cls_id": cls_id,
                    "cls_name": cls_name,
                    "area": area,
                    "depth_m": depth_m,
                })
            except Exception as e:
                rospy.logwarn_throttle(1.0, "Detection parse error: %s", str(e))

        target_candidates = [d for d in candidates if d["cls_name"] == self.target_class_name]
        if len(target_candidates) > 0:
            target_candidates = sorted(target_candidates, key=lambda d: d["area"], reverse=True)
            target_det = target_candidates[0]

        return target_det, candidates

    # =========================================================
    # Depth / Geometry
    # =========================================================
    def get_depth_m(self, u, v):
        if self.latest_depth is None:
            return -1.0

        depth = self.latest_depth
        h, w = depth.shape[:2]

        if u < 0 or u >= w or v < 0 or v >= h:
            return -1.0

        p = max(1, self.depth_patch)
        x1 = max(0, u - p)
        x2 = min(w, u + p + 1)
        y1 = max(0, v - p)
        y2 = min(h, v + p + 1)

        patch = depth[y1:y2, x1:x2]
        if patch.size == 0:
            return -1.0

        if patch.dtype == np.uint16:
            valid = patch[patch > 0]
            if len(valid) == 0:
                return -1.0
            depth_m = float(np.median(valid)) / 1000.0
        elif patch.dtype in [np.float32, np.float64]:
            valid = patch[np.isfinite(patch)]
            valid = valid[valid > 0.0]
            if len(valid) == 0:
                return -1.0
            depth_m = float(np.median(valid))
        else:
            return -1.0

        if depth_m < self.min_valid_depth_m or depth_m > self.max_valid_depth_m:
            return -1.0

        return depth_m

    def estimate_yaw_deg(self, x1, y1, x2, y2):
        w = max(1, x2 - x1)
        h = max(1, y2 - y1)
        ratio = float(w) / float(h)
        yaw_deg = (ratio - 1.0) * 35.0
        return max(-45.0, min(45.0, yaw_deg))

    # =========================================================
    # Control
    # =========================================================
    def control_approach(self, err_x, depth_m, center_tolerance_px):
        if depth_m <= 0.0:
            rospy.logwarn_throttle(1.0, "Invalid depth. Stop.")
            self.stop_robot()
            return

        if depth_m <= self.stop_distance_m:
            rospy.loginfo_throttle(1.0, "Target reached. depth=%.3fm", depth_m)
            self.stop_robot()
            self.publish_mode("DONE")
            self.pub_done.publish(Int32(data=self.active_mission))
            self.pub_docking_done.publish(Bool(data=True))
            self.enabled = False
            self.active_mission = 0
            return

        # 중앙 정렬 우선
        if err_x > center_tolerance_px:
            # 오른쪽에 보이면 우회전
            self.send_pwm(self.pwm_turn_right_l, self.pwm_turn_right_r)
            return
        elif err_x < -center_tolerance_px:
            # 왼쪽에 보이면 좌회전
            self.send_pwm(self.pwm_turn_left_l, self.pwm_turn_left_r)
            return

        # 중앙 정렬됐으면 전진
        if depth_m > 0.60:
            self.send_pwm(self.pwm_forward_l, self.pwm_forward_r)
        else:
            slow_l = int(np.sign(self.pwm_forward_l) * min(abs(self.pwm_forward_l), abs(self.forward_slow_pwm)))
            slow_r = int(np.sign(self.pwm_forward_r) * min(abs(self.pwm_forward_r), abs(self.forward_slow_pwm)))
            if slow_l == 0 and self.pwm_forward_l != 0:
                slow_l = 1 if self.pwm_forward_l > 0 else -1
            if slow_r == 0 and self.pwm_forward_r != 0:
                slow_r = 1 if self.pwm_forward_r > 0 else -1
            self.send_pwm(slow_l, slow_r)

    def send_pwm(self, left_pwm, right_pwm):
        msg = Int16MultiArray()
        msg.data = [int(left_pwm), int(right_pwm)]
        self.pub_doc_pwm.publish(msg)

    def stop_robot(self):
        for _ in range(max(1, self.stop_burst_count)):
            self.send_pwm(0, 0)
        self.pub_doc_cmd_char.publish(String(data="x"))

    # =========================================================
    # Drawing / Publish
    # =========================================================
    def draw_detections(self, img, candidates, target_det):
        for det in candidates:
            x1, y1, x2, y2 = det["x1"], det["y1"], det["x2"], det["y2"]
            cx, cy = det["cx"], det["cy"]
            cls_name = det["cls_name"]
            conf = det["conf"]
            depth_m = det.get("depth_m", -1.0)

            if target_det is not None and det is target_det:
                color = (0, 0, 255)
            elif cls_name == self.target_class_name:
                color = (0, 255, 0)
            else:
                color = (180, 180, 180)

            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            cv2.circle(img, (cx, cy), 4, color, -1)

            depth_text = "z={:.2f}m".format(depth_m) if depth_m > 0.0 else "z=N/A"
            label = "{} {:.2f} {}".format(cls_name, conf, depth_text)
            cv2.putText(img, label, (x1, max(20, y1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
        return img

    def draw_center_lines(self, img):
        h, w = img.shape[:2]
        cv2.line(img, (w // 2, 0), (w // 2, h), (255, 255, 0), 1)
        cv2.line(img, (0, h // 2), (w, h // 2), (255, 255, 0), 1)
        return img

    def publish_debug_image(self, annotated, header):
        # publish_debug_image:=false이면 디버그 이미지 변환/publish를 하지 않음.
        # 카메라 화면은 realsense-viewer/rqt_image_view 등 외부에서 이미 보고 있을 때 CPU 부하를 줄이기 위함.
        if not self.publish_debug_image_enabled:
            return

        try:
            out_msg = self.bridge.cv2_to_imgmsg(annotated, encoding="bgr8")
            out_msg.header = header
            self.pub_debug_image.publish(out_msg)
        except Exception as e:
            rospy.logerr_throttle(1.0, "debug image publish error: %s", str(e))

    def publish_mode(self, mode_str):
        self.mode = mode_str
        self.pub_mode.publish(String(data=mode_str))


if __name__ == "__main__":
    try:
        node = MarkerPoseControllerPanel()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
    finally:
        cv2.destroyAllWindows()
