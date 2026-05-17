#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import rospy
import cv2
import numpy as np

from ultralytics import YOLO
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from std_msgs.msg import Header, String, Float32MultiArray, MultiArrayDimension


class PanelDetectorNode:
    def __init__(self):
        rospy.init_node("panel_detector_node", anonymous=False)

        # =========================
        # Params
        # =========================
        default_model = os.path.expanduser("/home/inwoong/panel_best.pt")
        self.model_path = rospy.get_param("~model_path", default_model)

        self.color_topic = rospy.get_param("~color_topic", "/camera/color/image_raw")
        self.depth_topic = rospy.get_param("~depth_topic", "/camera/aligned_depth_to_color/image_raw")

        self.use_depth = rospy.get_param("~use_depth", True)
        self.view_image = rospy.get_param("~view_image", True)
        self.conf_thresh = rospy.get_param("~conf_thresh", 0.4)

        # "", "left_panel", "right_panel"
        self.target_class = rospy.get_param("~target_class", "")

        # bbox 중심 depth를 가져올 때 패치 크기
        self.depth_patch = int(rospy.get_param("~depth_patch", 5))

        # 시각화 / 출력
        self.annotated_topic = rospy.get_param("~annotated_topic", "/panel_detection/image_annotated")
        self.result_topic = rospy.get_param("~result_topic", "/panel_detection/target")
        self.result_text_topic = rospy.get_param("~result_text_topic", "/panel_detection/result_text")

        # =========================
        # Internal
        # =========================
        self.bridge = CvBridge()
        self.model = YOLO(self.model_path)

        self.latest_depth = None
        self.latest_depth_stamp = None

        self.class_names = self.model.names  # {0:'left_panel', 1:'right_panel'} 형태 예상

        rospy.loginfo("==== Panel Detector Node ====")
        rospy.loginfo("model_path   : %s", self.model_path)
        rospy.loginfo("color_topic  : %s", self.color_topic)
        rospy.loginfo("depth_topic  : %s", self.depth_topic)
        rospy.loginfo("use_depth    : %s", str(self.use_depth))
        rospy.loginfo("target_class : %s", self.target_class if self.target_class else "ALL")
        rospy.loginfo("class_names  : %s", str(self.class_names))

        # =========================
        # ROS pubs/subs
        # =========================
        self.pub_annotated = rospy.Publisher(self.annotated_topic, Image, queue_size=1)
        self.pub_result = rospy.Publisher(self.result_topic, Float32MultiArray, queue_size=1)
        self.pub_result_text = rospy.Publisher(self.result_text_topic, String, queue_size=1)

        if self.use_depth:
            rospy.Subscriber(self.depth_topic, Image, self.depth_cb, queue_size=1, buff_size=2**24)

        rospy.Subscriber(self.color_topic, Image, self.color_cb, queue_size=1, buff_size=2**24)

    # =========================
    # Depth callback
    # =========================
    def depth_cb(self, msg):
        try:
            if msg.encoding in ["16UC1", "mono16"]:
                depth = self.bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
            elif msg.encoding == "32FC1":
                depth = self.bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
            else:
                rospy.logwarn_throttle(3.0, "Unsupported depth encoding: %s", msg.encoding)
                return

            self.latest_depth = depth
            self.latest_depth_stamp = msg.header.stamp

        except Exception as e:
            rospy.logerr_throttle(1.0, "depth_cb error: %s", str(e))

    # =========================
    # Main color callback
    # =========================
    def color_cb(self, msg):
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception as e:
            rospy.logerr_throttle(1.0, "color_cb bridge error: %s", str(e))
            return

        try:
            results = self.model.predict(
                source=frame,
                conf=self.conf_thresh,
                verbose=False
            )
        except Exception as e:
            rospy.logerr_throttle(1.0, "YOLO inference error: %s", str(e))
            return

        annotated = frame.copy()
        target_det = None  # 가장 사용할 detection 1개
        det_texts = []

        if len(results) > 0:
            r = results[0]

            if r.boxes is not None and len(r.boxes) > 0:
                boxes = r.boxes

                candidates = []

                for i in range(len(boxes)):
                    box_xyxy = boxes.xyxy[i].cpu().numpy().astype(int)
                    conf = float(boxes.conf[i].cpu().numpy())
                    cls_id = int(boxes.cls[i].cpu().numpy())
                    cls_name = self.class_names.get(cls_id, str(cls_id))

                    x1, y1, x2, y2 = box_xyxy
                    cx = int((x1 + x2) * 0.5)
                    cy = int((y1 + y2) * 0.5)
                    area = max(0, x2 - x1) * max(0, y2 - y1)

                    # target_class 필터
                    if self.target_class and cls_name != self.target_class:
                        continue

                    candidates.append({
                        "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                        "cx": cx, "cy": cy,
                        "conf": conf,
                        "cls_id": cls_id,
                        "cls_name": cls_name,
                        "area": area
                    })

                # 후보 중 가장 큰 박스 선택
                if len(candidates) > 0:
                    candidates = sorted(candidates, key=lambda d: d["area"], reverse=True)
                    target_det = candidates[0]

                # 모든 박스 시각화
                for det in candidates:
                    x1, y1, x2, y2 = det["x1"], det["y1"], det["x2"], det["y2"]
                    cx, cy = det["cx"], det["cy"]
                    cls_name = det["cls_name"]
                    conf = det["conf"]

                    color = (0, 255, 0) if det is not target_det else (0, 0, 255)

                    cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
                    cv2.circle(annotated, (cx, cy), 4, color, -1)

                    label = "{} {:.2f}".format(cls_name, conf)
                    cv2.putText(
                        annotated, label, (x1, max(20, y1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2
                    )

                    det_texts.append(
                        "{} conf={:.2f} bbox=({}, {}, {}, {}) center=({}, {})".format(
                            cls_name, conf, x1, y1, x2, y2, cx, cy
                        )
                    )

        # 가장 사용할 detection 결과 publish
        result_msg = Float32MultiArray()
        result_text_msg = String()

        if target_det is not None:
            cx = target_det["cx"]
            cy = target_det["cy"]
            cls_id = target_det["cls_id"]
            conf = target_det["conf"]

            depth_m = self.get_depth_m(cx, cy)

            # 이미지 중심과의 오차
            h, w = frame.shape[:2]
            err_x = cx - (w // 2)
            err_y = cy - (h // 2)

            # [u, v, depth_m, err_x, err_y, class_id, conf, x1, y1, x2, y2]
            result_msg.layout.dim = [MultiArrayDimension(label="panel_result", size=11, stride=11)]
            result_msg.data = [
                float(cx), float(cy),
                float(depth_m),
                float(err_x), float(err_y),
                float(cls_id), float(conf),
                float(target_det["x1"]), float(target_det["y1"]),
                float(target_det["x2"]), float(target_det["y2"])
            ]
            self.pub_result.publish(result_msg)

            result_text = (
                "TARGET {} conf={:.2f} center=({}, {}) depth={:.3f}m err=({}, {})".format(
                    target_det["cls_name"], conf, cx, cy, depth_m, err_x, err_y
                )
            )
            result_text_msg.data = result_text
            self.pub_result_text.publish(result_text_msg)

            cv2.putText(
                annotated, result_text, (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 0, 0), 2
            )

            rospy.loginfo_throttle(0.5, result_text)

        else:
            result_msg.layout.dim = [MultiArrayDimension(label="panel_result", size=11, stride=11)]
            result_msg.data = [-1.0] * 11
            self.pub_result.publish(result_msg)

            result_text_msg.data = "TARGET NOT FOUND"
            self.pub_result_text.publish(result_text_msg)

            rospy.logwarn_throttle(1.0, "TARGET NOT FOUND")

        # 이미지 중앙선 표시
        h, w = annotated.shape[:2]
        cv2.line(annotated, (w // 2, 0), (w // 2, h), (255, 255, 0), 1)
        cv2.line(annotated, (0, h // 2), (w, h // 2), (255, 255, 0), 1)

        # 이미지 publish
        try:
            out_msg = self.bridge.cv2_to_imgmsg(annotated, encoding="bgr8")
            out_msg.header = msg.header
            self.pub_annotated.publish(out_msg)
        except Exception as e:
            rospy.logerr_throttle(1.0, "Annotated image publish error: %s", str(e))

        # 화면 보기
        if self.view_image:
            cv2.imshow("panel_detector", annotated)
            cv2.waitKey(1)

    # =========================
    # Depth utility
    # =========================
    def get_depth_m(self, u, v):
        if not self.use_depth or self.latest_depth is None:
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

        # 16UC1(mm) / 32FC1(m) 둘 다 처리
        if patch.dtype == np.uint16:
            valid = patch[(patch > 0)]
            if len(valid) == 0:
                return -1.0
            return float(np.median(valid)) / 1000.0

        elif patch.dtype == np.float32 or patch.dtype == np.float64:
            valid = patch[np.isfinite(patch)]
            valid = valid[valid > 0.0]
            if len(valid) == 0:
                return -1.0
            return float(np.median(valid))

        return -1.0


if __name__ == "__main__":
    try:
        node = PanelDetectorNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
    finally:
        cv2.destroyAllWindows()
