#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy
import cv2
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from ultralytics import YOLO


class DepthYoloPanelTest:
    """
    Depth camera RGB topic에서 YOLO로 panel 검출만 확인하는 테스트 노드.
    - /camera/color/image_raw 같은 RGB 토픽만 사용
    - left_panel / right_panel bbox 표시
    - 화면 표시(cv2.imshow) + debug image publish
    """

    def __init__(self):
        rospy.init_node("depth_yolo_panel_test", anonymous=False)

        self.image_topic = rospy.get_param("~image_topic", "/camera/color/image_raw")
        self.model_path = rospy.get_param("~model_path", "/home/inwoong/panel_best.pt")
        self.conf_threshold = float(rospy.get_param("~conf_threshold", 0.25))
        self.input_size = int(rospy.get_param("~input_size", 1280))
        self.show_window = bool(rospy.get_param("~show_window", True))
        self.window_name = rospy.get_param("~window_name", "depth_yolo_panel_test")
        self.debug_topic = rospy.get_param("~debug_topic", "/panel_yolo_test/debug_image")

        self.bridge = CvBridge()
        self.model = YOLO(self.model_path)
        self.names = self.model.names

        rospy.loginfo("Loading YOLO model: %s", self.model_path)
        rospy.loginfo("Class names: %s", str(self.names))
        rospy.loginfo("Subscribing image topic: %s", self.image_topic)
        rospy.loginfo("conf_threshold=%.2f, input_size=%d", self.conf_threshold, self.input_size)

        self.debug_pub = rospy.Publisher(self.debug_topic, Image, queue_size=1)
        self.image_sub = rospy.Subscriber(self.image_topic, Image, self.image_callback, queue_size=1, buff_size=2**24)

        self.frame_count = 0
        self.last_log_time = rospy.Time.now().to_sec()

    def image_callback(self, msg):
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception as e:
            rospy.logwarn_throttle(1.0, "cv_bridge error: %s", str(e))
            return

        results = self.model.predict(
            source=frame,
            imgsz=self.input_size,
            conf=self.conf_threshold,
            verbose=False
        )

        debug = frame.copy()
        det_count = 0
        left_count = 0
        right_count = 0
        other_count = 0

        if results and len(results) > 0:
            result = results[0]
            if result.boxes is not None:
                for box in result.boxes:
                    det_count += 1
                    cls_id = int(box.cls[0].item())
                    conf = float(box.conf[0].item())
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int).tolist()
                    cls_name = str(self.names.get(cls_id, f"class_{cls_id}"))

                    if cls_name == "left_panel":
                        color = (0, 255, 0)
                        left_count += 1
                        prefix = "LEFT"
                    elif cls_name == "right_panel":
                        color = (0, 255, 255)
                        right_count += 1
                        prefix = "RIGHT"
                    else:
                        color = (0, 128, 255)
                        other_count += 1
                        prefix = "OTHER"

                    cv2.rectangle(debug, (x1, y1), (x2, y2), color, 2)
                    label = f"{prefix} {cls_name} {conf:.2f}"
                    cv2.putText(debug, label, (x1, max(25, y1 - 8)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)

                    cx = (x1 + x2) // 2
                    cy = (y1 + y2) // 2
                    cv2.circle(debug, (cx, cy), 4, color, -1)

        h, w = debug.shape[:2]
        cv2.line(debug, (w // 2, 0), (w // 2, h), (255, 255, 0), 1)
        cv2.line(debug, (0, h // 2), (w, h // 2), (255, 255, 0), 1)

        info_lines = [
            f"topic: {self.image_topic}",
            f"model: {self.model_path}",
            f"detections: {det_count}  left: {left_count}  right: {right_count}  other: {other_count}",
            f"conf: {self.conf_threshold:.2f}  imgsz: {self.input_size}",
        ]
        for i, text in enumerate(info_lines):
            cv2.putText(debug, text, (20, 35 + 28 * i),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        now = rospy.Time.now().to_sec()
        if now - self.last_log_time > 1.0:
            rospy.loginfo("detections=%d left=%d right=%d other=%d",
                          det_count, left_count, right_count, other_count)
            self.last_log_time = now

        try:
            self.debug_pub.publish(self.bridge.cv2_to_imgmsg(debug, encoding="bgr8"))
        except Exception as e:
            rospy.logwarn_throttle(1.0, "debug image publish error: %s", str(e))

        if self.show_window:
            cv2.imshow(self.window_name, debug)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                rospy.signal_shutdown("q pressed")

    def run(self):
        rospy.loginfo("depth_yolo_panel_test started.")
        rospy.spin()
        if self.show_window:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    try:
        node = DepthYoloPanelTest()
        node.run()
    except rospy.ROSInterruptException:
        pass
