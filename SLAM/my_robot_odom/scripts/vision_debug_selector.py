#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy
from std_msgs.msg import String, Float64, Int32
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2


class VisionDebugSelector:
    def __init__(self):
        rospy.init_node("vision_debug_selector")

        self.mode_topic = rospy.get_param("~mode_topic", "/vision_debug_mode")
        self.marker_image_topic = rospy.get_param("~marker_image_topic", "/usb_cam/image_raw")
        self.panel_image_topic = rospy.get_param("~panel_image_topic", "/panel/debug_image")
        self.output_image_topic = rospy.get_param("~output_image_topic", "/vision_debug/image")

        self.publish_hz = float(rospy.get_param("~publish_hz", 15.0))

        self.bridge = CvBridge()

        self.current_mode = "off"

        self.last_marker_msg = None
        self.last_panel_msg = None

        # ArUco state
        self.aruco_id = None
        self.aruco_x = None
        self.aruco_z = None
        self.aruco_yaw = None

        self.pub = rospy.Publisher(self.output_image_topic, Image, queue_size=1)

        rospy.Subscriber(self.mode_topic, String, self._mode_cb, queue_size=1)
        rospy.Subscriber(self.marker_image_topic, Image, self._marker_img_cb, queue_size=1)
        rospy.Subscriber(self.panel_image_topic, Image, self._panel_img_cb, queue_size=1)

        rospy.Subscriber("/aruco/marker_id", Int32, self._aruco_id_cb, queue_size=1)
        rospy.Subscriber("/aruco/pose_x", Float64, self._aruco_x_cb, queue_size=1)
        rospy.Subscriber("/aruco/pose_z", Float64, self._aruco_z_cb, queue_size=1)
        rospy.Subscriber("/aruco/yaw_b_m", Float64, self._aruco_yaw_cb, queue_size=1)

        self.timer = rospy.Timer(rospy.Duration(1.0 / self.publish_hz), self._timer_cb)

        rospy.loginfo("[vision_debug_selector] ready")
        rospy.spin()

    def _mode_cb(self, msg):
        self.current_mode = str(msg.data).strip().lower()

    def _marker_img_cb(self, msg):
        self.last_marker_msg = msg

    def _panel_img_cb(self, msg):
        self.last_panel_msg = msg

    def _aruco_id_cb(self, msg):
        self.aruco_id = int(msg.data)

    def _aruco_x_cb(self, msg):
        self.aruco_x = float(msg.data)

    def _aruco_z_cb(self, msg):
        self.aruco_z = float(msg.data)

    def _aruco_yaw_cb(self, msg):
        self.aruco_yaw = float(msg.data)

    def _overlay_aruco_info(self, src_msg):
        try:
            img = self.bridge.imgmsg_to_cv2(src_msg, "bgr8")

            lines = [
                "ARUCO INFO",
                f"ID: {self.aruco_id}",
                f"pose_x: {self.aruco_x:.3f}" if self.aruco_x is not None else "pose_x: -",
                f"pose_z: {self.aruco_z:.3f}" if self.aruco_z is not None else "pose_z: -",
                f"yaw: {self.aruco_yaw:.1f}" if self.aruco_yaw is not None else "yaw: -",
            ]

            y = 30
            for line in lines:
                cv2.putText(img, line, (20, y), cv2.FONT_HERSHEY_SIMPLEX,
                            0.7, (0, 255, 255), 2)
                y += 30

            return self.bridge.cv2_to_imgmsg(img, "bgr8")

        except Exception:
            return src_msg

    def _timer_cb(self, _event):
        if self.current_mode == "marker" and self.last_marker_msg:
            msg = self._overlay_aruco_info(self.last_marker_msg)
        elif self.current_mode == "panel" and self.last_panel_msg:
            msg = self.last_panel_msg
        else:
            return

        self.pub.publish(msg)


if __name__ == "__main__":
    VisionDebugSelector()
