#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import signal
import subprocess
import threading
import time
import math

import rospy
import actionlib

from std_srvs.srv import SetBool, SetBoolResponse, Empty
from geometry_msgs.msg import PoseWithCovarianceStamped, Quaternion
from move_base_msgs.msg import MoveBaseAction
from tf.transformations import quaternion_from_euler


class NavStackManager:
    def __init__(self):
        # =========================================================
        # Params
        # =========================================================
        self.map_a_file = rospy.get_param("~map_a_file", "/home/limhs/catkin_ws/src/my_map/map_1F.yaml")
        self.map_b_file = rospy.get_param("~map_b_file", "/home/limhs/catkin_ws/src/my_map/map_vcc.yaml")

        self.mapA_init = rospy.get_param("~mapA_initial", [0.0, 0.0, 0.0])
        self.mapB_init = rospy.get_param("~mapB_initial", [0.0, 0.0, 0.0])

        self.initialpose_topic = rospy.get_param("~initialpose_topic", "/initialpose")
        self.pose_cov_xy = float(rospy.get_param("~pose_cov_xy", 0.25))
        self.pose_cov_yaw = float(rospy.get_param("~pose_cov_yaw", 0.20))

        self.publish_init_on_start = bool(rospy.get_param("~publish_init_on_start", True))
        self.start_with_map_b = bool(rospy.get_param("~start_with_map_b", False))

        self.switch_settle_sec = float(rospy.get_param("~switch_settle_sec", 2.0))

        self.clear_costmap = bool(rospy.get_param("~clear_costmap", True))
        self.clear_srv_name = rospy.get_param("~clear_srv_name", "/move_base/clear_costmaps")
        self.post_clear_delay = float(rospy.get_param("~post_clear_delay", 1.0))
        self.double_clear = bool(rospy.get_param("~double_clear", True))

        # nav_stack.launch 실행 정보
        self.nav_launch_pkg = rospy.get_param("~nav_launch_pkg", "my_nav")
        self.nav_launch_file = rospy.get_param("~nav_launch_file", "nav_stack.launch")
        self.nav_record = bool(rospy.get_param("~nav_record", False))

        # 재기동/대기 관련
        self.shutdown_timeout = float(rospy.get_param("~shutdown_timeout", 8.0))
        self.startup_timeout = float(rospy.get_param("~startup_timeout", 20.0))
        self.amcl_wait_timeout = float(rospy.get_param("~amcl_wait_timeout", 10.0))

        self.move_base_action_name = rospy.get_param("~move_base_action_name", "move_base")
        self.amcl_pose_topic = rospy.get_param("~amcl_pose_topic", "/amcl_pose")

        # =========================================================
        # Runtime state
        # =========================================================
        self.current_map_is_b = self.start_with_map_b
        self.nav_proc = None
        self.nav_proc_lock = threading.Lock()

        self.pub_initialpose = rospy.Publisher(
            self.initialpose_topic,
            PoseWithCovarianceStamped,
            queue_size=1
        )

        self.srv_switch = rospy.Service("~switch_map", SetBool, self.handle_switch)

        rospy.loginfo("[map_mux] mode changed: nav stack manager")
        rospy.loginfo("[map_mux] map A = %s", self.map_a_file)
        rospy.loginfo("[map_mux] map B = %s", self.map_b_file)
        rospy.loginfo("[map_mux] switch service = %s", rospy.resolve_name("~switch_map"))

        # startup
        rospy.sleep(0.5)
        self._startup_nav_stack()

    # =========================================================
    # Startup
    # =========================================================
    def _startup_nav_stack(self):
        target_is_b = self.start_with_map_b
        ok, msg = self._restart_nav_stack(target_is_b=target_is_b, publish_init=self.publish_init_on_start)
        if ok:
            rospy.loginfo("[map_mux] startup complete: %s", msg)
        else:
            rospy.logerr("[map_mux] startup failed: %s", msg)

    # =========================================================
    # Utils
    # =========================================================
    def _target_map_file(self, target_is_b: bool) -> str:
        return self.map_b_file if target_is_b else self.map_a_file

    def _target_init_pose(self, target_is_b: bool):
        return self.mapB_init if target_is_b else self.mapA_init

    def _publish_initialpose(self, x, y, yaw):
        msg = PoseWithCovarianceStamped()
        msg.header.stamp = rospy.Time.now()
        msg.header.frame_id = "map"

        msg.pose.pose.position.x = float(x)
        msg.pose.pose.position.y = float(y)
        msg.pose.pose.position.z = 0.0

        q = quaternion_from_euler(0.0, 0.0, float(yaw))
        msg.pose.pose.orientation = Quaternion(*q)

        cov = [0.0] * 36
        cov[0] = float(self.pose_cov_xy)
        cov[7] = float(self.pose_cov_xy)
        cov[35] = float(self.pose_cov_yaw)
        msg.pose.covariance = cov

        # 초기 pose는 한 번보다 여러 번 보내는 편이 안정적
        for _ in range(3):
            msg.header.stamp = rospy.Time.now()
            self.pub_initialpose.publish(msg)
            rospy.sleep(0.1)

        rospy.loginfo("[map_mux] initialpose published: x=%.3f y=%.3f yaw=%.3f", x, y, yaw)

    def _wait_for_move_base(self, timeout_sec: float) -> bool:
        try:
            ac = actionlib.SimpleActionClient(self.move_base_action_name, MoveBaseAction)
            ok = ac.wait_for_server(rospy.Duration(timeout_sec))
            if ok:
                rospy.loginfo("[map_mux] move_base action ready")
            else:
                rospy.logwarn("[map_mux] move_base action not ready within %.1fs", timeout_sec)
            return ok
        except Exception as e:
            rospy.logwarn("[map_mux] wait_for_move_base error: %s", e)
            return False

    def _wait_for_amcl_pose(self, timeout_sec: float) -> bool:
        try:
            rospy.wait_for_message(self.amcl_pose_topic, PoseWithCovarianceStamped, timeout=timeout_sec)
            rospy.loginfo("[map_mux] amcl_pose received")
            return True
        except Exception as e:
            rospy.logwarn("[map_mux] amcl_pose wait timeout/error: %s", e)
            return False

    def _call_clear_costmaps(self):
        if not self.clear_costmap:
            return

        try:
            rospy.wait_for_service(self.clear_srv_name, timeout=3.0)
            srv = rospy.ServiceProxy(self.clear_srv_name, Empty)
            srv()
            rospy.loginfo("[map_mux] clear_costmaps called: %s", self.clear_srv_name)
        except Exception as e:
            rospy.logwarn("[map_mux] clear_costmaps failed: %s", e)

    def _build_launch_cmd(self, map_file: str, target_is_b: bool):
        use_prohibition = "true" if target_is_b else "false"

        cmd = [
            "roslaunch",
            self.nav_launch_pkg,
            self.nav_launch_file,
            "map_file:={}".format(map_file),
            "record:={}".format("true" if self.nav_record else "false"),
            "use_prohibition:={}".format(use_prohibition),
        ]
        return cmd

    def _start_nav_stack(self, map_file: str, target_is_b: bool):
        cmd = self._build_launch_cmd(map_file, target_is_b)

        rospy.loginfo("[map_mux] starting nav stack: %s", " ".join(cmd))

        # 새 세션으로 띄워서 killpg 가능하게 함
        proc = subprocess.Popen(
            cmd,
            preexec_fn=os.setsid,
            stdout=None,
            stderr=None
        )
        return proc

    def _stop_nav_stack(self):
        with self.nav_proc_lock:
            proc = self.nav_proc
            self.nav_proc = None

        if proc is None:
            return True

        try:
            pgid = os.getpgid(proc.pid)
        except Exception:
            pgid = None

        try:
            if pgid is not None:
                rospy.loginfo("[map_mux] stopping nav stack pgid=%s", str(pgid))
                os.killpg(pgid, signal.SIGINT)
            else:
                rospy.loginfo("[map_mux] stopping nav stack pid=%s", str(proc.pid))
                proc.send_signal(signal.SIGINT)
        except Exception as e:
            rospy.logwarn("[map_mux] SIGINT stop failed: %s", e)

        t0 = time.time()
        while time.time() - t0 < self.shutdown_timeout:
            rc = proc.poll()
            if rc is not None:
                rospy.loginfo("[map_mux] nav stack exited with code %s", str(rc))
                return True
            rospy.sleep(0.1)

        rospy.logwarn("[map_mux] nav stack did not exit after %.1fs -> SIGTERM", self.shutdown_timeout)

        try:
            if pgid is not None:
                os.killpg(pgid, signal.SIGTERM)
            else:
                proc.terminate()
        except Exception as e:
            rospy.logwarn("[map_mux] SIGTERM failed: %s", e)

        rospy.sleep(1.0)

        rc = proc.poll()
        if rc is not None:
            rospy.loginfo("[map_mux] nav stack terminated with code %s", str(rc))
            return True

        rospy.logwarn("[map_mux] nav stack still alive -> SIGKILL")
        try:
            if pgid is not None:
                os.killpg(pgid, signal.SIGKILL)
            else:
                proc.kill()
        except Exception as e:
            rospy.logwarn("[map_mux] SIGKILL failed: %s", e)

        rospy.sleep(0.5)
        return proc.poll() is not None

    def _restart_nav_stack(self, target_is_b: bool, publish_init: bool = True):
        target_map_file = self._target_map_file(target_is_b)
        x, y, yaw = self._target_init_pose(target_is_b)

        if not os.path.isfile(target_map_file):
            return False, "map file not found: {}".format(target_map_file)

        # 1) 기존 nav stack 종료
        self._stop_nav_stack()

        rospy.sleep(0.5)

        # 2) 새 nav stack 시작
        proc = self._start_nav_stack(target_map_file, target_is_b)
        with self.nav_proc_lock:
            self.nav_proc = proc

        # 3) move_base / amcl 준비 대기
        move_base_ok = self._wait_for_move_base(self.startup_timeout)
        amcl_ok = self._wait_for_amcl_pose(self.amcl_wait_timeout)

        if publish_init:
            self._publish_initialpose(x, y, yaw)

        # 4) 안정화 시간
        if self.switch_settle_sec > 0.0:
            rospy.sleep(self.switch_settle_sec)

        # 5) costmap clear
        if self.clear_costmap:
            self._call_clear_costmaps()
            if self.double_clear and self.post_clear_delay > 0.0:
                rospy.sleep(self.post_clear_delay)
                self._call_clear_costmaps()

        self.current_map_is_b = bool(target_is_b)

        which = "B" if target_is_b else "A"
        msg = "nav stack relaunched with map {} ({:.3f},{:.3f},{:.3f}rad), move_base_ok={}, amcl_ok={}".format(
            which, x, y, yaw, move_base_ok, amcl_ok
        )
        return True, msg

    # =========================================================
    # Service
    # =========================================================
    def handle_switch(self, req):
        target_is_b = bool(req.data)

        if target_is_b == self.current_map_is_b:
            which = "B" if target_is_b else "A"
            return SetBoolResponse(True, "already on map {}".format(which))

        rospy.loginfo("[map_mux] switch_map requested -> %s", "B" if target_is_b else "A")

        ok, msg = self._restart_nav_stack(target_is_b=target_is_b, publish_init=True)
        return SetBoolResponse(ok, msg)

    # =========================================================
    # Shutdown
    # =========================================================
    def shutdown(self):
        rospy.loginfo("[map_mux] shutdown requested")
        self._stop_nav_stack()


if __name__ == "__main__":
    rospy.init_node("map_mux")
    manager = NavStackManager()
    rospy.on_shutdown(manager.shutdown)
    rospy.spin()
