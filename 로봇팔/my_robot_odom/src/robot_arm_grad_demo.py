#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
robot_arm_grad_demo.py

로봇팔 미션 관리자 노드.

기능:
  - launch 실행 시 항상 대기
  - /arm_mission/panel 서비스 호출 시 yolo_detect_4_ros.py 실행
  - /arm_mission/button 서비스 호출 시 yolo_detect_1.py 실행
  - 각 YOLO 노드가 뜬 뒤 /move_target_label에 목표 라벨 자동 publish
    * panel  -> up
    * button -> f3
  - /arm_mission/stop 서비스로 실행 중인 미션 종료
  - /arm_mission/status 서비스로 상태 확인

기본 실행 대상:
  panel  : rosrun my_robot_odom yolo_detect_4_ros.py
  button : rosrun my_robot_odom yolo_detect_1.py

기본 목표 라벨:
  panel  : up
  button : f3
"""

import os
import signal
import subprocess
import threading
import time

import rospy
from std_msgs.msg import String, Bool
from std_srvs.srv import Trigger, TriggerResponse


class RobotArmMissionManager(object):
    ST_IDLE = "IDLE"
    ST_PANEL_RUNNING = "PANEL_RUNNING"
    ST_BUTTON_RUNNING = "BUTTON_RUNNING"
    ST_STOPPING = "STOPPING"
    ST_ERROR = "ERROR"

    def __init__(self):
        rospy.init_node("robot_arm_grad_demo", anonymous=False)

        # =========================
        # Params
        # =========================
        self.panel_pkg = rospy.get_param("~panel_pkg", "my_robot_odom")
        self.panel_script = rospy.get_param("~panel_script", "yolo_detect_4_ros.py")
        self.panel_args = rospy.get_param("~panel_args", "")

        self.button_pkg = rospy.get_param("~button_pkg", "my_robot_odom")
        self.button_script = rospy.get_param("~button_script", "yolo_detect_1.py")
        self.button_args = rospy.get_param("~button_args", "")

        # /arm_mission/button 호출 시 yolo_detect_1.py 실행 전에 수행할 시작 자세 이동 sh
        self.button_start_pose_sh = rospy.get_param("~button_start_pose_sh", "/home/inwoong/move_start_pose_right.sh")
        self.button_start_pose_wait = float(rospy.get_param("~button_start_pose_wait", 1.0))

        # yolo_detect_4_ros.py / yolo_detect_1.py가 구독하는 목표 라벨 토픽
        self.target_label_topic = rospy.get_param("~target_label_topic", "/move_target_label")
        self.panel_target_label = rospy.get_param("~panel_target_label", "up")
        self.button_target_label = rospy.get_param("~button_target_label", "f3")

        # 프로세스가 뜬 뒤 YOLO/MoveIt 초기화 시간을 조금 기다린 후 publish
        self.label_publish_delay = float(rospy.get_param("~label_publish_delay", 6.0))
        self.label_publish_repeat = int(rospy.get_param("~label_publish_repeat", 1))
        self.label_publish_interval = float(rospy.get_param("~label_publish_interval", 0.3))

        self.auto_restart_if_crashed = bool(rospy.get_param("~auto_restart_if_crashed", False))
        self.shutdown_timeout = float(rospy.get_param("~shutdown_timeout", 5.0))

        self.state_topic = rospy.get_param("~state_topic", "/robot_arm/state")
        self.running_topic = rospy.get_param("~running_topic", "/robot_arm/running")
        self.event_topic = rospy.get_param("~event_topic", "/robot_arm/event")

        # =========================
        # Runtime
        # =========================
        self.lock = threading.Lock()
        self.state = self.ST_IDLE
        self.current_mission = "none"
        self.proc = None
        self.proc_name = None
        self.stop_requested = False

        # =========================
        # Pub & Services
        # =========================
        self.pub_state = rospy.Publisher(self.state_topic, String, queue_size=1, latch=True)
        self.pub_running = rospy.Publisher(self.running_topic, Bool, queue_size=1, latch=True)
        self.pub_event = rospy.Publisher(self.event_topic, String, queue_size=10)
        self.pub_target_label = rospy.Publisher(self.target_label_topic, String, queue_size=10)

        self.srv_panel = rospy.Service("/arm_mission/panel", Trigger, self._srv_panel)
        self.srv_button = rospy.Service("/arm_mission/button", Trigger, self._srv_button)
        self.srv_stop = rospy.Service("/arm_mission/stop", Trigger, self._srv_stop)
        self.srv_status = rospy.Service("/arm_mission/status", Trigger, self._srv_status)
        self.srv_restart_panel = rospy.Service("/arm_mission/restart_panel", Trigger, self._srv_restart_panel)
        self.srv_restart_button = rospy.Service("/arm_mission/restart_button", Trigger, self._srv_restart_button)

        rospy.on_shutdown(self._on_shutdown)

        self._publish_state("manager started")
        rospy.loginfo("[robot_arm_manager] ready")
        rospy.loginfo("[robot_arm_manager] panel command: rosrun %s %s %s",
                      self.panel_pkg, self.panel_script, self.panel_args)
        rospy.loginfo("[robot_arm_manager] button command: rosrun %s %s %s",
                      self.button_pkg, self.button_script, self.button_args)
        rospy.loginfo("[robot_arm_manager] button start pose sh=%s wait=%.2fs",
                      self.button_start_pose_sh, self.button_start_pose_wait)
        rospy.loginfo("[robot_arm_manager] target label topic=%s panel_label=%s button_label=%s",
                      self.target_label_topic, self.panel_target_label, self.button_target_label)

    # =========================================================
    # State
    # =========================================================
    def _set_state(self, state, event_msg=""):
        self.state = state
        running = self.proc is not None and self.proc.poll() is None
        self.pub_state.publish(String(data=self.state))
        self.pub_running.publish(Bool(data=running))

        if event_msg:
            self.pub_event.publish(String(data=event_msg))
            rospy.loginfo("[robot_arm_manager] %s | state=%s running=%s", event_msg, self.state, str(running))
        else:
            rospy.loginfo("[robot_arm_manager] state=%s running=%s", self.state, str(running))

    def _publish_state(self, event_msg=""):
        running = self.proc is not None and self.proc.poll() is None
        self.pub_state.publish(String(data=self.state))
        self.pub_running.publish(Bool(data=running))
        if event_msg:
            self.pub_event.publish(String(data=event_msg))

    def _is_process_running(self):
        return self.proc is not None and self.proc.poll() is None

    def _state_for_mission(self, mission_name):
        if mission_name == "button":
            return self.ST_BUTTON_RUNNING
        return self.ST_PANEL_RUNNING

    def _config_for_mission(self, mission_name):
        if mission_name == "button":
            return self.button_pkg, self.button_script, self.button_args, self.button_target_label
        return self.panel_pkg, self.panel_script, self.panel_args, self.panel_target_label

    # =========================================================
    # Process Control
    # =========================================================
    def _build_rosrun_cmd(self, pkg, script, args_text=""):
        cmd = ["rosrun", pkg, script]
        if args_text:
            cmd.extend(str(args_text).split())
        return cmd

    def _start_process(self, mission_name, pkg, script, args_text=""):
        with self.lock:
            if self._is_process_running():
                msg = "already running: %s" % self.proc_name
                rospy.logwarn("[robot_arm_manager] %s", msg)
                return False, msg

            cmd = self._build_rosrun_cmd(pkg, script, args_text)

            try:
                rospy.loginfo("[robot_arm_manager] start process: %s", " ".join(cmd))

                self.proc = subprocess.Popen(
                    cmd,
                    stdout=None,
                    stderr=None,
                    preexec_fn=os.setsid
                )

                self.proc_name = mission_name
                self.current_mission = mission_name
                self.stop_requested = False
                self._set_state(self._state_for_mission(mission_name), "%s started" % mission_name)

                threading.Thread(
                    target=self._watch_process,
                    args=(self.proc, mission_name),
                    daemon=True
                ).start()

                return True, "%s started" % mission_name

            except Exception as e:
                self.proc = None
                self.proc_name = None
                self.current_mission = "none"
                self._set_state(self.ST_ERROR, "failed to start %s: %s" % (mission_name, str(e)))
                return False, str(e)

    def _stop_process(self, reason="stop requested"):
        with self.lock:
            if not self._is_process_running():
                self.proc = None
                self.proc_name = None
                self.current_mission = "none"
                self._set_state(self.ST_IDLE, "no running process")
                return True, "no running process"

            proc = self.proc
            name = self.proc_name or "unknown"
            self.stop_requested = True
            self._set_state(self.ST_STOPPING, "%s: %s" % (name, reason))

        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGINT)

            t0 = time.time()
            while time.time() - t0 < self.shutdown_timeout:
                if proc.poll() is not None:
                    break
                time.sleep(0.1)

            if proc.poll() is None:
                rospy.logwarn("[robot_arm_manager] SIGINT timeout. send SIGTERM")
                os.killpg(os.getpgid(proc.pid), signal.SIGTERM)

                t1 = time.time()
                while time.time() - t1 < 2.0:
                    if proc.poll() is not None:
                        break
                    time.sleep(0.1)

            if proc.poll() is None:
                rospy.logwarn("[robot_arm_manager] SIGTERM timeout. send SIGKILL")
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)

            with self.lock:
                self.proc = None
                self.proc_name = None
                self.current_mission = "none"
                self._set_state(self.ST_IDLE, "%s stopped" % name)

            return True, "%s stopped" % name

        except Exception as e:
            with self.lock:
                self.proc = None
                self.proc_name = None
                self.current_mission = "none"
                self._set_state(self.ST_ERROR, "stop error: %s" % str(e))
            return False, str(e)

    def _watch_process(self, proc, mission_name):
        ret = proc.wait()

        with self.lock:
            if self.proc is not proc:
                return

            self.proc = None
            self.proc_name = None
            self.current_mission = "none"

            if self.stop_requested:
                self._set_state(self.ST_IDLE, "%s exited after stop request, code=%s" % (mission_name, str(ret)))
                return

            if ret == 0:
                self._set_state(self.ST_IDLE, "%s finished normally" % mission_name)
            else:
                self._set_state(self.ST_ERROR, "%s crashed/exited, code=%s" % (mission_name, str(ret)))

        if self.auto_restart_if_crashed and ret != 0 and not rospy.is_shutdown():
            rospy.logwarn("[robot_arm_manager] auto restart enabled. restarting %s", mission_name)
            time.sleep(1.0)
            pkg, script, args_text, _label = self._config_for_mission(mission_name)
            self._start_process(mission_name, pkg, script, args_text)

    # =========================================================
    # Target label publishing
    # =========================================================
    def _publish_target_label_later(self, label):
        def worker():
            rospy.sleep(self.label_publish_delay)
            for i in range(max(1, self.label_publish_repeat)):
                if rospy.is_shutdown():
                    return
                if not self._is_process_running():
                    rospy.logwarn("[robot_arm_manager] target label publish skipped: process not running")
                    return
                self.pub_target_label.publish(String(data=str(label)))
                rospy.loginfo("[robot_arm_manager] publish target label: %s (%d/%d)",
                              str(label), i + 1, max(1, self.label_publish_repeat))
                rospy.sleep(self.label_publish_interval)

        threading.Thread(target=worker, daemon=True).start()

    # =========================================================
    # Button start pose
    # =========================================================
    def _run_button_start_pose_sh(self):
        """
        /arm_mission/button 호출 시 yolo_detect_1.py 실행 전에
        /home/inwoong/move_start_pose_right.sh를 실행하여 로봇팔을 버튼 탐색 시작 자세로 이동시킨다.
        """
        sh_path = str(self.button_start_pose_sh)

        if not sh_path:
            msg = "button_start_pose_sh is empty"
            rospy.logerr("[robot_arm_manager] %s", msg)
            return False, msg

        if not os.path.exists(sh_path):
            msg = "button start pose sh not found: %s" % sh_path
            rospy.logerr("[robot_arm_manager] %s", msg)
            return False, msg

        try:
            rospy.loginfo("[robot_arm_manager] run button start pose sh: %s", sh_path)

            ret = subprocess.call(["bash", sh_path])

            if ret != 0:
                msg = "button start pose sh failed, ret=%s" % str(ret)
                rospy.logerr("[robot_arm_manager] %s", msg)
                return False, msg

            if self.button_start_pose_wait > 0.0:
                rospy.loginfo(
                    "[robot_arm_manager] wait %.2fs after button start pose",
                    self.button_start_pose_wait
                )
                rospy.sleep(self.button_start_pose_wait)

            rospy.loginfo("[robot_arm_manager] button start pose done")
            return True, "button start pose done"

        except Exception as e:
            msg = "button start pose sh exception: %s" % str(e)
            rospy.logerr("[robot_arm_manager] %s", msg)
            return False, msg


    # =========================================================
    # Services
    # =========================================================
    def _srv_panel(self, _req):
        ok, msg = self._start_process(
            mission_name="panel",
            pkg=self.panel_pkg,
            script=self.panel_script,
            args_text=self.panel_args
        )
        if ok:
            self._publish_target_label_later(self.panel_target_label)
        return TriggerResponse(success=ok, message=msg)

    def _srv_button(self, _req):
        # 요구 흐름:
        # /arm_mission/button 호출
        # -> 기존 panel/button 프로세스가 살아있으면 먼저 종료
        # -> /home/inwoong/move_start_pose_right.sh 실행
        # -> 1초 대기
        # -> yolo_detect_1.py 실행
        # -> label_publish_delay 후 f3 publish

        if self._is_process_running():
            rospy.loginfo(
                "[robot_arm_manager] stop current process before button mission: %s",
                str(self.proc_name)
            )
            stop_ok, stop_msg = self._stop_process(reason="switch to button mission")
            if not stop_ok:
                return TriggerResponse(success=False, message=stop_msg)

        ok_pose, pose_msg = self._run_button_start_pose_sh()
        if not ok_pose:
            return TriggerResponse(success=False, message=pose_msg)

        ok, msg = self._start_process(
            mission_name="button",
            pkg=self.button_pkg,
            script=self.button_script,
            args_text=self.button_args
        )
        if ok:
            self._publish_target_label_later(self.button_target_label)
        return TriggerResponse(success=ok, message=msg)

    def _srv_restart_panel(self, _req):
        self._stop_process(reason="restart panel requested")
        ok, msg = self._start_process(
            mission_name="panel",
            pkg=self.panel_pkg,
            script=self.panel_script,
            args_text=self.panel_args
        )
        if ok:
            self._publish_target_label_later(self.panel_target_label)
        return TriggerResponse(success=ok, message=msg)

    def _srv_restart_button(self, _req):
        self._stop_process(reason="restart button requested")

        ok_pose, pose_msg = self._run_button_start_pose_sh()
        if not ok_pose:
            return TriggerResponse(success=False, message=pose_msg)

        ok, msg = self._start_process(
            mission_name="button",
            pkg=self.button_pkg,
            script=self.button_script,
            args_text=self.button_args
        )
        if ok:
            self._publish_target_label_later(self.button_target_label)
        return TriggerResponse(success=ok, message=msg)

    def _srv_stop(self, _req):
        ok, msg = self._stop_process(reason="service stop requested")
        return TriggerResponse(success=ok, message=msg)

    def _srv_status(self, _req):
        running = self._is_process_running()
        msg = (
            "state=%s, running=%s, mission=%s, process=%s, "
            "panel_label=%s, panel_command='rosrun %s %s %s', "
            "button_label=%s, button_command='rosrun %s %s %s'"
        ) % (
            self.state,
            str(running),
            self.current_mission,
            str(self.proc_name),
            self.panel_target_label,
            self.panel_pkg,
            self.panel_script,
            self.panel_args,
            self.button_target_label,
            self.button_pkg,
            self.button_script,
            self.button_args
        )
        return TriggerResponse(success=True, message=msg)

    # =========================================================
    # Shutdown
    # =========================================================
    def _on_shutdown(self):
        rospy.loginfo("[robot_arm_manager] shutdown")
        try:
            self._stop_process(reason="ros shutdown")
        except Exception:
            pass


if __name__ == "__main__":
    try:
        RobotArmMissionManager()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
