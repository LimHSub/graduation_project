roslaunch my_arm_bringup arm_total_delay_fixed.launch

roslaunch ydlidar_ros_driver lidar.launch

roslaunch my_nav grad_demo.launch record:=false

rosrun my_robot_odom system_sequence.py

===========================================================
rosservice call /waypoint_navigator/goto "index: 0"  # 1층 Lidar 주행

rosservice call /waypoint_navigator/marker_start_1   # 엘리베이터 접근(D435)

rosservice call /arm_mission/panel   # 패널 버튼 누르기

rosservice call /waypoint_navigator/marker_start_2   # 아루코마커 전진(로지텍)

rosservice call /arm_mission/button   # 엘레베이터 버튼 누르기

rosservice call /waypoint_navigator/marker_start_3   # 아루코마커 후진(로지텍)

rosservice call /waypoint_navigator/switch_next_map  # 맵 전환

rosservice call /waypoint_navigator/goto "index: 1"  # 3층 Lidar 주행

===========================================================

event 번호 확인 

ls -l /dev/input/by-id/ | grep -i logitech

event 번호 확인후 odom_imu_grad_demo.py event 번호 수정



usb cam 

ls /dev/video*

===========================================================

(정면 방향 - panel) ~/move_start_pose_panel.sh

(오른쪽 방향 - button) ~/move_start_pose_right.sh

=======================================================================

포트 번호 수정
파일에 있는 ACM0, ACM1 등을 보드 및 센서 by-id 고정 경로로 변경(어느 파일인지는 임형섭에게 물어보기)

STM32:
/dev/serial/by-id/usb-STMicroelectronics_STM32_STLink_0668FF565087534867135942-if02

/home/park/catkin_ws/src/my_robot_odom/scripts/odom_imu_grad_demo.py

OpenCR:
/dev/serial/by-id/usb-ROBOTIS_OpenCR_Virtual_ComPort_in_FS_Mode_FFFFFFFEFFFF-if00\

/home/park/catkin_ws/src/my_arm_bringup/launch/bringup.launch

OpenRB:
/dev/serial/by-id/usb-ROBOTIS_OpenRB-150_E2A54C7C50304A46462E3120FF061C2B-if00

/home/park/catkin_ws/src/my_arm_bringup/launch/delivery_app_bridge.launch

LiDAR:
/dev/serial/by-id/usb-Silicon_Labs_CP2102_USB_to_UART_Bridge_Controller_0001-if00-port0

/home/park/catkin_ws/src/ydlidar_ros_driver/launch

Logitech C922 Webcam:
/dev/v4l/by-id/usb-046d_C922_Pro_Stream_Webcam_882E697F-video-index0

/home/park/catkin_ws/src/my_nav/launch/grad_demo.launch

Gamepad F710:
/dev/input/by-id/usb-Logitech_Wireless_Gamepad_F710_46789A23-event-joystick

/home/park/catkin_ws/src/my_robot_odom/scripts/odom_imu_grad_demo.py

뎁스카메라는 수정 안함

====================================================================
