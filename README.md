roslaunch my_arm_bringup arm_total_delay_fixed.launch

roslaunch ydlidar_ros_driver lidar.launch

roslaunch my_nav grad_demo.launch record:=false

rosrun my_robot_odom system_sequence.py

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

OpenCR:
/dev/serial/by-id/usb-ROBOTIS_OpenCR_Virtual_ComPort_in_FS_Mode_FFFFFFFEFFFF-if00

OpenRB:
/dev/serial/by-id/usb-ROBOTIS_OpenRB-150_E2A54C7C50304A46462E3120FF061C2B-if00

LiDAR:
/dev/serial/by-id/usb-Silicon_Labs_CP2102_USB_to_UART_Bridge_Controller_0001-if00-port0

Logitech C922 Webcam:
/dev/v4l/by-id/usb-046d_C922_Pro_Stream_Webcam_882E697F-video-index0

Gamepad F710:
/dev/input/by-id/usb-Logitech_Wireless_Gamepad_F710_46789A23-event-joystick
