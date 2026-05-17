roslaunch my_arm_bringup arm_total_delay_fixed.launch

roslaunch ydlidar_ros_driver lidar.launch

roslaunch my_nav grad_demo.launch record:=false

rosrun my_robot_odom system_sequence.py

===========================================================

event 번호 확인 
ls -l /dev/input/by-id/ | grep -i logitech event 

번호 확인후 odom_imu_grad_demo.py event 번호 수정

usb cam 
ls /dev/video*

===========================================================

(정면 방향 - panel) ~/move_start_pose_panel.sh

(오른쪽 방향 - button) ~/move_start_pose_right.sh
