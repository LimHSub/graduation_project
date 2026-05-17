cmake_minimum_required(VERSION 3.0.2)
project(my_robot_odom)

## 메시지/서비스 생성을 위해 message_generation 추가
find_package(catkin REQUIRED COMPONENTS
  geometry_msgs
  nav_msgs
  rospy
  std_msgs
  tf
  message_generation
)

## srv 파일 등록
add_service_files(
  FILES
  SetMode.srv
    GotoWaypoint.srv
)

## 메시지/서비스 코드 생성
generate_messages(
  DEPENDENCIES
    std_msgs
)

## 패키지 설정: message_runtime 의존 추가
catkin_package(
  CATKIN_DEPENDS
    geometry_msgs
    nav_msgs
    rospy
    std_msgs
    tf
    message_runtime
)

include_directories(
  ${catkin_INCLUDE_DIRS}
)

## (파이썬 스크립트 설치가 필요하면 주석 해제)
# catkin_install_python(PROGRAMS
#   scripts/odom_imu.py
#   scripts/waypoint_navigator.py
#   DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
# )
