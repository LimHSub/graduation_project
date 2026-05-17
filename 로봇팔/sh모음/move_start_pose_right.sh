#!/bin/bash

rostopic pub -1 /arm_controller/command trajectory_msgs/JointTrajectory "
joint_names:
- Revolute1
- Revolute2
- Revolute3
- Revolute4
- Revolute5
points:
- positions: [0.0, 0.098, -1.318, 0.0, 1.243]
  velocities: [0.0, 0.0, 0.0, 0.0, 0.0]
  time_from_start: {secs: 3, nsecs: 0}
"
