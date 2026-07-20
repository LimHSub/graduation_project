#!/bin/bash

rostopic pub -1 /initialpose geometry_msgs/PoseWithCovarianceStamped "
header:
  frame_id: 'map'
pose:
  pose:
    position:
      x: 2.71
      y: -16.14
      z: 0.0
    orientation:
      x: 0.0
      y: 0.0
      z: 1.0
      w: 0.0
  covariance: [0.25, 0.0, 0.0, 0.0, 0.0, 0.0,
               0.0, 0.25, 0.0, 0.0, 0.0, 0.0,
               0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
               0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
               0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
               0.0, 0.0, 0.0, 0.0, 0.0, 0.068]
}"
