#!/usr/bin/env python3
import rospy, yaml, os, tempfile
from geometry_msgs.msg import PoseStamped
from tf.transformations import euler_from_quaternion

class Recorder:
    def __init__(self):
        self.frame = rospy.get_param('~map_frame', 'map')
        self.output = rospy.get_param('~output', '/tmp/waypoints.yaml')
        self.prefix = rospy.get_param('~name_prefix', 'p')
        self.idx = 1
        self.data = {'frame_id': self.frame, 'waypoints': []}
        rospy.loginfo("[recorder] output file = %s", self.output)
        rospy.Subscriber('/move_base_simple/goal', PoseStamped, self.cb_goal, queue_size=1)

    def cb_goal(self, msg):
        q = msg.pose.orientation
        _, _, yaw = euler_from_quaternion([q.x, q.y, q.z, q.w])
        yaw_deg = yaw * 180.0 / 3.141592653589793

        wp = {
            'name': f'{self.prefix}{self.idx}',
            'x': round(msg.pose.position.x, 3),
            'y': round(msg.pose.position.y, 3),
            'yaw_deg': round(yaw_deg, 1),    # navigator가 yaw(rad) 쓰면 거기로 맞추세요
            'slow_zone': False,
            'aruco_on': False,
            'wait_key': False
        }
        self.data['waypoints'].append(wp)
        self.idx += 1
        rospy.loginfo("추가: %s", wp)

        self.save_yaml()  # ★ 추가 직후 즉시 저장

        # 미리보기
        print("\n--- 현재 누적 waypoints 미리보기 ---")
        print(yaml.safe_dump({'waypoints': self.data['waypoints']}, sort_keys=False))

    def save_yaml(self):
        # 안전한 원자적 저장: tmp에 쓰고 rename
        os.makedirs(os.path.dirname(self.output), exist_ok=True)
        dirpath = os.path.dirname(self.output)
        fd, tmppath = tempfile.mkstemp(prefix='.wp_', dir=dirpath)
        try:
            with os.fdopen(fd, 'w') as f:
                yaml.safe_dump({'frame_id': self.frame, 'waypoints': self.data['waypoints']},
                               f, sort_keys=False, allow_unicode=True)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmppath, self.output)
            rospy.loginfo("[recorder] YAML 저장 완료 → %s", self.output)
        except Exception as e:
            rospy.logerr("[recorder] 저장 실패: %s", e)
            try:
                os.remove(tmppath)
            except Exception:
                pass

if __name__ == '__main__':
    rospy.init_node('waypoint_recorder')
    Recorder()
    rospy.spin()
