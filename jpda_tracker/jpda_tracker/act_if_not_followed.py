import numpy as np
import quaternion
import rclpy
import subprocess
import sys


from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.duration import Duration
from geometry_msgs.msg import Pose, PoseArray, PoseWithCovarianceStamped
from std_msgs.msg import Bool


def quaternion_multiply(quaternion1, quaternion0):
    w0, x0, y0, z0 = quaternion0
    w1, x1, y1, z1 = quaternion1
    return np.array([-x1 * x0 - y1 * y0 - z1 * z0 + w1 * w0,
                     x1 * w0 + y1 * z0 - z1 * y0 + w1 * x0,
                     -x1 * z0 + y1 * w0 + z1 * x0 + w1 * y0,
                     x1 * y0 - y1 * x0 + z1 * w0 + w1 * z0], dtype=np.float64)
                     
                     
class EngagementReactor(Node):
    def __init__(self):
        super().__init__("engagement_reactor_node")
        
        # subscribers
        self.is_followed_sub = self.create_subscription(
            Bool, "is_followed", self.is_followed_callback, 10
        )
        self.amcl_sub = self.create_subscription(
            PoseWithCovarianceStamped, "amcl_pose", self.amcl_callback, 10
        )
                
         
        self.not_followed_count = 0
        self.last_orientation = None
        
        self.rotz180 = quaternion.as_float_array(quaternion.from_euler_angles(0,0,np.pi))

    def amcl_callback(self, msg):
        self.last_position = msg.pose.pose.position
        self.last_orientation = msg.pose.pose.orientation
        
        
    def is_followed_callback(self, msg):            
        if not msg.data:
            self.not_followed_count += 1
        else:
            self.not_followed_count = 0
            
        if self.not_followed_count > 20:
            print("NOT FOLLOWED! ACTING NOW")
            pos = self.last_position
            old_orient = [self.last_orientation.w, self.last_orientation.x, self.last_orientation.y, self.last_orientation.z]
            new_orient = quaternion_multiply(self.rotz180, old_orient)
            
            # send goal action
            msg = "{header: {frame_id: \"map\"}, pose: {position: {x: " + \
                    str(pos.x) + \
                    ', y: ' \
                    + str(pos.y) + \
                    ', z: ' \
                    + str(pos.z) + \
                    "}, orientation: {x: " + \
                    str(new_orient[1]) + \
                    ', y: ' \
                    + str(new_orient[2]) + \
                    ', z: ' \
                    + str(new_orient[3]) + \
                    ', w: ' \
                    + str(new_orient[0]) + \
                    "}}}" 

            action_cmd = [
                        'ros2', 
                        'topic', 
                        'pub', 
                        '--once',
                        '/goal_pose', 
                        'geometry_msgs/msg/PoseStamped', 
                        msg
                    ]

            action_result = subprocess.run(action_cmd)
        
def main(args=None):
    rclpy.init(args=args)
    node = EngagementReactor()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(1)
