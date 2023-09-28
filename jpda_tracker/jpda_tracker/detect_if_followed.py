from datetime import datetime, timedelta
import numpy as np
import rclpy
import cv2
import quaternion
import seaborn as sns
import sys

from cv_bridge import CvBridge
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.duration import Duration
from geometry_msgs.msg import Point, Pose, PoseArray
from sensor_msgs.msg import Image
from visualization_msgs.msg import Marker, MarkerArray

from tf2_ros import TransformException
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener

from stonesoup.models.transition.linear import CombinedLinearGaussianTransitionModel, \
                                               ConstantVelocity, RandomWalk
from stonesoup.types.groundtruth import GroundTruthPath, GroundTruthState
from stonesoup.types.detection import TrueDetection
from stonesoup.types.detection import Clutter
from stonesoup.models.measurement.linear import LinearGaussian
from stonesoup.predictor.kalman import KalmanPredictor
from stonesoup.updater.kalman import KalmanUpdater
from stonesoup.hypothesiser.probability import PDAHypothesiser
from stonesoup.dataassociator.probability import JPDA
from stonesoup.types.state import GaussianState
from stonesoup.types.track import Track
from stonesoup.types.array import StateVectors
from stonesoup.functions import gm_reduce_single
from stonesoup.types.update import GaussianStateUpdate
from stonesoup.hypothesiser.distance import DistanceHypothesiser
from stonesoup.measures import Mahalanobis
from stonesoup.dataassociator.neighbour import GNNWith2DAssignment
from stonesoup.deleter.error import CovarianceBasedDeleter
from stonesoup.types.state import GaussianState
from stonesoup.initiator.simple import MultiMeasurementInitiator


class FollowingGroupDetector(Node):
    def __init__(self):
        super().__init__("following_group_detector_node")
        
        # tf buffer init
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        
        self.inliers_sub = self.create_subscription(
            PoseArray, "inliers", self.inliers_callback, 10
        )
        
        self.marker_pub = self.create_publisher(
            MarkerArray, "detected_group_marker", 10
        )
        
        self.moving_avg_group = None # format: centerx, centery, convariance in robot frame
         
        self.start_time = datetime.now()
        self.count = 0

    def inliers_callback(self, msg):            
        group_poses = []    
        group_center = None
        group_covariance = None
        
        for pose in msg.poses:
            x = pose.position.x
            y = pose.position.y
            if x < -0.5 and x > -6 and np.abs(y) < 3:
                group_poses.append([x,y])
        
        if len(group_poses) > 0:   
            group_center = np.mean(group_poses,axis=0)
            group_covariance = np.max(np.std(group_poses,axis=0))    
            #print(group_covariance)
            if group_covariance > 0.6:
                group_covariance = 0.6
             
            if self.moving_avg_group is not None:
                self.moving_avg_group[:2] = (self.moving_avg_group[:2] + group_center) / 2
                self.moving_avg_group[2] = (self.moving_avg_group[2] + group_covariance) / 2
            else:
                self.moving_avg_group = np.zeros(3)
                self.moving_avg_group[:2] = group_center
                self.moving_avg_group[2] = group_covariance
                
            #visualization
            marker_msg = MarkerArray()
            
            # circle
            r = 0.1
            ang = np.linspace(0, 2 * np.pi, 20)
            xy_offsets = r * np.stack((np.cos(ang), np.sin(ang)), axis=1)
                                     
            marker = Marker()
            marker.id = 1
            marker.lifetime = Duration(seconds=1.0).to_msg()
            marker.header.stamp = msg.header.stamp
            marker.header.frame_id = "mobile_base_body_link"
            marker.type = marker.SPHERE
            marker.action = marker.ADD
            marker.scale.x = self.moving_avg_group[2] * 10
            marker.scale.y = self.moving_avg_group[2] * 10
            marker.scale.z = self.moving_avg_group[2] * 10
            
            marker.color.a = 0.3
            marker.color.r = 1.0
            marker.color.g = 1.0
            marker.color.b = 0.0
            
            marker.pose.orientation.w = 1.0
            marker.pose.position.x = self.moving_avg_group[0]
            marker.pose.position.y = self.moving_avg_group[1]
            marker.pose.position.z = 0.0
            marker_msg.markers.append(marker)
                       
            self.marker_pub.publish(marker_msg)
        
def main(args=None):
    rclpy.init(args=args)
    node = FollowingGroupDetector()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(1)
