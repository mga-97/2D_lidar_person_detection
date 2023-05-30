from datetime import datetime, timedelta
import numpy as np
import rclpy
import cv2
import seaborn as sns
import sys

from cv_bridge import CvBridge
from rclpy.node import Node
from rclpy.parameter import Parameter
from geometry_msgs.msg import Point, Pose, PoseArray
from sensor_msgs.msg import Image
from visualization_msgs.msg import Marker, MarkerArray

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


class JPDATracker(Node):
    def __init__(self):
        super().__init__("jpda_tracker_node")
	
        self.palette = sns.color_palette() * 2
        for i in range(len(self.palette)):
            self.palette[i] = (self.palette[i][0] * 255, self.palette[i][1] * 255, self.palette[i][2] * 255)

        self.br = CvBridge()
        
        self.dr_spaam_sub = self.create_subscription(
            PoseArray, "inliers", self.dr_spaam_callback, 10
        )
        self.yolo_sub = self.create_subscription(
            PoseArray, "yolvdsdvdo_detections", self.yolo_callback, 10
        )        

        self.tracks_img_pub = self.create_publisher(
            Image, "tracks_new", 10
	    )
        self.marker_pub = self.create_publisher(
            MarkerArray, "tracks_marker_array_new", 10
        )
    
        # init JPDA
        self.all_measurements = []
        self.tracks = set()
        
        self.measurement_model = LinearGaussian(
            ndim_state=4,
            mapping=(0, 2),
            noise_covar=np.array([[0.5, 0],   # 0.1 best, 0.01, 0
                                  [0, 0.5]])  
            )
        # transition_model = CombinedLinearGaussianTransitionModel([ConstantVelocity(0.000001), ConstantVelocity(0.000001)])
        self.transition_model = CombinedLinearGaussianTransitionModel([RandomWalk(0.001),RandomWalk(0.001),   #0.001
                                                                  RandomWalk(0.001),RandomWalk(0.0011)])
        self.predictor = KalmanPredictor(self.transition_model)
        self.updater = KalmanUpdater(self.measurement_model)
        self.hypothesiser = DistanceHypothesiser(self.predictor, self.updater, measure=Mahalanobis(), missed_distance=3)
        self.data_associator = GNNWith2DAssignment(self.hypothesiser)
        self.deleter = CovarianceBasedDeleter(covar_trace_thresh=10) #const best 3
        self.initiator = MultiMeasurementInitiator(
                prior_state=GaussianState([[0], [0], [0], [0]], np.diag([0, 1, 0, 1])),
                measurement_model=self.measurement_model,
                deleter=self.deleter,
                data_associator=self.data_associator,
                updater=self.updater,
                min_points=2,
            )
        
        self.start_time = datetime.now()
        self.count = 0

    def dr_spaam_callback(self, msg):
        measurement_set = set()
        
        for pose in msg.poses:
            det = np.array([pose.position.x, pose.position.y])
            measurement_set.add(TrueDetection(state_vector=det,
                                              groundtruth_path=None,
                                              timestamp=self.start_time + timedelta(seconds=self.count),
                                              measurement_model=self.measurement_model))
                                              
        self.all_measurements.append(measurement_set)
        
        self.update_tracks()
        self.all_measurements = []

    def yolo_callback(self, msg):
        measurement_set = set()
        
        for pose in msg.poses:
            det = np.array([pose.position.x, pose.position.y])
            measurement_set.add(TrueDetection(state_vector=det,
                                              groundtruth_path=None,
                                              timestamp=self.start_time + timedelta(seconds=self.count),
                                              measurement_model=self.measurement_model))
                                              
        self.all_measurements.append(measurement_set)
        self.update_tracks()
        self.all_measurements = []

    def update_tracks(self):
        for n, measurements in enumerate(self.all_measurements):
            # Calculate all hypothesis pairs and associate the elements in the best subset to the tracks.
            hypotheses = self.data_associator.associate(self.tracks,
                                                   measurements,
                                                   self.start_time + timedelta(seconds=self.count))
            associated_measurements = set()
            for track in self.tracks:
                hypothesis = hypotheses[track]
                if hypothesis.measurement:
                    post = self.updater.update(hypothesis)
                    track.append(post)
                    associated_measurements.add(hypothesis.measurement)
                else:  # When data associator says no detections are good enough, we'll keep the prediction
                    track.append(hypothesis.prediction)

            # Carry out deletion and initiation
            self.tracks -= self.deleter.delete_tracks(self.tracks)
            self.tracks |= self.initiator.initiate(measurements - associated_measurements,
                                         self.start_time + timedelta(seconds=n))
            
            #print(self.tracks)
            self.count += 1
            self.visualize_tracks()
            
    def visualize_tracks(self, W=512, H=512, scale=30):
        image = np.zeros(shape=[H, W, 3], dtype=np.uint8)

        marker_msg = MarkerArray()
        
        # circle
        r = 0.1
        ang = np.linspace(0, 2 * np.pi, 20)
        xy_offsets = r * np.stack((np.cos(ang), np.sin(ang)), axis=1)
    
        if len(self.tracks) > 0:
            for i, track in enumerate(list(self.tracks)):
                if len(track) > 1:
                    for state in track[-1:]:
                        
                        marker = Marker()
                        marker.header.frame_id = "base_link"
                        marker.type = marker.SPHERE
                        marker.action = marker.ADD
                        marker.scale.x = 0.2 + state.covar[0,0]
                        marker.scale.y = 0.2 + state.covar[0,0]
                        marker.scale.z = 0.2 + state.covar[0,0]
                        marker.color.a = max(0.0, 1.0-state.covar[0,0])
                        marker.color.r = self.palette[i][0] / 255.0
                        marker.color.g = self.palette[i][1] / 255.0
                        marker.color.b = self.palette[i][2] / 255.0
                        marker.pose.orientation.w = 1.0
                        marker.pose.position.x = state.state_vector[0]
                        marker.pose.position.y = state.state_vector[2]
                        marker.pose.position.z = 0.0
                        marker_msg.markers.append(marker)
                       
                        x = int(W/2) + int(state.state_vector[2] * scale)
                        y = int(H/2) - int(state.state_vector[0] * scale)
                        if x>0 and x<W and y>0 and y<H:
                            cv2.circle(image, (x,y), 2, self.palette[i], 1)
                    x = int(W/2) + int(track[-1].state_vector[2] * scale)
                    y = int(H/2) - int(track[-1].state_vector[0] * scale)
                    if x>0 and x<W and y>0 and y<H:
                        cv2.circle(image, (x,y), 4, self.palette[i], -1)
                        cv2.circle(image, (x,y), int(track[-1].covar[0,0]*20), self.palette[i], 1)

        image = cv2.flip(image, 0)
        image = cv2.putText(image, 'step: '+str(self.count), (5,15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)        
        image_msg = self.br.cv2_to_imgmsg(image)
        self.tracks_img_pub.publish(image_msg)
        
        id = 0
        for m in marker_msg.markers:
            m.id = id
            id += 1
        self.marker_pub.publish(marker_msg)
        
                        
def main(args=None):
    rclpy.init(args=args)
    node = JPDATracker()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(1)
