from datetime import datetime, timedelta
import numpy as np
import rclpy
import seaborn as sns
import sys

from rclpy.node import Node
from rclpy.parameter import Parameter
from geometry_msgs.msg import Point, Pose, PoseArray
from visualization_msgs.msg import Marker

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
        for i in range(len(palette)):
            palette[i] = (palette[i][0] * 255, palette[i][1] * 255, palette[i][2] * 255)

        self.dr_spaam_sub = self.create_subscription(
            PoseArray, "dr_spaam_detections", self.dr_spaam_callback, 1
        )
        self.yolo_sub = self.create_subscription(
            PoseArray, "dr_spaam_yolo_detections", self.yolo_callback, 1
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
        self.transition_model = CombinedLinearGaussianTransitionModel([RandomWalk(0.03),RandomWalk(0.03),   #0.001
                                                                  RandomWalk(0.03),RandomWalk(0.03)])
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

    def dr_spaam_callback(self, msg):
        measurement_set = set()
        
        for pose in msg.poses:
            det = np.array([pose.position.x, pose.position.z])
            measurement_set.add(TrueDetection(state_vector=det,
                                              groundtruth_path=None,
                                              timestamp=self.start_time + timedelta(seconds=line_count),
                                              measurement_model=self.measurement_model))
                                              
        self.all_measurements.append(measurement_set)

        
    def yolo_callback(self, msg):
        measurement_set = set()
        
        for pose in msg.poses:
            det = np.array([pose.position.x, pose.position.z])
            measurement_set.add(TrueDetection(state_vector=det,
                                              groundtruth_path=None,
                                              timestamp=self.start_time + timedelta(seconds=line_count),
                                              measurement_model=self.measurement_model))
                                              
        self.all_measurements.append(measurement_set)
        
    def update_tracks(self):
        for n, measurements in enumerate(self.all_measurements):
            # Calculate all hypothesis pairs and associate the elements in the best subset to the tracks.
            hypotheses = self.data_associator.associate(tracks,
                                                   measurements,
                                                   self.start_time + timedelta(seconds=line_count))
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
            
            print(self.tracks)
                        
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