import cv2
import numpy as np
import rclpy
import sys
import torch

from rclpy.node import Node
from rclpy.parameter import Parameter
from sensor_msgs.msg import LaserScan, Image
from geometry_msgs.msg import Point, Pose, PoseArray
from visualization_msgs.msg import Marker

from dr_spaam.detector import Detector
from dr_spaam.utils import utils as u
from cv_bridge import CvBridge


class DrSpaamROS(Node):
    """ROS node to detect pedestrian using DROW3 or DR-SPAAM."""

    def __init__(self):
        super().__init__("dr_spaam_ros")
        self._read_params()
        self._detector = Detector(
            self.weight_file,
            model=self.detector_model,
            gpu=self.use_gpu,
            stride=self.stride,
            panoramic_scan=self.panoramic_scan,
        )
        self._init()

    def _read_params(self):
        """
        @brief      Reads parameters from ROS server.
        """
        self.declare_parameter("weight_file", "/code/humble/self_supervised_person_detection/checkpoints/ckpt_jrdb_ann_drow3_e40.pth")
        self.declare_parameter("conf_thresh", 0.9)
        self.declare_parameter("stride", 2)
        self.declare_parameter("detector_model", "DROW3")
        self.declare_parameter("panoramic_scan", True)	
        self.declare_parameter("use_gpu", False)	
		
        self.weight_file = str(self.get_parameter("weight_file").value)
        self.conf_thresh = float(self.get_parameter("conf_thresh").value)
        self.stride = int(self.get_parameter("stride").value)
        self.detector_model = str(self.get_parameter("detector_model").value)
        self.panoramic_scan = bool(self.get_parameter("panoramic_scan").value)
        self.use_gpu = bool(self.get_parameter("use_gpu").value)

    def read_subscriber_param(self, name):
        """
        @brief      Convenience function to read subscriber parameter.
        """
        self.declare_parameter("subscriber/" + name + "/topic", "laser")
        self.declare_parameter("subscriber/" + name + "/queue_size", 10)
	
        topic = str(self.get_parameter("subscriber/" + name + "/topic").value)
        queue_size = int(self.get_parameter("subscriber/" + name + "/queue_size").value)
        return topic, queue_size

    def read_publisher_param(self, name):
        """
        @brief      Convenience function to read publisher parameter.
        """
        self.declare_parameter("publisher/" + name + "/topic", "dr_spaam_" + name)
        self.declare_parameter("publisher/" + name + "/queue_size", 10)
        self.declare_parameter("publisher/" + name + "/latch", False)

        topic = str(self.get_parameter("publisher/" + name + "/topic").value)
        queue_size = int(self.get_parameter("publisher/" + name + "/queue_size").value)
        latch = bool(self.get_parameter("publisher/" + name + "/latch").value)
        return topic, queue_size, latch
    
    def _init(self):
        """
        @brief      Initialize ROS connection.
        """
        # Publisher
        topic, queue_size, latch = self.read_publisher_param("detections")
        self._dets_pub = self.create_publisher(
            PoseArray, topic, queue_size #, latch=latch
        )

        topic, queue_size, latch = self.read_publisher_param("rviz")
        self._rviz_pub = self.create_publisher(
            Marker, topic, queue_size #, latch=latch
        )
        self._rviz_detectron2_pub = self.create_publisher(
            Marker, "dr_spaam_detectron2_viz", queue_size #, latch=latch
        )
        self._img_pub = self.create_publisher(
            Image, "image", 1
	)

        # init YOLOv5
        self.model = torch.hub.load('ultralytics/yolov5', 'yolov5m')
	
        # Subscriber
        topic, queue_size = self.read_subscriber_param("scan")
        self._scan_sub = self.create_subscription(
            LaserScan, topic, self._scan_callback, queue_size
        )
        self._img_sub = self.create_subscription(Image, 
            '/camera/color/image_raw',
            self._image_callback,
            1
        )

        self.br = CvBridge()
        #self.video_out = cv2.VideoWriter('/tmp/output.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 20.0, (512,512))
	
    def _image_callback(self, msg):
        current_frame = self.br.imgmsg_to_cv2(msg)
        processed_image = self.model(current_frame)
        for det in processed_image.xyxy[0].cpu().numpy():
            if det[5] == 0:
                cv2.rectangle(current_frame, (int(det[0]), int(det[1])), (int(det[2]), int(det[3])), (0,0,255), 2)
        cv2.imshow("result", current_frame)
        cv2.waitKey(2)        
	
        dets_xy, dets_cls = [], []
        for det in processed_image.xyxy[0].cpu().numpy():
            if det[5] == 0:
                start_angle = (det[0] / 640.0) * 75
                end_angle = (det[2] / 640.0) * 75
                angle = ( (det[0] + (det[2] - det[0]) / 2) / 640.0) * 75	  
                angle = np.deg2rad(-angle + 75 / 2)
                d_z = 3.0
                det = [ -d_z * np.cos(angle), -d_z * np.sin(angle), 0]
                dets_xy.append(det)
                dets_cls.append(1.0)
        
        rviz_msg = detections_to_rviz_marker(np.array(dets_xy), np.array(dets_cls), (0.0, 1.0, 0.0, 1.0))
        rviz_msg.header = msg.header
        rviz_msg.header.frame_id = "mobile_base_double_lidar"
        self._rviz_detectron2_pub.publish(rviz_msg)
	
    def _depth_callback(self, msg):
        self._last_depth = msg
	
    def _scan_callback(self, msg):
        if (
            self._dets_pub.get_subscription_count() == 0
            and self._rviz_pub.get_subscription_count() == 0
        ):
            return

        # TODO check the computation here
        if not self._detector.is_ready():
            self._detector.set_laser_fov(
                np.rad2deg(msg.angle_increment * len(msg.ranges))
            )

        scan = np.array(msg.ranges)
        scan[scan == 0.0] = 29.99
        scan[np.isinf(scan)] = 29.99
        scan[np.isnan(scan)] = 29.99

        # t = time.time()
        dets_xy, dets_cls, _ = self._detector(scan)
        # print("[DrSpaamROS] End-to-end inference time: %f" % (t - time.time()))

        # confidence threshold
        conf_mask = (dets_cls >= self.conf_thresh).reshape(-1)
        dets_xy = dets_xy[conf_mask]
        dets_cls = dets_cls[conf_mask]

        # convert to ros msg and publish
        dets_msg = detections_to_pose_array(dets_xy, dets_cls)
        dets_msg.header = msg.header
        self._dets_pub.publish(dets_msg)

        rviz_msg = detections_to_rviz_marker(dets_xy, dets_cls)
        rviz_msg.header = msg.header
        self._rviz_pub.publish(rviz_msg)

	# create scan image
        W, H = 512, 512
        scale = 30
        image = np.zeros(shape=[H, W, 3], dtype=np.uint8)
        step = 2*np.pi / scan.shape[0]
        for i, s in enumerate(scan):
            if s < np.inf:
                x = int(W/2) + int(np.cos(i*step) * s * scale)
                y = int(H/2) - int(np.sin(i*step) * s * scale)
                if x>0 and x<W and y>0 and y<H:
                    image[x,y] = (255,255,255)

        clean_image = image.copy()
        for det, cl in zip(dets_xy, dets_cls):
            x = int(W/2) + int(det[1] * scale)
            y = int(H/2) - int(det[0] * scale)
            if x>0 and x<W and y>0 and y<H:
                sz = 8
                patch = clean_image[y-sz:y+sz, x-sz:x+sz, 0]
                cv2.rectangle(image, (x-sz,y-sz), (x+sz,y+sz), (80,0,0), 1)

                nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(patch, connectivity=8)
                sizes = stats[1:, -1]
                centroids = centroids[1:]
                nb_components = nb_components - 1

                #for i, c in enumerate(centroids):
                #    cv2.circle(image, (int(c[0]+x-sz), int(c[1]+y-sz)), 3, (100,100,100), 1)
                if len(centroids) == 2 and sizes[0]>3 and sizes[1]>3:
                    cv2.circle(image, (x,y), 12, (0,255,0), 2)
                else:
                    cv2.circle(image, (x,y), 12, (0,0,20), 2)
                		
		    
        image = cv2.flip(image, 0)
        #self.video_out.write(image)
        msg2 = self.br.cv2_to_imgmsg(image)
        self._img_pub.publish(msg2)

def detections_to_rviz_marker(dets_xy, dets_cls, color = (1.0, 0.0, 0.0, 1.0)):
    """
    @brief     Convert detection to RViz marker msg. Each detection is marked as
               a circle approximated by line segments.
    """
    msg = Marker()
    msg.action = Marker.ADD
    msg.ns = "dr_spaam_ros"
    msg.id = 0
    msg.type = Marker.LINE_LIST
    #msg.header.frame_id = "mobile_base_double_lidar"
    # set quaternion so that RViz does not give warning
    msg.pose.orientation.x = 0.0
    msg.pose.orientation.y = 0.0
    msg.pose.orientation.z = 0.0
    msg.pose.orientation.w = 1.0

    msg.scale.x = 0.03  # line width
    msg.color.r = color[0]
    msg.color.g = color[1]
    msg.color.b = color[2]    
    msg.color.a = color[3]

    # circle
    r = 0.4
    ang = np.linspace(0, 2 * np.pi, 20)
    xy_offsets = r * np.stack((np.cos(ang), np.sin(ang)), axis=1)

    # to msg
    for d_xy, d_cls in zip(dets_xy, dets_cls):
        d_xy = -d_xy
        for i in range(len(xy_offsets) - 1):
            # start point of a segment
            p0 = Point()
            p0.x = d_xy[0] + xy_offsets[i, 0]
            p0.y = d_xy[1] + xy_offsets[i, 1]
            p0.z = 0.0
            msg.points.append(p0)

            # end point
            p1 = Point()
            p1.x = d_xy[0] + xy_offsets[i + 1, 0]
            p1.y = d_xy[1] + xy_offsets[i + 1, 1]
            p1.z = 0.0
            msg.points.append(p1)

    return msg


def detections_to_pose_array(dets_xy, dets_cls):
    pose_array = PoseArray()
    for d_xy, d_cls in zip(dets_xy, dets_cls):
        d_xy = -d_xy
        # Detector uses following frame convention:
        # x forward, y rightward, z downward, phi is angle w.r.t. x-axis
        p = Pose()
        p.position.x = float(d_xy[0])
        p.position.y = float(d_xy[1])
        p.position.z = 0.0
        pose_array.poses.append(p)

    return pose_array


def main(args=None):
    rclpy.init(args=args)
    node = DrSpaamROS()
    rclpy.spin(node)
    #node.video_out.release()
    rclpy.shutdown()

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(1)