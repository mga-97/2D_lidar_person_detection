#!/usr/bin/env python
import cv2
import numpy as np
import os
import quaternion
import rclpy
import sys

from copy import deepcopy
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Pose, PoseArray, Point
from visualization_msgs.msg import Marker
from nav_msgs.msg import OccupancyGrid

from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSDurabilityPolicy

from tf2_ros import TransformException
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener


class FrameDiffDet(Node):
    def __init__(self):
        super().__init__("frame_diff_det")

        self.map_img = None
        self.map_origin = None
        self.keepout_img = None
        self.keepout_origin = None
        self.yolo_poses = None
        self.prev_scans = []
        self.window_size = 20

        # tf buffer init
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        latching_qos = QoSProfile(depth=1, durability=QoSDurabilityPolicy.RMW_QOS_POLICY_DURABILITY_TRANSIENT_LOCAL)
        
        self.map_sub = self.create_subscription(
                    OccupancyGrid,
                    'map',
                    self.map_callback,
                    qos_profile=latching_qos
                    )

        self.keepout_sub = self.create_subscription(
                    OccupancyGrid,
                    'keepout_filter_mask',
                    self.keepout_callback,
                    qos_profile=latching_qos
                    )

        self.scan= self.create_subscription(
                    LaserScan,
                    'laser_local', 
                    self.scan_callback, 
                    10
                )

        self.dets = self.create_publisher(
            PoseArray, "frame_diff_dets", 1
        )

        self.det_markers = self.create_publisher(
            Marker, "frame_diff_marker", 1
        )

        self.inliers_pub = self.create_publisher(
            PoseArray, "frame_diff_inliers", 1
        )

    def map_callback(self, msg):
        self.map_img = cv2.flip(np.array(msg.data).reshape(
                            (
                                msg.info.height, 
                                msg.info.width
                            )
                    ).astype(np.uint8), 0)

        self.map_origin = np.array(
                                [
                                    msg.info.origin.position.x,
                                    msg.info.origin.position.y,
                                ]
                            )

    def keepout_callback(self, msg):
        self.keepout_img = cv2.flip(np.array(msg.data).reshape(
                            (
                                msg.info.height, 
                                msg.info.width
                            )
                    ).astype(np.uint8), 0)

        self.keepout_origin = np.array(
                                [
                                    msg.info.origin.position.x,
                                    msg.info.origin.position.y,
                                ]
                            )

    def yolo_callback(self, msg):
        self.yolo_poses = msg.poses

    def scan_callback(self, msg):
        if self.map_img is None:
            return

        T = None
        try:
            T = self.tf_buffer.lookup_transform(
                    "map",
                    "mobile_base_body_link", 
                    msg.header.stamp
                )
        except TransformException:
            print("Could not transform!")
            return
        
        if len(self.prev_scans) < self.window_size:
            self.prev_scans.append(clean_scan(msg.ranges))

        else:
            current_scan = clean_scan(msg.ranges)
            previous_scan = self.prev_scans.pop(0)
            diff = current_scan - previous_scan

            abs_diff = np.abs(diff)
            abs_diff[np.where((abs_diff > 20) | (abs_diff < 0.05))] = 0
            change_mask = abs_diff.astype(bool)

            laser_fov_deg = 360
            angles = np.linspace(-np.radians(laser_fov_deg/2), np.radians(laser_fov_deg/2), len(current_scan))

            y = current_scan * np.sin(angles)
            x = current_scan * np.cos(angles)
            change_x = x[change_mask]
            change_y = y[change_mask]

            dets = np.column_stack((change_x, change_y))
            self.prev_scans.append(current_scan)

            dets_msg = detections_to_pose_array(dets)
            dets_msg.header = msg.header
            self.dets.publish(dets_msg)

            rviz_msg = detections_to_rviz_marker(dets)
            rviz_msg.header = msg.header
            self.det_markers.publish(rviz_msg)

            # remove outliers
            TR = np.identity(4)
            TR[0, 3] = T.transform.translation.x 
            TR[1, 3] = T.transform.translation.y 
            TR[2, 3] = T.transform.translation.z

            q = np.quaternion(T.transform.rotation.w, T.transform.rotation.x, T.transform.rotation.y, T.transform.rotation.z)
            TR[:3, :3] = quaternion.as_rotation_matrix(q)

            img = self.map_img.copy()
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

            # from top-left
            map_offset_x = int(-self.map_origin[0] / 0.05)
            map_offset_y = self.map_img.shape[0] - int(-self.map_origin[1] / 0.05)
            keepout_offset_x = int(-self.keepout_origin[0] / 0.05)
            keepout_offset_y = self.keepout_img.shape[0] - int(-self.keepout_origin[1] / 0.05)

            cv2.circle(img, (map_offset_x, map_offset_y), 5, (0,0,255), -1)

            robot_pos = (TR[:2,3] / 0.05).astype(int) 
            cv2.circle(img, (robot_pos[0]+ map_offset_x, -robot_pos[1]+map_offset_y), 5, (255,0,0), -1)

            pose_array = PoseArray()
            for pose in dets_msg._poses:
                det = np.array([pose.position.x, pose.position.y, 0, 1])
                det_in_mapf = TR @ det

                det_in_mapf_pixels = (det_in_mapf[:2] / 0.05).astype(int)
                det_in_mapf_pixels[1] = -det_in_mapf_pixels[1]

                cell_to_check = det_in_mapf_pixels + np.array([map_offset_x, map_offset_y])
                keepout_cell_to_check = det_in_mapf_pixels + np.array([keepout_offset_x, keepout_offset_y])

                try:
                    cv2.circle(img, tuple(cell_to_check), 10, (0,0,20), 3)
                except:
                    print("cell has wrong format")
                # if map cell is free
                step = 10
                sum_obstacles = self.map_img[
                                    cell_to_check[1]-step:cell_to_check[1]+step,
                                    cell_to_check[0]-step:cell_to_check[0]+step
                                ].sum()

                if self.keepout_img is not None:
                    sum_obstacles += self.keepout_img[
                                        keepout_cell_to_check[1]-step:keepout_cell_to_check[1]+step,
                                        keepout_cell_to_check[0]-step:keepout_cell_to_check[0]+step
                                    ].sum()
                if sum_obstacles < 7000:
                    # print("Inlier:", det_in_mapf)
                    # dets_msg.header = msg.header
                    # dets_msg.header.frame_id = "mobile_base_double_lidar" 
                    pose_array.poses.append(pose)
                    try:
                        cv2.circle(img, tuple(cell_to_check), 10, (0,0,255), 3)
                    except:
                        print("cell has wrong format")
            # convert to ros msg and publish
            pose_array.header = msg.header
            self.inliers_pub.publish(pose_array)

def clean_scan(raw_scan, magic_num=29.99):
    scan = np.array(raw_scan)
    scan[scan == 0.0] = magic_num
    scan[np.isinf(scan)] = magic_num
    scan[np.isnan(scan)] = magic_num
    return scan

def detections_to_pose_array(dets_xy):
    pose_array = PoseArray()
    for d_xy in dets_xy:
        d_xy = -d_xy
        # Detector uses following frame convention:
        # x forward, y rightward, z downward, phi is angle w.r.t. x-axis
        p = Pose()         
        p.position.x = float(d_xy[0])
        p.position.y = float(d_xy[1])
        p.position.z = 0.0
        pose_array.poses.append(p)

    return pose_array

def detections_to_rviz_marker(dets_xy, color = (1.0, 0.0, 0.0, 1.0)):
    """
    @brief     Convert detection to RViz marker msg. Each detection is marked as
               a circle approximated by line segments.
    """
    msg = Marker()
    msg.action = Marker.ADD
    msg.ns = "dr_spaam_ros"
    msg.id = 1
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
    for d_xy in dets_xy:
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

def main(args=None):
    rclpy.init(args=args)
    node = FrameDiffDet()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(1)
