#!/usr/bin/env python
import cv2
import numpy as np
import os
import quaternion
import rclpy
import sys
import sklearn.cluster as skcluster
from abc import ABC, abstractmethod

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


class BackSubDet(Node):
    def __init__(self):
        super().__init__("back_sub_det")
        self.backSub = frameDiffBackSub()

        self.scan= self.create_subscription(
                    LaserScan,
                    'laser_local', 
                    self.scan_callback, 
                    10
                )

        self.dets = self.create_publisher(
            PoseArray, "back_sub_dets", 1
        )

        self.det_markers = self.create_publisher(
            Marker, "back_sub_marker", 1
        )

    def scan_callback(self, msg):  
        scan = clean_scan(msg.ranges)
        fgMask = self.backSub.get_foreground_mask(scan)

        if fgMask is None:
            return

        laser_fov_deg = 360
        angles = np.linspace(-np.radians(laser_fov_deg/2), np.radians(laser_fov_deg/2), len(scan))

        y = scan * np.sin(angles)
        x = scan * np.cos(angles)
        change_x = x[fgMask]
        change_y = y[fgMask]

        det_points = np.column_stack((change_x, change_y))
        finite_mask = np.isfinite(det_points).all(axis=1)
        det_points = det_points[finite_mask]

        if len(det_points) > 0:
            ms = skcluster.MeanShift(bandwidth=0.7, bin_seeding=True)
            ms.fit(det_points)
            labels = ms.labels_
            dets = ms.cluster_centers_
        else:
            return

        dets_msg = detections_to_pose_array(dets)
        dets_msg.header = msg.header
        self.dets.publish(dets_msg)

        rviz_msg = detections_to_rviz_marker(dets)
        rviz_msg.header = msg.header
        self.det_markers.publish(rviz_msg)


class BackSub(ABC):
    @abstractmethod
    def get_foreground_mask(self):
        pass

class cv2BackSub(BackSub):
    def __init__(self):
        self.backSub = cv2.createBackgroundSubtractorMOG2()
    
    def get_foreground_mask(self, ranges):
        fgMask = self.backSub.apply(clean_scan(ranges))
        fgMask = fgMask.squeeze(1).astype(bool)
        return fgMask

class frameDiffBackSub(BackSub):
    def __init__(self, window_size=15, threshold=0.1):
        self.prev_scans = []
        self.window_size = window_size
        self.threshold = threshold

    def get_foreground_mask(self, ranges):
        if len(self.prev_scans) < self.window_size:
            self.prev_scans.append(clean_scan(ranges))
            return

        current_scan = clean_scan(ranges)
        previous_scan = self.prev_scans.pop(0)
        diff = current_scan - previous_scan

        abs_diff = np.abs(diff)
        abs_diff[np.where((abs_diff > 20) | (abs_diff < self.threshold))] = 0
        fgMask = abs_diff.astype(bool)

        self.prev_scans.append(current_scan)
        return fgMask
    

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
    node = BackSubDet()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(1)
