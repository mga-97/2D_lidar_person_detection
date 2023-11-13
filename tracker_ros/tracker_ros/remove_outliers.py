#!/usr/bin/env python
import cv2
import numpy as np
import os
import quaternion
import rclpy
import sys

from copy import deepcopy
from geometry_msgs.msg import PoseArray
from nav_msgs.msg import OccupancyGrid

from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSDurabilityPolicy

from tf2_ros import TransformException
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener


class OutlierRemover(Node):
    def __init__(self):
        super().__init__("outlier_remover_node")

        self.map_img = None
        self.map_origin = None
        self.keepout_img = None
        self.keepout_origin = None
        self.yolo_poses = None

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

        self.dr_spaam_sub = self.create_subscription(
                    PoseArray, 
                    'dr_spaam_detections', 
                    self.dr_spaam_callback, 
                    10
                )

        self.yolo_sub = self.create_subscription(
                    PoseArray, 
                    'yolo_detections', 
                    self.yolo_callback, 
                    10
                )
        
        self.back_subtraction_sub = self.create_subscription(
                    PoseArray, 
                    'back_sub_dets', 
                    self.back_sub_callback, 
                    10
                )

        self.inliers_pub = self.create_publisher(
            PoseArray, "inliers", 10
        )

        self.back_sub_inliers_pub = self.create_publisher(
            PoseArray, "back_sub_inliers", 10
        )

        self.back_sub_inliers_pub = self.create_publisher(
            PoseArray, "back_sub_inliers", 10
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
        poses = self.remove_outliers(msg)
        if poses is not None:
            self.inliers_pub.publish(poses)

    def dr_spaam_callback(self, msg):
        poses = self.remove_outliers(msg)
        if poses is not None:
            self.inliers_pub.publish(poses)

    def back_sub_callback(self, msg):
        poses = self.remove_outliers(msg)
        if poses is not None:
            self.back_sub_inliers_pub.publish(poses)

    def remove_outliers(self, msg):
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
        for pose in msg.poses:
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
        return pose_array

def main(args=None):
    rclpy.init(args=args)
    node = OutlierRemover()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(1)
