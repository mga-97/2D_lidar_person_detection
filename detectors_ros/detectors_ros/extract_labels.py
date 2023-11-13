#!/usr/bin/env python
import csv
import cv2
import message_filters  
import numpy as np
import os
import rclpy
import sys
import torch

from copy import deepcopy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan, Image
from cv_bridge import CvBridge

class PeopleSegmenting(Node):
    def __init__(self):
        super().__init__("people_segmenting_node")

        self.count = 0
        self.initial_time = 0.0

        scancsvfile = open('scans.csv', 'w', newline='')
        self.scan_writer = csv.writer(scancsvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        labelcsvfile = open('point_labels.csv', 'w', newline='')
        self.point_label_writer = csv.writer(labelcsvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        self.xy_det_file = open('xy_labels.txt', 'w')
        self.centre_angles = open('centre_angles.txt', 'w')
                
        # init YOLOv5
        self.model = torch.hub.load('ultralytics/yolov5', 'yolov5m')

        self.br = CvBridge()

        # init subs
        self.img_sub = message_filters.Subscriber(self, Image, "/cer/realsense_repeater/color_image")
        self.depth_sub = message_filters.Subscriber(self, Image, "/cer/realsense_repeater/depth_image")
        self.scan_sub = message_filters.Subscriber(self, LaserScan, "/laser_local")
        self.tss = message_filters.ApproximateTimeSynchronizer([self.img_sub, self.depth_sub, self.scan_sub], 1, slop=.1)
        self.tss.registerCallback(self.combined_callback)

        # init pubs
        self.yolo_pub = self.create_publisher(
            Image, "/yolo_image", 10
        )
        self.scan_pub = self.create_publisher(
            LaserScan, "/yolo_scan", 10
        )
                

    def combined_callback(self, img_msg, depth_msg, scan_msg):
        scan_time = scan_msg.header._stamp.sec + scan_msg.header._stamp.nanosec/1000000000
        if self.count == 0:
            self.initial_time = scan_time

        current_time = scan_time - self.initial_time

        img = self.br.imgmsg_to_cv2(img_msg)

        yolo_img = img.copy()
        
        depth = self.br.imgmsg_to_cv2(
                            depth_msg, 
                            desired_encoding='passthrough'
                    )

        processed_image = self.model(img)
               
        # write scan csv
        self.scan_writer.writerow([self.count, current_time] + list(scan_msg.ranges))

        angles_centre_dets = []
        # init labels
        labels = np.zeros((len(scan_msg.ranges),), dtype=int)
            
        dets_xy, dets_cls = [], []
        for i, det in enumerate(processed_image.xyxy[0].cpu().numpy()):
            det = det.astype(int)
            if det[5] == 0: #if class=person
                hfov = 80.14162846685939
                xc = 321.87

                start_angle = (det[0] / 640.0) * hfov
                end_angle = (det[2] / 640.0) * hfov
                angular_width = end_angle - start_angle
                angle = ( (det[0]  + (det[2] - det[0]) / 2) / 640.0) * hfov     
                angle = np.deg2rad(-angle + 6 + hfov / 2)
           
                det = det.astype(int)
                # compute avg depth of detection
                det_center_y = int(det[1] + (det[3]-det[1])/2)
                det_center_x = int(det[0] + (det[2]-det[0])/2) 
                depth_patch = self.br.imgmsg_to_cv2(
		                    depth_msg, 
				    desired_encoding='passthrough'
				    )[det_center_y-10:det_center_y+10, det_center_x-10:det_center_x+10]

                d_z = np.mean(depth_patch) + 0.2                 
                print("depth:", d_z)
                det = [ -d_z * np.cos(angle), -d_z * np.sin(angle), 0]
                dets_xy.append(det)
                dets_cls.append(0.9) # add confidence
                
                cv2.rectangle(yolo_img, (det_center_x-10, det_center_y-10), ( det_center_x+10, det_center_y+10), (0,0,255), 2)
                cv2.putText(yolo_img, '{:.3}'.format(d_z), (det_center_x, det_center_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)        
                cv2.putText(yolo_img, '{:.2}'.format(angle), (det_center_x, det_center_y+20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)             

                ranges = scan_msg.ranges
                for n in range(len(ranges)):
                    if ranges[n] > 5.0:
                        ranges[n] = float("NaN")

                angles_centre_dets.append(angle)

                # get point labels    
                window = int(np.deg2rad(angular_width) / scan_msg.angle_increment * 4) # calculate width in terms of points
                if angle > 0.02:
                    angle += np.deg2rad(4)
                    angle_index =  int(angle / scan_msg.angle_increment)
                    start_index = max(0, angle_index-window)
                    end_index =  min(angle_index+window, len(scan_msg.ranges))
                    new_scan = LaserScan()
                    new_scan.header.stamp = scan_msg.header.stamp
                    new_scan.header.frame_id = 'mobile_base_body_link'
                    new_scan.angle_min = start_index * scan_msg.angle_increment
                    new_scan.angle_max = end_index * scan_msg.angle_increment
                    new_scan.angle_increment = scan_msg.angle_increment
                    new_scan.time_increment = scan_msg.time_increment
                    new_scan.range_min = scan_msg.range_min
                    new_scan.range_max = scan_msg.range_max
                    new_scan.ranges = ranges[start_index:end_index]
                    labels[start_index:end_index] = 1
                    self.scan_pub.publish(new_scan)

                if angle < -0.02:
                    angle = -angle
                    angle -= np.deg2rad(4)
                    angle_index =  len(scan_msg.ranges) - int(angle / scan_msg.angle_increment)
                    start_index = angle_index-window
                    end_index =  angle_index+window
                    new_scan = LaserScan()
                    new_scan.header.stamp = scan_msg.header.stamp
                    new_scan.header.frame_id = 'mobile_base_body_link'
                    new_scan.angle_min = start_index * scan_msg.angle_increment
                    new_scan.angle_max = end_index * scan_msg.angle_increment
                    new_scan.angle_increment = scan_msg.angle_increment
                    new_scan.time_increment = scan_msg.time_increment
                    new_scan.range_min = scan_msg.range_min
                    new_scan.range_max = scan_msg.range_max
                    new_scan.ranges = ranges[start_index:end_index]
                    labels[start_index:end_index] = 1
                    self.scan_pub.publish(new_scan)
                                    
        self.point_label_writer.writerow(labels)
        self.xy_det_file.write(f'{self.count},{[xy[:2] for xy in dets_xy]}\n')
        self.centre_angles.write(f'{str(angles_centre_dets)}\n')
        
        rgb = cv2.cvtColor(yolo_img, cv2.COLOR_BGR2RGB)
        image_msg = self.br.cv2_to_imgmsg(rgb)
        self.yolo_pub.publish(image_msg)
        cv2.imwrite(f'./images/det{self.count}.jpg', rgb)
        self.count += 1
                


def main(args=None):
    rclpy.init(args=args)
    node = PeopleSegmenting()
    rclpy.spin(node)
    node.scan_writer.close()
    node.point_label_writer.close()
    node.centre_angles.close()
    rclpy.shutdown()

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(1)

