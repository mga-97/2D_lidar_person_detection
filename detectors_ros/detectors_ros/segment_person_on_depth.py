#!/usr/bin/env python
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

        # init YOLOv5
        self.model = torch.hub.load('ultralytics/yolov5', 'yolov5m')

        self.br = CvBridge()

        # init subs
        self.img_sub = message_filters.Subscriber(self, Image, "/cer/realsense_repeater/color_image")
        self.depth_sub = message_filters.Subscriber(self, Image, "/cer/realsense_repeater/depth_image")
        self.tss = message_filters.ApproximateTimeSynchronizer([self.img_sub, self.depth_sub], 1, slop=.1)
        self.tss.registerCallback(self.combined_callback)
    

    def combined_callback(self, img_msg, depth_msg):
        img = self.br.imgmsg_to_cv2(img_msg)

        depth = self.br.imgmsg_to_cv2(
                            depth_msg, 
                            desired_encoding='passthrough'
                    )

        processed_image = self.model(img)       
     

        dets_xy, dets_cls = [], []
        for i, det in enumerate(processed_image.xyxy[0].cpu().numpy()):

            if det[5] == 0: #if class=person
                start_angle = (det[0] / 640.0) * 75
                end_angle = (det[2] / 640.0) * 75
                angle = ( (det[0] + (det[2] - det[0]) / 2) / 640.0) * 75      
                angle = np.deg2rad(-angle + 75 / 2)
                d_z = 3.0
        
                det = det.astype(int)
                increase = int((det[2]-det[0])*0.2)
                depth_patch = depth[
                                det[1]:det[3] - int((det[3]-det[1])/2), 
                                max(0, det[0]-increase):min(640, det[2]+increase)
                            ]
                depth_arr = depth_patch.reshape((-1, 1))
                depth_arr = np.float32(depth_arr)

                # define criteria and apply kmeans()
                criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
                ret, label, center = cv2.kmeans(depth_arr, 2,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)
                A = depth_arr[label.ravel()==0]
                # B = depth_arr[label.ravel()==1]

                avg_distance = None
                if np.unique(center)[0] > 0:
                    avg_distance = np.unique(center)[0]     
                print("avg depth:", avg_distance)

                # # Now convert back into uint8, and make original image
                # center[center == np.unique(center)[1]] = 0.0
                # center[center == np.unique(center)[0]] = 100.0
                
                center = np.uint8(center * 25.5)
                center = center[label.flatten()]
                center = center.reshape((depth_patch.shape))

                # img = cv2.cvtColor(img,  cv2.COLOR_BGR2RGB)
                # img[
                #     det[1]:det[3] - int((det[3]-det[1])/2), 
                #     max(0, det[0]-increase):min(640, det[2]+increase),
                #     0
                #     ] -= center 
                # img[
                #     det[1]:det[3] - int((det[3]-det[1])/2), 
                #     max(0, det[0]-increase):min(640, det[2]+increase),
                #     2
                #     ] -= center 
                img = cv2.cvtColor(img,  cv2.COLOR_BGR2RGB)
                cv2.rectangle(img, (det[0],det[1]), (det[2],det[3]), (0,0,255), 2)
                cv2.rectangle(img, (max(0, det[0]-increase),det[1]), (min(640, det[2]+increase),det[3] - int((det[3]-det[1])/2)), (255,0,0), 2)
                cv2.imshow(str(i), center)

        cv2.imshow("img", img)        
        cv2.imshow("depth", (depth * 25.5).astype(np.uint8))        
        cv2.waitKey(1)

                # # compute avg depth of detection
                # det_center_y = int(det[1] + (det[3]-det[1])/2)
                # det_center_x = int(det[0] + (det[2]-det[0])/2) 
                # depth_patch = self.br.imgmsg_to_cv2(
                #             depth_msg, 
                #     desired_encoding='passthrough'
                #     )[det_center_y-10:det_center_y+10, det_center_x-10:det_center_x+10]
                # d_z = np.mean(depth_patch) + 0.2                 
                # det = [ -d_z * np.cos(angle), -d_z * np.sin(angle), 0]
                # dets_xy.append(det)
                # dets_cls.append(0.9) # add confidence


def main(args=None):
    rclpy.init(args=args)
    node = PeopleSegmenting()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(1)


