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

class ConvertToCSV(Node):
    def __init__(self):
        super().__init__("convert_to_csv")

        self.count = 0
        self.time = 0.0

        scancsvfile = open('scans.csv', 'w', newline='')
        self.scan_writer = csv.writer(scancsvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        
        self._scan_sub = self.create_subscription(
            LaserScan, '/laser_local', self.callback, 1
        )
                

    def callback(self, scan_msg):
        index_columns = np.array([self.count, self.time])
        row = np.concatenate([index_columns, scan_msg.ranges])
        row[np.isinf(row)] = 29.99
        row = list(row)
        row[0] = int(row[0])
        self.scan_writer.writerow(row)
        print(row[:2])
        self.count += 1
        self.time += scan_msg.scan_time
                


def main(args=None):
    rclpy.init(args=args)
    node = ConvertToCSV()
    rclpy.spin(node)
    node.scan_writer.close()
    rclpy.shutdown()

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(1)

