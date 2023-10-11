import os
from launch import LaunchDescription
from ament_index_python.packages import get_package_share_directory
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node


def generate_launch_description():
    yolo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([os.path.join(
            get_package_share_directory('detectors_ros'), 'launch'),
            '/yolo.launch.py'])
    )

    dr_spaam = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([os.path.join(
            get_package_share_directory('detectors_ros'), 'launch'),
            '/dr_spaam_ros.launch.py'])
    )
        
    return LaunchDescription([
        yolo,
        dr_spaam
   ])
