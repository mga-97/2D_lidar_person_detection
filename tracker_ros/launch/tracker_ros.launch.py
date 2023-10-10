import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    ld = LaunchDescription()
        
    tracker=Node(
        package = 'tracker_ros',
        executable = 'jpda_tracker'
    )
    outlier_removal=Node(
        package = 'tracker_ros',
        executable = 'outlier_remover'
    )
    tracker_pose_pub=Node(
        package = 'tracker_ros',
        executable = 'jpda_tracker_pose_pub'
    )
    
    ld.add_action(tracker)
    ld.add_action(outlier_removal)
    ld.add_action(tracker_pose_pub)
    return ld
