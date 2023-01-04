import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    ld = LaunchDescription()
    config = os.path.join(
        get_package_share_directory('dr_spaam_ros'),
        'config',
        'dr_spaam_ros.yaml'
        )
        
    node=Node(
        package = 'dr_spaam_ros',
        name = 'dr_spaam_ros_node',
        executable = 'dr_spaam_ros',
        parameters = [config]
    )
    ld.add_action(node)
    return ld
