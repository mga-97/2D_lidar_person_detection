import os
from glob import glob
from setuptools import find_packages, setup

package_name = 'tracker_ros'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py'))
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Misael González ALmeida',
    maintainer_email='misael.gonzalez.almeida@gmail.com',
    description='JPDA data association for tracking PoseArray measurements',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'jpda_tracker = tracker_ros.jpda_tracker:main',
            'outlier_remover = tracker_ros.remove_outliers:main',
            'jpda_tracker_pose_pub = tracker_ros.jpda_tracker_publish_poses:main'
        ],
    },
)
