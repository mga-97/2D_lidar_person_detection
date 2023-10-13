import os
from glob import glob
from setuptools import find_packages, setup

package_name = 'crowd_management_ros'

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
    maintainer='user1',
    maintainer_email='misael.gonzalez.almeida@gmail.com',
    description='Package handling a crowd following the robot',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'detect_following = crowd_management_ros.detect_if_followed:main',
            'act_if_not_followed = crowd_management_ros.act_if_not_followed:main'
        ],
    },
)
