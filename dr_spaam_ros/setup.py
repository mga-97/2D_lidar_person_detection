from setuptools import setup

package_name = 'dr_spaam_ros'

setup(
 name=package_name,
 version='0.0.0',
 packages=[package_name],
 data_files=[
     ('share/ament_index/resource_index/packages',
             ['resource/' + package_name]),
     ('share/' + package_name, ['package.xml']),
   ],
 install_requires=['setuptools'],
 zip_safe=True,
 maintainer='Dan Jia',
 maintainer_email='jia@vision.rwth-aachen.de',
 description='ROS interface for DR-SPAAM detector',
 license='TODO: License declaration',
 tests_require=['pytest'],
 entry_points={
     'console_scripts': [
             'test = dr_spaam_ros.dr_spaam_ros:main'
     ],
   },
)
