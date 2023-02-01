from setuptools import setup

package_name = 'jpda_tracker'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/config/', ['config/jpda_rviz.rviz']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Stefano Rosa',
    maintainer_email='stefano.rosa@iit.it',
    description='JPDA data association for tracking PoseArray measurements',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'jpda_tracker = jpda_tracker.jpda_tracker:main'
        ],
    },
)
