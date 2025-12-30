from setuptools import find_packages, setup
import os
from glob import glob
package_name = 'pubsub_package'


setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
        (os.path.join('share', package_name, 'maps'), glob('maps/*')),
        # (os.path.join('share',package_name,'srv'),glob('pubsub_package/srv/Plan.srv')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='vmwsp',
    maintainer_email='vmwsp@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
        'global_pub = pubsub_package.global:main',
        'local_pub = pubsub_package.local:main',
        'sim2d = pubsub_package.sim2d:main',
        ],
    },
)
