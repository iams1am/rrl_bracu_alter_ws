from setuptools import find_packages, setup

package_name = 'csv_logger'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='sbuntu',
    maintainer_email='siam.abdullah@g.bracu.ac.bd',
    description='Log localized objects to RoboCup CSV format.',
    license='MIT',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'csv_logger = csv_logger.logger_node:main',
        ],
    },
)
