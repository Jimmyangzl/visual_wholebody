from launch import LaunchDescription
from launch_ros.actions import Node
import os
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():
    pkg_path = get_package_share_directory('b2z1_description')
    # urdf_file = os.path.join(pkg_path, 'urdf', 'b2z1.urdf')
    urdf_file = os.path.join(pkg_path, 'urdf', 'b1z1.urdf')
    with open(urdf_file, 'r') as f:
        robot_desc = f.read()
    return LaunchDescription([
        Node(
            package='robot_state_publisher',
            executable='robot_state_publisher',
            name='robot_state_publisher',
            output='screen',
            arguments=[{'use_sim_time': False}],
            parameters=[{'robot_description': robot_desc}]
        ),
        Node(
            package='joint_state_publisher',
            executable='joint_state_publisher',
            name='joint_state_publisher',
            output='screen'
        ),
        Node(
            package='rviz2',
            executable='rviz2',
            name='rviz2',
            output='screen'
        )
    ])