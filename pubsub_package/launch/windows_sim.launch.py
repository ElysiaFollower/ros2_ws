from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    default_map = PathJoinSubstitution(
        [FindPackageShare("pubsub_package"), "maps", "simple_map.yaml"]
    )

    map_yaml = LaunchConfiguration("map")
    real = LaunchConfiguration("real")

    return LaunchDescription(
        [
            DeclareLaunchArgument("map", default_value=default_map),
            DeclareLaunchArgument("real", default_value="false"),
            Node(
                package="pubsub_package",
                executable="sim2d",
                name="sim2d",
                output="screen",
                parameters=[{"map_yaml": map_yaml}],
            ),
            Node(
                package="pubsub_package",
                executable="global_pub",
                name="global_planner",
                output="screen",
            ),
            Node(
                package="pubsub_package",
                executable="local_pub",
                name="local_planner",
                output="screen",
                parameters=[{"real": ParameterValue(real, value_type=bool)}],
            ),
        ]
    )
