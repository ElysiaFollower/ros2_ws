#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
from tf2_ros import TransformListener, Buffer
from nav_msgs.msg import Path, OccupancyGrid
from geometry_msgs.msg import PoseStamped
import numpy as np
import math
import time

# Import the refactored planner
from pubsub_package.planner.rrt_star import RRT_star

class GlobalPlanner(Node):
    def __init__(self):
        super().__init__('global_planner')
        
        # Parameters
        self.declare_parameter('map_frame', 'map')
        self.declare_parameter('base_frame', 'base_footprint')
        self.declare_parameter('robot_radius', 0.25)
        self.declare_parameter('inflation_radius', 0.15)
        
        self.map_frame = self.get_parameter('map_frame').value
        self.base_frame = self.get_parameter('base_frame').value
        self.robot_radius = self.get_parameter('robot_radius').value
        self.inflation_radius = self.get_parameter('inflation_radius').value

        # TF
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # Communication
        qos_map = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.map_sub = self.create_subscription(OccupancyGrid, '/map', self.map_callback, qos_map)
        self.goal_sub = self.create_subscription(PoseStamped, '/course_agv/goal', self.goal_callback, 10)
        self.path_pub = self.create_publisher(Path, '/course_agv/global_path', 10)

        # State
        self.map_data = None
        self.map_info = None
        self.current_goal = None

        self.get_logger().info("Global Planner Initialized (Grid-based RRT*)")

    def map_callback(self, msg):
        self.get_logger().info(f"Received Map: {msg.info.width}x{msg.info.height}")
        self.map_info = msg.info
        
        # Convert to numpy array for fast access (Row-major)
        # 0: Free, 100: Occupied, -1: Unknown
        raw_data = np.array(msg.data, dtype=np.int8).reshape(msg.info.height, msg.info.width)
        
        # Inflate obstacles (Simple morphological dilation can be done here if needed)
        # For simplicity in python, we assume the RRT checks radius or we use a pre-inflated map
        # Here we just store the raw map, and RRT* checks "is_collision" with a safety margin
        self.map_data = raw_data

    def goal_callback(self, msg):
        self.current_goal = msg
        self.get_logger().info(f"New Goal Received: ({msg.pose.position.x:.2f}, {msg.pose.position.y:.2f})")
        self.plan_path()

    def get_robot_pose(self):
        try:
            trans = self.tf_buffer.lookup_transform(
                self.map_frame, 
                self.base_frame, 
                rclpy.time.Time(),
                rclpy.duration.Duration(seconds=0.5) # Non-blocking short timeout
            )
            return trans.transform.translation.x, trans.transform.translation.y
        except Exception as e:
            self.get_logger().warn(f"Could not get robot pose: {e}")
            return None

    def plan_path(self):
        if self.map_data is None:
            self.get_logger().warn("Map not received yet, cannot plan.")
            return

        start_pose = self.get_robot_pose()
        if start_pose is None:
            return

        sx, sy = start_pose
        gx, gy = self.current_goal.pose.position.x, self.current_goal.pose.position.y

        # Initialize RRT* with the grid directly
        planner = RRT_star(
            grid_map=self.map_data,
            resolution=self.map_info.resolution,
            origin_x=self.map_info.origin.position.x,
            origin_y=self.map_info.origin.position.y,
            robot_radius=self.robot_radius,
            safe_dist=self.inflation_radius,
            max_iter=3000
        )

        self.get_logger().info(f"Start Planning: ({sx:.1f},{sy:.1f}) -> ({gx:.1f},{gy:.1f})")
        start_time = time.time()
        
        found, path = planner.plan(sx, sy, gx, gy)
        
        if found:
            # Smooth path
            path = planner.smooth_path(path)
            self.publish_path(path)
            duration = time.time() - start_time
            self.get_logger().info(f"Path Found! Length: {len(path)}. Time: {duration:.2f}s")
        else:
            self.get_logger().error("Path Not Found!")

    def publish_path(self, points):
        path_msg = Path()
        path_msg.header.stamp = self.get_clock().now().to_msg()
        path_msg.header.frame_id = self.map_frame
        
        for pt in points:
            pose = PoseStamped()
            pose.header = path_msg.header
            pose.pose.position.x = pt[0]
            pose.pose.position.y = pt[1]
            pose.pose.orientation.w = 1.0
            path_msg.poses.append(pose)
            
        self.path_pub.publish(path_msg)

def main(args=None):
    rclpy.init(args=args)
    node = GlobalPlanner()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()