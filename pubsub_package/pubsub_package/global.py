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
        self.declare_parameter('robot_radius', 0.01)
        self.declare_parameter('inflation_radius', 0.05)
        
        self.map_frame = self.get_parameter('map_frame').value
        self.base_frame = self.get_parameter('base_frame').value
        self.robot_radius = self.get_parameter('robot_radius').value
        self.inflation_radius = self.get_parameter('inflation_radius').value

        # TF
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # Communication
        # Transient Local QoS for Map is critical
        qos_map = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL, reliability=ReliabilityPolicy.RELIABLE)
        self.map_sub = self.create_subscription(OccupancyGrid, '/map', self.map_callback, qos_map)
        self.goal_sub = self.create_subscription(PoseStamped, '/course_agv/goal', self.goal_callback, 10)
        self.path_pub = self.create_publisher(Path, '/course_agv/global_path', 10)

        # State
        self.map_data = None
        self.map_info = None
        self.current_goal = None

        self.get_logger().info(">>> Global Planner Ready. Waiting for Map...")

    def map_callback(self, msg):
        self.get_logger().info(f"[Global] Map Received: {msg.info.width}x{msg.info.height}, Res: {msg.info.resolution}")
        self.map_info = msg.info
        
        # Convert to numpy array for fast access
        # Data is row-major
        raw_data = np.array(msg.data, dtype=np.int8).reshape(msg.info.height, msg.info.width)
        self.map_data = raw_data

        if self.publish_inflated_map:
            try:
                self._publish_inflated_map()
            except Exception as e:
                self.get_logger().error(f"Inflated map publish error: {e}")

    def goal_callback(self, msg):
        self.current_goal = msg
        self.get_logger().info(f"[Global] New Goal: ({msg.pose.position.x:.2f}, {msg.pose.position.y:.2f})")
        self.plan_path()

    def get_robot_pose(self):
        try:
            trans = self.tf_buffer.lookup_transform(
                self.map_frame, 
                self.base_frame, 
                rclpy.time.Time(),
                rclpy.duration.Duration(seconds=1.0)
            )
            return trans.transform.translation.x, trans.transform.translation.y
        except Exception as e:
            self.get_logger().error(f"[Global] TF Error (Map->Base): {e}")
            return None

    @staticmethod
    def interpolate_path(path, step_size=0.1):
        """
        Interpolate sparse path points to create a dense path.
        Args:
            path: [[x1, y1], [x2, y2], ...]
            step_size: interpolation interval (m)
        """
        if len(path) < 2:
            return path
            
        dense_path = []
        for i in range(len(path) - 1):
            p1 = np.array(path[i])
            p2 = np.array(path[i+1])
            dist = np.linalg.norm(p2 - p1)
            
            num_steps = int(math.ceil(dist / step_size))
            if num_steps == 0: num_steps = 1
            
            for j in range(num_steps):
                t = j / num_steps
                pt = p1 + (p2 - p1) * t
                dense_path.append(pt.tolist())
        
        dense_path.append(path[-1])
        return dense_path

    def plan_path(self):
        if self.map_data is None:
            self.get_logger().warn("[Global] Cannot plan: No Map data yet.")
            return

        start_pose = self.get_robot_pose()
        if start_pose is None:
            return

        sx, sy = start_pose
        gx, gy = self.current_goal.pose.position.x, self.current_goal.pose.position.y

        # Initialize RRT* with grid map
        planner = RRT_star(
            grid_map=self.map_data,
            resolution=self.map_info.resolution,
            origin_x=self.map_info.origin.position.x,
            origin_y=self.map_info.origin.position.y,
            robot_radius=self.robot_radius,
            safe_dist=self.inflation_radius,
            expand_dis=1.0, # Increased for speed
            max_iter=2000   # Reduced for speed (was 5000)
        )

        self.get_logger().info(f"[Global] Start RRT*: ({sx:.2f},{sy:.2f}) -> ({gx:.2f},{gy:.2f})")
        start_time = time.time()
        
        try:
            found, path = planner.plan(sx, sy, gx, gy)
        except Exception as e:
            self.get_logger().error(f"[Global] RRT* Exception: {e}")
            return
        
        if found:
            # 1. Smooth the path (removes unnecessary waypoints)
            path = planner.smooth_path(path)
            # 2. Interpolate (adds dense points back) - FIXES PATH LEN: 2 ISSUE
            path = self.interpolate_path(path, step_size=0.05)
            
            self.publish_path(path)
            duration = time.time() - start_time
            self.get_logger().info(f"[Global] Success! Path len: {len(path)}. Time: {duration:.3f}s")
        else:
            self.get_logger().error("[Global] Path Not Found!")

    def publish_path(self, points):
        path_msg = Path()
        path_msg.header.stamp = self.get_clock().now().to_msg()
        path_msg.header.frame_id = self.map_frame
        
        for pt in points:
            pose = PoseStamped()
            pose.header = path_msg.header
            pose.pose.position.x = pt[0]
            pose.pose.position.y = pt[1]
            # Lift path slightly so it shows above the map in RViz
            pose.pose.position.z = 0.1 
            pose.pose.orientation.w = 1.0
            path_msg.poses.append(pose)
            
        self.path_pub.publish(path_msg)

def main(args=None):
    rclpy.init(args=args)
    node = GlobalPlanner()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()