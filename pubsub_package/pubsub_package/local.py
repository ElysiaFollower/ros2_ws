#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rclpy
from rclpy.node import Node
from rclpy.duration import Duration
from tf2_ros import TransformListener, Buffer, TransformException
from geometry_msgs.msg import PoseStamped, Twist, Point
from nav_msgs.msg import Path
from sensor_msgs.msg import LaserScan
from visualization_msgs.msg import Marker

import math
import numpy as np
import threading

# Import Planner
from pubsub_package.planner.dwa import DWA

class LocalPlanner(Node):
    def __init__(self):
        super().__init__('local_planner')
        
        # Parameters
        self.declare_parameter('base_frame', 'base_footprint')
        self.declare_parameter('laser_frame', 'laser_link') # 重要：知道雷达在哪里
        self.declare_parameter('control_freq', 10.0)
        
        self.base_frame = self.get_parameter('base_frame').value
        self.laser_frame = self.get_parameter('laser_frame').value
        self.freq = self.get_parameter('control_freq').value

        # TF
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # Subs/Pubs
        self.path_sub = self.create_subscription(Path, '/course_agv/global_path', self.path_cb, 10)
        self.scan_sub = self.create_subscription(LaserScan, '/scan', self.scan_cb, 10) # 也可以是 /course_agv/laser/scan
        self.vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.vis_pub = self.create_publisher(Marker, '/local_planner/debug_obs', 10)

        # Logic
        self.global_path = None
        self.scan_data = None
        self.lock = threading.Lock()
        
        self.planner = DWA()
        self.planner.config(max_speed=0.5, max_yawrate=1.0, base=0.25)

        # Control Loop
        self.timer = self.create_timer(1.0/self.freq, self.control_loop)
        self.get_logger().info("Local Planner Initialized (TF-Aware)")

    def path_cb(self, msg):
        with self.lock:
            self.global_path = msg
            self.get_logger().info("Received Global Path")

    def scan_cb(self, msg):
        with self.lock:
            self.scan_data = msg

    def get_transform(self, target, source):
        try:
            # Look up latest available transform
            return self.tf_buffer.lookup_transform(target, source, rclpy.time.Time())
        except TransformException:
            return None

    def control_loop(self):
        # 1. Check prerequisites
        if self.global_path is None or len(self.global_path.poses) == 0:
            return # No path, do nothing (idle)
            
        if self.scan_data is None:
            self.get_logger().warn("No scan data", throttle_duration_sec=2.0)
            self.publish_stop()
            return

        # 2. Get Robot Pose in Map (to find local goal)
        t_map_base = self.get_transform('map', self.base_frame)
        if not t_map_base:
            self.get_logger().warn("TF map->base missing")
            return
        
        rx = t_map_base.transform.translation.x
        ry = t_map_base.transform.translation.y
        ryaw = self.quat_to_yaw(t_map_base.transform.rotation)

        # 3. Find Lookahead Goal on Global Path
        goal_x, goal_y = self.get_local_goal(rx, ry, lookahead=1.0)
        
        # Transform Goal to Robot Frame (required for DWA)
        # Coordinate shift: (gx - rx, gy - ry) then rotate by -ryaw
        dx = goal_x - rx
        dy = goal_y - ry
        local_gx = dx * math.cos(ryaw) + dy * math.sin(ryaw)
        local_gy = -dx * math.sin(ryaw) + dy * math.cos(ryaw)

        # 4. Process Obstacles (CRITICAL: Transform Scan to Base Frame)
        obstacles = self.process_scan_to_base_frame(self.scan_data)
        
        # 5. Plan
        # Current velocity assumption (could verify from odom)
        v_curr = 0.0 # simplified
        w_curr = 0.0
        
        v, w = self.planner.planning(
            pose=[0,0,0], # Robot is origin of itself
            velocity=[v_curr, w_curr],
            goal=[local_gx, local_gy],
            obstacles=obstacles
        )
        
        # 6. Execute
        cmd = Twist()
        cmd.linear.x = float(v)
        cmd.angular.z = float(w)
        self.vel_pub.publish(cmd)
        
        # Debug Visualization of obstacles seen by planner
        self.pub_debug_obs(obstacles)

    def process_scan_to_base_frame(self, scan_msg):
        """
        Convert LaserScan (ranges) to (x,y) points in base_footprint frame.
        Handles TF offset properly.
        """
        # 1. Polar to Cartesian in Laser Frame
        angles = np.arange(scan_msg.angle_min, scan_msg.angle_max, scan_msg.angle_increment)
        # Truncate angles to match ranges length if needed
        if len(angles) > len(scan_msg.ranges): angles = angles[:len(scan_msg.ranges)]
        
        ranges = np.array(scan_msg.ranges)
        # Filter invalid ranges
        valid_mask = (ranges > scan_msg.range_min) & (ranges < scan_msg.range_max)
        
        valid_ranges = ranges[valid_mask]
        valid_angles = angles[valid_mask]
        
        lx = valid_ranges * np.cos(valid_angles)
        ly = valid_ranges * np.sin(valid_angles)
        
        # 2. Transform from Laser Frame to Base Frame
        t_base_laser = self.get_transform(self.base_frame, scan_msg.header.frame_id)
        if not t_base_laser:
            # Fallback: Assume Identity (dangerous but keeps running)
            return np.column_stack((lx, ly))
            
        # Apply transform
        tx = t_base_laser.transform.translation.x
        ty = t_base_laser.transform.translation.y
        q = t_base_laser.transform.rotation
        yaw = self.quat_to_yaw(q)
        
        # Rotation + Translation
        # x_base = x_laser * cos(yaw) - y_laser * sin(yaw) + tx
        bx = lx * math.cos(yaw) - ly * math.sin(yaw) + tx
        by = lx * math.sin(yaw) + ly * math.cos(yaw) + ty
        
        return np.column_stack((bx, by)).tolist()

    def get_local_goal(self, rx, ry, lookahead):
        # Find nearest point index
        poses = self.global_path.poses
        min_d = float('inf')
        idx = 0
        for i, p in enumerate(poses):
            d = math.hypot(p.pose.position.x - rx, p.pose.position.y - ry)
            if d < min_d:
                min_d = d
                idx = i
        
        # Look ahead
        goal_idx = idx
        dist_acc = 0
        while goal_idx < len(poses) - 1:
            p1 = poses[goal_idx].pose.position
            p2 = poses[goal_idx+1].pose.position
            d = math.hypot(p2.x - p1.x, p2.y - p1.y)
            dist_acc += d
            if dist_acc > lookahead:
                break
            goal_idx += 1
            
        gx = poses[goal_idx].pose.position.x
        gy = poses[goal_idx].pose.position.y
        
        # If reached end
        dist_to_final = math.hypot(poses[-1].pose.position.x - rx, poses[-1].pose.position.y - ry)
        if dist_to_final < 0.2:
            self.publish_stop()
            self.global_path = None # Done
            self.get_logger().info("Goal Reached!")
            
        return gx, gy

    def publish_stop(self):
        self.vel_pub.publish(Twist())

    def pub_debug_obs(self, obs):
        if not obs: return
        m = Marker()
        m.header.frame_id = self.base_frame
        m.header.stamp = self.get_clock().now().to_msg()
        m.type = Marker.POINTS
        m.action = Marker.ADD
        m.scale.x = 0.05
        m.scale.y = 0.05
        m.color.a = 1.0
        m.color.r = 1.0
        for (x,y) in obs:
            p = Point()
            p.x = float(x); p.y = float(y)
            m.points.append(p)
        self.vis_pub.publish(m)

    @staticmethod
    def quat_to_yaw(q):
        # yaw (z-axis rotation)
        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        return math.atan2(siny_cosp, cosy_cosp)

def main(args=None):
    rclpy.init(args=args)
    node = LocalPlanner()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()