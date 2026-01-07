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
from pubsub_package.param_panel import ParamSpec, ParamWebPanel

class LocalPlanner(Node):
    def __init__(self):
        super().__init__('local_planner')
        
        # Parameters
        self.declare_parameter('base_frame', 'base_footprint')
        self.declare_parameter('laser_frame', 'laser_link') 
        self.declare_parameter('control_freq', 10.0)
        
        self.base_frame = self.get_parameter('base_frame').value
        self.laser_frame = self.get_parameter('laser_frame').value
        self.freq = self.get_parameter('control_freq').value

        # TF
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # Subs/Pubs
        self.path_sub = self.create_subscription(Path, '/course_agv/global_path', self.path_cb, 10)
        self.scan_sub = self.create_subscription(LaserScan, '/scan', self.scan_cb, 10)
        self.vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        # Debug publisher for obstacles in RViz
        self.vis_pub = self.create_publisher(Marker, '/local_planner/debug_obs', 10)
        # Trajectory publisher for DWA viz
        self.traj_pub = self.create_publisher(Path, '/course_agv/trajectory', 10)

        # Logic
        self.global_path = None
        self.scan_data = None
        self.lock = threading.Lock()
        
        self.planner = DWA()
        self.planner.config(
            max_speed=0.3,      # Safer speed
            max_yawrate=0.8, 
            base=0.25,          # Robot radius
            to_goal_cost_gain=0.8, 
            obstacle_cost_gain=2.0
        )

        # Control Loop
        self.timer = self.create_timer(1.0/self.freq, self.control_loop)
        self.get_logger().info(f">>> Local Planner Started. Frame: {self.base_frame} <-> {self.laser_frame}")

    def path_cb(self, msg):
        with self.lock:
            self.global_path = msg
            self.get_logger().info(f"[Local] Global Path Updated. Len: {len(msg.poses)}")

    def scan_cb(self, msg):
        with self.lock:
            self.scan_data = msg

    def get_transform(self, target, source):
        try:
            return self.tf_buffer.lookup_transform(target, source, rclpy.time.Time())
        except TransformException:
            return None

    def control_loop(self):
        # 1. Check prerequisites
        if self.global_path is None or len(self.global_path.poses) == 0:
            return # Idle
            
        if self.scan_data is None:
            self.get_logger().warn("Waiting for LaserScan...", throttle_duration_sec=2.0)
            self.publish_stop()
            return

        # 2. Get Robot Pose in Map
        t_map_base = self.get_transform('map', self.base_frame)
        if not t_map_base:
            self.get_logger().warn(f"TF map->{self.base_frame} missing", throttle_duration_sec=2.0)
            self.publish_stop()
            return
        
        rx = t_map_base.transform.translation.x
        ry = t_map_base.transform.translation.y
        ryaw = self.quat_to_yaw(t_map_base.transform.rotation)

        # 3. Find Lookahead Goal
        goal_x, goal_y, reached = self.get_local_goal(rx, ry, lookahead=0.8)
        
        if reached:
            self.get_logger().info("Goal Reached!")
            self.publish_stop()
            with self.lock:
                self.global_path = None
            return

        # Transform Goal to Robot Frame
        dx = goal_x - rx
        dy = goal_y - ry
        local_gx = dx * math.cos(ryaw) + dy * math.sin(ryaw)
        local_gy = -dx * math.sin(ryaw) + dy * math.cos(ryaw)

        # 4. Process Obstacles (Laser -> Base)
        obstacles = self.process_scan_to_base_frame(self.scan_data)
        if obstacles is None:
            self.publish_stop() # TF Error
            return
        
        # 5. Plan
        v, w = self.planner.planning(
            pose=[0,0,0], 
            velocity=[0, 0], # Simplified start velocity
            goal=[local_gx, local_gy],
            obstacles=obstacles
        )
        
        # 6. Execute
        cmd = Twist()
        cmd.linear.x = float(v)
        cmd.angular.z = float(w)
        self.vel_pub.publish(cmd)
        
        # 7. Visualize
        self.pub_debug_obs(obstacles)

    def process_scan_to_base_frame(self, scan_msg):
        angles = np.arange(scan_msg.angle_min, scan_msg.angle_max, scan_msg.angle_increment)
        if len(angles) > len(scan_msg.ranges): angles = angles[:len(scan_msg.ranges)]
        
        ranges = np.array(scan_msg.ranges)
        # Filter: Valid range and not too far (ignore walls 10m away)
        valid_mask = (ranges > scan_msg.range_min) & (ranges < 4.0)
        
        if not np.any(valid_mask):
            return []

        valid_ranges = ranges[valid_mask]
        valid_angles = angles[valid_mask]
        
        lx = valid_ranges * np.cos(valid_angles)
        ly = valid_ranges * np.sin(valid_angles)
        
        t_base_laser = self.get_transform(self.base_frame, scan_msg.header.frame_id)
        if not t_base_laser:
            self.get_logger().error(f"TF Error: {self.base_frame} -> {scan_msg.header.frame_id}")
            return None
            
        tx = t_base_laser.transform.translation.x
        ty = t_base_laser.transform.translation.y
        q = t_base_laser.transform.rotation
        yaw = self.quat_to_yaw(q)
        
        bx = lx * math.cos(yaw) - ly * math.sin(yaw) + tx
        by = lx * math.sin(yaw) + ly * math.cos(yaw) + ty
        
        return np.column_stack((bx, by)).tolist()

    def get_local_goal(self, rx, ry, lookahead):
        poses = self.global_path.poses
        # Find nearest point index on path
        # Optimized: just search neighborhood of previous index if available, 
        # but for robustness we search all (len is usually < 500)
        dists = [(p.pose.position.x - rx)**2 + (p.pose.position.y - ry)**2 for p in poses]
        min_idx = np.argmin(dists)
        
        # Check if near end
        end_dist = math.hypot(poses[-1].pose.position.x - rx, poses[-1].pose.position.y - ry)
        if end_dist < 0.2:
            return 0, 0, True

        # Look ahead
        goal_idx = min_idx
        curr_dist = 0
        while goal_idx < len(poses) - 1:
            p1 = poses[goal_idx].pose.position
            p2 = poses[goal_idx+1].pose.position
            d = math.hypot(p2.x - p1.x, p2.y - p1.y)
            curr_dist += d
            if curr_dist > lookahead:
                break
            goal_idx += 1
            
        return poses[goal_idx].pose.position.x, poses[goal_idx].pose.position.y, False

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
        m.color.r = 1.0 # Red
        for (x,y) in obs:
            p = Point()
            p.x = float(x); p.y = float(y)
            m.points.append(p)
        self.vis_pub.publish(m)

    @staticmethod
    def quat_to_yaw(q):
        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        return math.atan2(siny_cosp, cosy_cosp)

def main(args=None):
    rclpy.init(args=args)
    node = LocalPlanner()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()