#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rclpy
from rclpy.node import Node
from tf2_ros import TransformListener, Buffer
from nav_msgs.srv import GetMap
from nav_msgs.msg import Path, OccupancyGrid
# 修正：PoseStamped 必须从 geometry_msgs 导入
from geometry_msgs.msg import PoseStamped 
import numpy as np
import math

# import your own planner
from pubsub_package.planner.rrt_star import RRT_star as planner


class GlobalPlanner(Node):
    def __init__(self):
        super().__init__('global_planner')
        self.plan_robot_radius = 0.15
        self.plan_ox = []  # obstacle x
        self.plan_oy = []  # obstacle y
        self.plan_sx = 0.0  # start pose x
        self.plan_sy = 0.0
        self.plan_gx = 0.0  # goal pose x
        self.plan_gy = 0.0
        self.plan_rx = []  # plan result x
        self.plan_ry = []

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.map_sub = self.create_subscription(OccupancyGrid, '/map', self.map_callback, 1)
        self.goal_sub = self.create_subscription(PoseStamped, '/course_agv/goal', self.goal_callback, 1)
        self.path_pub = self.create_publisher(Path, '/course_agv/global_path', 1)
        
        self.map = None

    def map_callback(self, msg):
        self.map = msg
        self.get_logger().info("Map received.")

    def goal_callback(self, msg):
        if self.map is None:
            self.get_logger().warn("Map not received yet, cannot plan.")
            return

        self.plan_gx = msg.pose.position.x
        self.plan_gy = msg.pose.position.y
        self.get_logger().info(f"Received new goal: ({self.plan_gx:.2f}, {self.plan_gy:.2f})")
        self.replan()

    def replan(self):
        self.get_logger().info('Replanning...')
        
        # 更新起点位置
        if not self.update_global_pose():
            self.get_logger().error("Could not get robot pose, aborting plan.")
            return

        self.init_planner()

        res = False
        attempt = 5 # 减少尝试次数，避免阻塞太久
        for i in range(attempt):
            is_found, path = self.planner.plan(self.plan_sx, self.plan_sy, self.plan_gx, self.plan_gy)
            if is_found:
                # 路径后处理：简化 -> 平滑 -> 插值
                path = self.simplify_path(path, min_dist=self.map.info.resolution * 0.5)
                if hasattr(self.planner, "smooth_path"):
                    path = self.planner.smooth_path(path, max_iter=100)
                path = self.simplify_path(path, min_dist=self.map.info.resolution * 0.5)
                plan_path = self.insert_midpoints(path)
                
                # 检查路径是否为空，避免数组索引错误
                if not plan_path or len(plan_path) == 0:
                    self.get_logger().warn(f"Path is empty after processing, retrying...")
                    continue
                
                plan_array = np.array(plan_path)
                if plan_array.shape[0] == 0 or plan_array.shape[1] < 2:
                    self.get_logger().warn(f"Invalid path shape: {plan_array.shape}, retrying...")
                    continue
                
                self.plan_rx = plan_array[:, 0]
                self.plan_ry = plan_array[:, 1]
                self.publish_path()
                res = True
                self.get_logger().info(f"Path found on attempt {i+1}!")
                break
            else:
                self.get_logger().info(f"RRT search failed attempt {i+1}...")

        if not res:
            self.get_logger().error("Path not found after all attempts!")

    @staticmethod
    def simplify_path(path, min_dist=0.05):
        if not path:
            return []
        out = [path[0]]
        last_x, last_y = float(path[0][0]), float(path[0][1])
        for p in path[1:]:
            x, y = float(p[0]), float(p[1])
            if math.hypot(x - last_x, y - last_y) >= min_dist:
                out.append([x, y])
                last_x, last_y = x, y
        if len(out) == 1 and len(path) > 1:
            out.append([float(path[-1][0]), float(path[-1][1])])
        return out

    def insert_midpoints(self, path):
        if not path:
            return []
        plan_path = []
        for i in range(len(path) - 1):
            plan_path.append(path[i])
            x, y = path[i]
            dx = path[i + 1][0] - x
            dy = path[i + 1][1] - y
            angle = math.atan2(dy, dx)
            dist = math.hypot(dx, dy)
            step_size = self.plan_robot_radius
            if step_size <= 0: step_size = 0.1
            
            steps = int(dist / step_size)
            for _ in range(steps):
                x += step_size * math.cos(angle)
                y += step_size * math.sin(angle)
                plan_path.append((x, y))
        plan_path.append(path[-1])
        return plan_path

    def init_planner(self):
        # 注意：这里假设地图数据是 row-major 且需要转置才能对应 (x, y)
        # 如果发现障碍物位置不对，请尝试去掉 .transpose()
        map_data = np.array(self.map.data).reshape((self.map.info.height, -1)).transpose()
      
        minx = self.map.info.origin.position.x
        maxx = self.map.info.origin.position.x + map_data.shape[0] * self.map.info.resolution
        miny = self.map.info.origin.position.y
        maxy = self.map.info.origin.position.y + map_data.shape[1] * self.map.info.resolution
      
        ox, oy = np.nonzero(map_data > 50)  # 使用 >50 阈值确定障碍物，比 !=0 更稳健
    
        self.plan_ox = ox * self.map.info.resolution + self.map.info.origin.position.x
        self.plan_oy = oy * self.map.info.resolution + self.map.info.origin.position.y
        obstacles = list(zip(self.plan_ox, self.plan_oy))
        
        # 初始化 RRT*
        self.planner = planner(minx=minx, maxx=maxx, miny=miny, maxy=maxy, obstacles=obstacles,
                               robot_size=self.plan_robot_radius, safe_dist=self.map.info.resolution)

    def update_global_pose(self):
        try:
            # 使用 Time() 获取最新变换
            trans = self.tf_buffer.lookup_transform('map', 'base_footprint', rclpy.time.Time(), rclpy.duration.Duration(seconds=1.0))
            self.plan_sx = trans.transform.translation.x
            self.plan_sy = trans.transform.translation.y
            return True
        except Exception as e:
            self.get_logger().error(f"TF error: {e}")
            return False

    def publish_path(self):
        # 检查路径是否有效
        if len(self.plan_rx) == 0 or len(self.plan_ry) == 0:
            self.get_logger().warn("Cannot publish empty path!")
            return
        
        if len(self.plan_rx) != len(self.plan_ry):
            self.get_logger().error(f"Path arrays length mismatch: rx={len(self.plan_rx)}, ry={len(self.plan_ry)}")
            return
            
        path = Path()
        path.header.stamp = self.get_clock().now().to_msg()
        path.header.frame_id = 'map'
        for i in range(len(self.plan_rx)):
            pose = PoseStamped()
            pose.header.stamp = path.header.stamp
            pose.header.frame_id = 'map'
            pose.pose.position.x = float(self.plan_rx[i])
            pose.pose.position.y = float(self.plan_ry[i])
            pose.pose.position.z = 0.05 
            pose.pose.orientation.w = 1.0 # 关键：有效的四元数
            path.poses.append(pose)
        self.path_pub.publish(path)

def main(args=None):
    rclpy.init(args=args)
    global_planner = GlobalPlanner()
    rclpy.spin(global_planner)
    global_planner.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()