#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rclpy
from rclpy.node import Node
from tf2_ros import TransformListener, Buffer
from nav_msgs.srv import GetMap
from nav_msgs.msg import Path, OccupancyGrid
from geometry_msgs.msg import PoseStamped

import numpy as np
import sys
import time
import math
# import your own planner
from pubsub_package.planner.rrt_star import RRT_star as planner


class GlobalPlanner(Node):
    def __init__(self):
        super().__init__('global_planner')
        self.declare_parameter('map_frame', 'map')
        self.declare_parameter('base_frame', 'base_footprint')
        self.declare_parameter('log_interval_sec', 1.0)
        self.declare_parameter('robot_radius', 0.2)
        self.declare_parameter('inflation_radius', 0.1)
        self.map_frame = str(self.get_parameter('map_frame').value)
        self.base_frame = str(self.get_parameter('base_frame').value)
        self.log_interval_sec = float(self.get_parameter('log_interval_sec').value)
        self._last_log_time = 0.0

        self.plan_robot_radius = float(self.get_parameter('robot_radius').value)
        self.inflation_radius = float(self.get_parameter('inflation_radius').value)
        self.plan_ox = []  # obstacle
        self.plan_oy = []
        self.plan_sx = 0.0  # start pose
        self.plan_sy = 0.0
        self.plan_gx = 0.0  # goal pose
        self.plan_gy = 0.0
        self.plan_rx = []  # plan
        self.plan_ry = []

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.map_sub = self.create_subscription(OccupancyGrid, '/map', self.map_callback, 1)
        self.goal_sub = self.create_subscription(PoseStamped, '/course_agv/goal', self.goal_callback, 1)
        self.path_pub = self.create_publisher(Path, '/course_agv/global_path', 1)
        # Note: Service-based planning is disabled, using subscription-based approach instead
        # self.plan_srv = self.create_service(Plan, '/course_agv/global_plan', self.replan)

    def update_map(self):
        """Request map from map server via service (alternative to subscription).

        Note: This method is currently not used as map is received via subscription.
        Kept for potential future use.
        """
        client = self.create_client(GetMap, '/static_map')
        while not client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Waiting for /static_map service...')
        try:
            request = GetMap.Request()
            future = client.call_async(request)
            rclpy.spin_until_future_complete(self, future)
            if future.result():
                self.map_callback(future.result().map)
            else:
                self.get_logger().error('Failed to get map')
        except Exception as e:
            self.get_logger().error(f'Service call failed: {e}')

    def map_callback(self, msg):
        self.map = msg
        now = time.time()
        if now - self._last_log_time >= max(self.log_interval_sec, 0.1):
            self._last_log_time = now
            self.get_logger().info(
                f"Received /map: size=({msg.info.width}x{msg.info.height}) res={msg.info.resolution:.3f} "
                f"origin=({msg.info.origin.position.x:.2f},{msg.info.origin.position.y:.2f})"
            )

    def goal_callback(self, msg):
        self.plan_gx = msg.pose.position.x
        self.plan_gy = msg.pose.position.y
        self.get_logger().info(
            f"Received new goal: ({self.plan_gx:.3f},{self.plan_gy:.3f}) frame={msg.header.frame_id or self.map_frame}"
        )
        self.replan()

    def replan(self):
        self.get_logger().info('Replanning...')
        if not hasattr(self, "map"):
            self.get_logger().warning("No /map received yet, skipping planning")
            return
        self.init_planner()
        self.update_global_pose()

        res = False
        attempt = 10
        for _ in range(attempt):
            is_found, path = self.planner.plan(self.plan_sx, self.plan_sy, self.plan_gx, self.plan_gy)
            if is_found:
                path = self.simplify_path(path, min_dist=self.map.info.resolution * 0.5)
                if hasattr(self.planner, "smooth_path"):
                    path = self.planner.smooth_path(path, max_iter=200)
                path = self.simplify_path(path, min_dist=self.map.info.resolution * 0.5)
                plan_path = self.insert_midpoints(path)
                self.plan_rx = np.array(plan_path)[:, 0]
                self.plan_ry = np.array(plan_path)[:, 1]
                self.publish_path()
                res = True
                self.get_logger().info(
                    f"Path found: raw_len={len(path)}, published_len={len(self.plan_rx)} "
                    f"start=({self.plan_sx:.2f},{self.plan_sy:.2f}) goal=({self.plan_gx:.2f},{self.plan_gy:.2f})"
                )
                break
            else:
                self.get_logger().info("Retry...")

        if not res:
            self.get_logger().error("Path not found!")



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
        plan_path = []
        for i in range(len(path) - 1):
            plan_path.append(path[i])
            x, y = path[i]
            dx = path[i + 1][0] - x
            dy = path[i + 1][1] - y
            angle = math.atan2(dy, dx)
            dist = math.hypot(dx, dy)
            step_size = self.plan_robot_radius
            steps = int(round(dist / step_size))
            for _ in range(steps - 1):
                x += step_size * math.cos(angle)
                y += step_size * math.sin(angle)
                plan_path.append((x, y))
        plan_path.append(path[-1])
        return plan_path

    @staticmethod
    def inflate_obstacles(occ_grid: np.ndarray, radius_cells: int) -> np.ndarray:
        """Inflate occupancy grid (binary) by a circular radius in cells.

        Args:
            occ_grid: bool array, True means occupied
            radius_cells: inflation radius in grid cells

        Returns:
            Inflated bool array
        """
        r = int(max(radius_cells, 0))
        if r == 0:
            return occ_grid

        inflated = np.zeros_like(occ_grid, dtype=bool)
        offsets = []
        r2 = r * r
        for dx in range(-r, r + 1):
            for dy in range(-r, r + 1):
                if dx * dx + dy * dy <= r2:
                    offsets.append((dx, dy))

        w, h = occ_grid.shape
        for dx, dy in offsets:
            xs_src_start = max(0, -dx)
            xs_src_end = min(w, w - dx)
            ys_src_start = max(0, -dy)
            ys_src_end = min(h, h - dy)

            xs_dst_start = xs_src_start + dx
            xs_dst_end = xs_src_end + dx
            ys_dst_start = ys_src_start + dy
            ys_dst_end = ys_src_end + dy

            inflated[xs_dst_start:xs_dst_end, ys_dst_start:ys_dst_end] |= occ_grid[
                xs_src_start:xs_src_end, ys_src_start:ys_src_end
            ]

        return inflated

    def init_planner(self):
      
        map_data = np.array(self.map.data).reshape((self.map.info.height, -1)).transpose()  # 实物
      
        minx = self.map.info.origin.position.x
        maxx = self.map.info.origin.position.x + map_data.shape[0] * self.map.info.resolution
        miny = self.map.info.origin.position.y
        maxy = self.map.info.origin.position.y + map_data.shape[1] * self.map.info.resolution
      
        resolution = float(self.map.info.resolution)
        occ = map_data != 0  # treat unknown as occupied for safety
        inflation_cells = int(math.ceil(max(self.plan_robot_radius + self.inflation_radius, 0.0) / max(resolution, 1e-6)))
        occ_inflated = self.inflate_obstacles(occ, inflation_cells)

        ox, oy = np.nonzero(occ_inflated)  # 实物
    
        self.plan_ox = ox * self.map.info.resolution + self.map.info.origin.position.x
        self.plan_oy = oy * self.map.info.resolution + self.map.info.origin.position.y
        obstacles = list(zip(self.plan_ox, self.plan_oy))
        obstacles.append((-9999, -9999))
        self.get_logger().info(
            f"Planner init: bounds=({minx:.2f},{miny:.2f})-({maxx:.2f},{maxy:.2f}) obstacles={len(obstacles)} "
            f"robot_radius={self.plan_robot_radius:.2f} inflation={self.inflation_radius:.2f} (cells={inflation_cells})"
        )
        self.planner = planner(minx=minx, maxx=maxx, miny=miny, maxy=maxy, obstacles=obstacles,
                               robot_size=self.plan_robot_radius, safe_dist=self.map.info.resolution)

    def update_global_pose(self):
        try:
          
            trans = self.tf_buffer.lookup_transform(
                self.map_frame,
                self.base_frame,
                rclpy.time.Time(),
                rclpy.duration.Duration(seconds=4.0),
            )
            
            self.plan_sx = trans.transform.translation.x
            self.plan_sy = trans.transform.translation.y
        except Exception as e:
            self.get_logger().error(f"TF error: {e}")

    def publish_path(self):
        path = Path()
        path.header.stamp = rclpy.time.Time().to_msg()
        path.header.frame_id = self.map_frame
        for i in range(len(self.plan_rx)):
            pose = PoseStamped()
            pose.header.stamp = rclpy.time.Time().to_msg()
            pose.header.frame_id = self.map_frame
            pose.pose.position.x = self.plan_rx[i]
            pose.pose.position.y = self.plan_ry[i]
            pose.pose.position.z = 0.01
            pose.pose.orientation.w = 1.0
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
