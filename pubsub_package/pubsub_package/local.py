#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import rclpy
from rclpy.node import Node
from tf2_ros import TransformListener, Buffer
from tf2_ros import TransformException
from geometry_msgs.msg import PoseStamped
import math
import time
import numpy as np
from nav_msgs.msg import Path
from geometry_msgs.msg import  Twist

from sensor_msgs.msg import LaserScan
from threading import Lock, Thread
# Import your own planner
from pubsub_package.planner.dwa import DWA

class LocalPlanner(Node):
    def __init__(self, real=None):
        super().__init__('local_planner')
        self.declare_parameter('real', True)
        self.declare_parameter('map_frame', 'map')
        self.declare_parameter('base_frame', 'base_footprint')
        self.declare_parameter('robot_radius', 0.2)
        self.declare_parameter('safety_dist', 0.05)
        self.declare_parameter('lookahead_dist', 0.8)
        self.declare_parameter('local_path_max_points', 30)
        self.declare_parameter('controller_frequency', 10.0)
        self.declare_parameter('rotate_to_goal', True)
        self.declare_parameter('yaw_goal_tolerance', 0.3)
        self.declare_parameter('log_interval_sec', 1.0)
        self.declare_parameter('dwa.to_goal_cost_gain', 0.5)
        self.declare_parameter('dwa.path_cost_gain', 2.0)
        self.declare_parameter('dwa.heading_cost_gain', 0.3)
        self.declare_parameter('dwa.obstacle_cost_gain', 1.0)
        self.declare_parameter('dwa.speed_cost_gain', 0.2)
        self.declare_parameter('dwa.predict_time', 2.0)
        self.declare_parameter('dwa.v_samples', 6)
        self.declare_parameter('dwa.w_samples', 12)
        if real is None:
            real = bool(self.get_parameter('real').value)
        self.real = real
        self.map_frame = str(self.get_parameter('map_frame').value)
        self.base_frame = str(self.get_parameter('base_frame').value)
        self.x = 0.0
        self.y = 0.0
        self.yaw = 0.0
        self.vx = 0.0
        self.vw = 0.0
        self.path = Path()
        self.arrive = 0.2  # Standard for arrival
        self.threshold = 1.5  # Laser threshold
        self.robot_size = float(self.get_parameter('robot_radius').value)
        self.safety_dist = float(self.get_parameter('safety_dist').value)
        self.V_X = 0.5
        self.V_W = 0.5

        self.planner = DWA()  # Initialize planner
        self.planner.config(max_speed=self.V_X, max_yawrate=self.V_W, base=self.robot_size + self.safety_dist)
        self.planner.to_goal_cost_gain = float(self.get_parameter('dwa.to_goal_cost_gain').value)
        self.planner.path_cost_gain = float(self.get_parameter('dwa.path_cost_gain').value)
        self.planner.heading_cost_gain = float(self.get_parameter('dwa.heading_cost_gain').value)
        self.planner.obstacle_cost_gain = float(self.get_parameter('dwa.obstacle_cost_gain').value)
        self.planner.speed_cost_gain = float(self.get_parameter('dwa.speed_cost_gain').value)
        self.planner.predict_time = float(self.get_parameter('dwa.predict_time').value)
        self.planner.v_samples = int(self.get_parameter('dwa.v_samples').value)
        self.planner.w_samples = int(self.get_parameter('dwa.w_samples').value)

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.path_sub = self.create_subscription(Path, '/course_agv/global_path', self.path_callback, 1)
        self.midpose_pub = self.create_publisher(PoseStamped, '/course_agv/mid_goal', 1)
        if self.real:
            self.laser_sub = self.create_subscription(LaserScan, '/scan', self.laser_callback, 1)  # Real robot
            self.vel_pub = self.create_publisher(Twist, '/cmd_vel', 1)  # Real robot
        else:
            self.laser_sub = self.create_subscription(LaserScan, '/course_agv/laser/scan', self.laser_callback, 1)  # Simulation
            self.vel_pub = self.create_publisher(Twist, '/course_agv/velocity', 1)  # Simulation

        self.planning_thread = None
        self.lock = Lock()
        self.laser_lock = Lock()

        self.traj_pub = self.create_publisher(Path, '/course_agv/trajectory', 1)
        self.traj = Path()
        self.traj.header.frame_id = self.map_frame

        self.lookahead_dist = float(self.get_parameter('lookahead_dist').value)
        self.local_path_max_points = int(self.get_parameter('local_path_max_points').value)
        self.controller_frequency = float(self.get_parameter('controller_frequency').value)
        self.rotate_to_goal = bool(self.get_parameter('rotate_to_goal').value)
        self.yaw_goal_tolerance = float(self.get_parameter('yaw_goal_tolerance').value)
        self.log_interval_sec = float(self.get_parameter('log_interval_sec').value)
        self.plan_path_points = []

        self._active = False
        self._path_version = 0
        self._last_seen_path_version = -1
        self._last_log_time = 0.0

    def path_callback(self, msg):
        self.lock.acquire()
        self.path = msg
        self._path_version += 1
        self._active = True
        self.update_global_pose(init=True)
        self.lock.release()

        goal = msg.poses[-1].pose.position if msg.poses else None
        if goal is not None:
            self.get_logger().info(
                f"Received global_path: poses={len(msg.poses)}, goal=({goal.x:.3f},{goal.y:.3f}), "
                f"frames(map={self.map_frame}, base={self.base_frame})"
            )

        if self.planning_thread is None:
            self.planning_thread = Thread(target=self.plan_thread_func, daemon=True)
            self.planning_thread.start()

    def plan_thread_func(self):
        self.get_logger().info("Planning loop started (waits for paths and supports multiple goals).")
        while rclpy.ok():
            sleep_dt = 1.0 / max(self.controller_frequency, 1.0)
            self.lock.acquire()
            try:
                if len(self.path.poses) < 2:
                    self.lock.release()
                    time.sleep(0.1)
                    continue

                if self._last_seen_path_version != self._path_version:
                    self._last_seen_path_version = self._path_version
                    self.vx = 0.0
                    self.vw = 0.0
                    self.get_logger().info("Switched to new path (reset controller state).")

                if self.goal_dis < self.arrive:
                    if self.rotate_to_goal and len(self.path.poses) >= 2:
                        p1 = self.path.poses[-2].pose.position
                        p2 = self.path.poses[-1].pose.position
                        yaw_des = math.atan2(p2.y - p1.y, p2.x - p1.x)
                        yaw_err = normalize_angle(yaw_des - self.yaw)
                        if abs(yaw_err) > self.yaw_goal_tolerance:
                            self.vx = 0.0
                            self.vw = max(min(1.5 * yaw_err, self.V_W), -self.V_W)
                            self.publish_velocity(zero=False)
                            self.lock.release()
                            time.sleep(sleep_dt)
                            continue

                    if self._active:
                        self.publish_velocity(zero=True)
                        self._active = False
                        self.get_logger().info("Arrived at goal; waiting for next global_path...")
                    self.lock.release()
                    time.sleep(0.1)
                    continue

                if not self._active:
                    self.lock.release()
                    time.sleep(0.1)
                    continue

                self.plan_once()

                now = time.time()
                if now - self._last_log_time >= max(self.log_interval_sec, 0.1):
                    self._last_log_time = now
                    ob_n = int(getattr(self, "plan_ob", np.empty((0, 2))).shape[0])
                    gx, gy = getattr(self, "plan_goal", (float("nan"), float("nan")))
                    self.get_logger().info(
                        f"state: x={self.x:.2f} y={self.y:.2f} yaw={self.yaw:.2f} "
                        f"goal_r=({gx:.2f},{gy:.2f}) dist={self.goal_dis:.2f} "
                        f"ob={ob_n} cmd(v={self.vx:.2f}, w={self.vw:.2f})"
                    )
            except Exception as e:
                self.get_logger().error(f"Planning loop error: {e}")
            finally:
                if self.lock.locked():
                    self.lock.release()
            time.sleep(sleep_dt)
        self.get_logger().info("Planning loop stopped.")

    def plan_once(self):
        self.update_global_pose(init=False)
        self.update_obstacle()

        # DWA expects pose in robot frame, so robot is at (0, 0, 0)
        # Goal and obstacles are already in robot frame
        pose_robot_frame = (0.0, 0.0, 0.0)
        velocity = (self.vx, self.vw)

        # Use DWA planner for local path planning
        u = self.planner.planning(
            pose=pose_robot_frame, 
            velocity=velocity, 
            goal=self.plan_goal, 
            points_cloud=self.plan_ob.tolist(),
            path_points=self.plan_path_points,
        )

        # Apply velocity limits
        self.vx = max(min(u[0], self.V_X), -self.V_X)
        self.vw = max(min(u[1], self.V_W), -self.V_W)
        
        # Debug logging (can be removed later)
        if abs(self.vx) < 0.01 and abs(self.vw) < 0.01:
            self.get_logger().debug(
                f"Zero velocity - goal: {self.plan_goal}, "
                f"obstacles: {len(self.plan_ob)}, "
                f"velocity: {velocity}"
            )
        
        self.get_logger().debug(f"cmd: v={self.vx:.3f}, w={self.vw:.3f}")
        self.publish_velocity(zero=False)




    def update_global_pose(self, init=False):
        try:
            # 根据是否为真实环境选择不同的坐标系
            source_frame = 'map' 
            target_frame = 'base_footprint'

            # 尝试获取坐标变换
            trans = self.tf_buffer.lookup_transform(
                self.map_frame,
                self.base_frame,
                rclpy.time.Time(),
                rclpy.duration.Duration(seconds=1.0)  # 超时时间设置为1秒
            )

            # 更新位置信息
            self.x = trans.transform.translation.x
            self.y = trans.transform.translation.y

            q = trans.transform.rotation
            siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
            cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
            self.yaw = math.atan2(siny_cosp, cosy_cosp)

        except TransformException as e:
            # 捕获变换异常并记录错误日志
            self.get_logger().error(f"TF 错误: {e}")

        pose = PoseStamped()
        pose.header.frame_id = self.map_frame
        pose.pose.position.x = self.x
        pose.pose.position.y = self.y
        self.traj.poses.append(pose)
        self.traj_pub.publish(self.traj)

        if init:
            self.goal_index = 0
            self.traj.poses = []

        if len(self.path.poses) < 2:
            return

        pts = np.array(
            [[p.pose.position.x, p.pose.position.y] for p in self.path.poses],
            dtype=float,
        )
        d2 = (pts[:, 0] - self.x) ** 2 + (pts[:, 1] - self.y) ** 2
        nearest = int(np.argmin(d2))

        goal_index = min(nearest + 1, len(self.path.poses) - 1)
        acc = 0.0
        for i in range(nearest, len(self.path.poses) - 1):
            seg = math.hypot(pts[i + 1, 0] - pts[i, 0], pts[i + 1, 1] - pts[i, 1])
            acc += seg
            if acc >= self.lookahead_dist:
                goal_index = i + 1
                break

        self.goal_index = goal_index
        goal = self.path.poses[self.goal_index]
        self.midpose_pub.publish(goal)

        try:
            lgoal = self.tf_buffer.transform(goal, self.base_frame)
            self.plan_goal = (lgoal.pose.position.x, lgoal.pose.position.y)
        except Exception as e:
            self.get_logger().error(f"TF transformation error: {e}")

        local_pts = []
        end = min(nearest + max(self.local_path_max_points, 2), len(self.path.poses))
        for i in range(nearest, end):
            try:
                lp = self.tf_buffer.transform(self.path.poses[i], self.base_frame)
                local_pts.append((lp.pose.position.x, lp.pose.position.y))
            except Exception:
                break
        self.plan_path_points = local_pts

        self.goal_dis = math.hypot(
            self.x - self.path.poses[-1].pose.position.x,
            self.y - self.path.poses[-1].pose.position.y,
        )

    def laser_callback(self, msg):
        self.laser_lock.acquire()
        self.ob = []
        angle_min = msg.angle_min
        angle_increment = msg.angle_increment
        for i, r in enumerate(msg.ranges):
            # Skip invalid readings
            if not (msg.range_min <= r <= msg.range_max):
                continue
            a = angle_min + angle_increment * i
            if r < self.threshold:
                self.ob.append((math.cos(a) * r, math.sin(a) * r))
        self.laser_lock.release()

    def update_obstacle(self):
        self.laser_lock.acquire()
        self.plan_ob = np.array(self.ob)
        self.laser_lock.release()

    def publish_velocity(self, zero=False):
        if zero:
            self.vx = 0.0
            self.vw = 0.0
        cmd = Twist()
        cmd.linear.x = float(self.vx)
        cmd.angular.z = float(self.vw)
        self.vel_pub.publish(cmd)


def normalize_angle(angle):
    while angle > math.pi:
        angle -= 2 * math.pi
    while angle < -math.pi:
        angle += 2 * math.pi
    return angle


def main(args=None):
    rclpy.init(args=args)
    local_planner = LocalPlanner()
    rclpy.spin(local_planner)
    local_planner.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
