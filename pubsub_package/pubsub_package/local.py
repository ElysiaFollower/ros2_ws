#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import rclpy
from rclpy.node import Node
from tf2_ros import TransformListener, Buffer
from tf2_ros import TransformException
from geometry_msgs.msg import PoseStamped
from rclpy.parameter import SetParametersResult
import math
import time
import numpy as np
from nav_msgs.msg import Path
from geometry_msgs.msg import  Twist

from sensor_msgs.msg import LaserScan
from threading import Lock, Thread
# Import your own planner
from pubsub_package.planner.dwa import DWA
from pubsub_package.param_panel import ParamSpec, ParamWebPanel

class LocalPlanner(Node):
    def __init__(self, real=None):
        super().__init__('local_planner')
        self.declare_parameter('enable_param_panel', False)
        self.declare_parameter('param_panel_host', '127.0.0.1')
        self.declare_parameter('param_panel_port', 8890)
        self.declare_parameter('param_panel_open_browser', False)

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
        self.declare_parameter('verbose_log', True)
        self.declare_parameter('arrive_dist', 0.2)
        self.declare_parameter('laser_threshold', 1.5)
        self.declare_parameter('laser_frame', '')
        self.declare_parameter('laser_yaw_offset', 0.0)
        self.declare_parameter('scan_stale_timeout', 0.5)
        self.declare_parameter('cmd_smoothing_alpha', 0.35)
        self.declare_parameter('cmd_max_accel', 0.8)
        self.declare_parameter('cmd_max_dyawrate', 4.0)
        self.declare_parameter('max_speed', 0.5)
        self.declare_parameter('max_yawrate', 0.5)
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
        self.arrive = float(self.get_parameter('arrive_dist').value)  # Standard for arrival
        self.threshold = float(self.get_parameter('laser_threshold').value)  # Laser threshold
        self.laser_frame = str(self.get_parameter('laser_frame').value)
        self.laser_yaw_offset = float(self.get_parameter('laser_yaw_offset').value)
        self.scan_stale_timeout = float(self.get_parameter('scan_stale_timeout').value)
        self.cmd_smoothing_alpha = float(self.get_parameter('cmd_smoothing_alpha').value)
        self.cmd_max_accel = float(self.get_parameter('cmd_max_accel').value)
        self.cmd_max_dyawrate = float(self.get_parameter('cmd_max_dyawrate').value)
        self.robot_size = float(self.get_parameter('robot_radius').value)
        self.safety_dist = float(self.get_parameter('safety_dist').value)
        self.V_X = float(self.get_parameter('max_speed').value)
        self.V_W = float(self.get_parameter('max_yawrate').value)

        self.planner = DWA()  # Initialize planner
        self._sync_params_locked()

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
        self.verbose_log = bool(self.get_parameter('verbose_log').value)
        self.plan_path_points = []

        self._active = False
        self._path_version = 0
        self._last_seen_path_version = -1
        self._last_log_time = 0.0
        self._param_sync_needed = False

        self._last_scan_time = 0.0
        self._last_pose_time = 0.0
        self._last_cmd_time = 0.0
        self._dbg: dict[str, float | int] = {}
        self._last_laser_tf_log_time = 0.0
        self._laser_tf_cache_time = 0.0
        self._laser_tf_cache = (0.0, 0.0, 0.0)  # tx, ty, yaw

        self.add_on_set_parameters_callback(self._on_set_parameters)

        self._param_panel: ParamWebPanel | None = None
        if bool(self.get_parameter('enable_param_panel').value):
            self._start_param_panel()
        self._param_panel_timer = self.create_timer(0.2, self._param_panel_tick)

        laser_topic = '/scan' if self.real else '/course_agv/laser/scan'
        cmd_topic = '/cmd_vel' if self.real else '/course_agv/velocity'
        self.get_logger().info(
            "LocalPlanner started: "
            f"real={self.real} map_frame={self.map_frame} base_frame={self.base_frame} "
            f"laser_topic={laser_topic} cmd_topic={cmd_topic} "
            f"robot_radius={self.robot_size:.3f} safety_dist={self.safety_dist:.3f} "
            f"max_speed={self.V_X:.3f} max_yawrate={self.V_W:.3f} "
            f"lookahead_dist={self.lookahead_dist:.2f} controller_frequency={self.controller_frequency:.1f}Hz "
            f"arrive_dist={self.arrive:.2f} laser_threshold={self.threshold:.2f} "
            f"log_interval_sec={self.log_interval_sec:.2f} verbose_log={self.verbose_log}"
        )

    def _on_set_parameters(self, params):
        for p in params:
            if p.name in ('map_frame', 'base_frame'):
                if not str(p.value).strip():
                    return SetParametersResult(successful=False, reason=f"{p.name} cannot be empty")
            if p.name in ('robot_radius', 'safety_dist'):
                if float(p.value) < 0.0:
                    return SetParametersResult(successful=False, reason=f"{p.name} must be >= 0")
            if p.name in ('controller_frequency',):
                if float(p.value) <= 0.0:
                    return SetParametersResult(successful=False, reason="controller_frequency must be > 0")
            if p.name.startswith('dwa.'):
                if p.name in ('dwa.v_samples', 'dwa.w_samples') and int(p.value) < 2:
                    return SetParametersResult(successful=False, reason=f"{p.name} must be >= 2")
                if p.name == 'dwa.predict_time' and float(p.value) <= 0.0:
                    return SetParametersResult(successful=False, reason="dwa.predict_time must be > 0")
                if p.name.endswith('cost_gain') and float(p.value) < 0.0:
                    return SetParametersResult(successful=False, reason=f"{p.name} must be >= 0")
        self._param_sync_needed = True
        return SetParametersResult(successful=True)

    def _sync_params_locked(self) -> None:
        self.map_frame = str(self.get_parameter('map_frame').value)
        self.base_frame = str(self.get_parameter('base_frame').value)

        self.arrive = float(self.get_parameter('arrive_dist').value)
        self.threshold = float(self.get_parameter('laser_threshold').value)
        self.laser_frame = str(self.get_parameter('laser_frame').value)
        self.laser_yaw_offset = float(self.get_parameter('laser_yaw_offset').value)
        self.scan_stale_timeout = float(self.get_parameter('scan_stale_timeout').value)
        self.cmd_smoothing_alpha = float(self.get_parameter('cmd_smoothing_alpha').value)
        self.cmd_max_accel = float(self.get_parameter('cmd_max_accel').value)
        self.cmd_max_dyawrate = float(self.get_parameter('cmd_max_dyawrate').value)

        self.robot_size = float(self.get_parameter('robot_radius').value)
        self.safety_dist = float(self.get_parameter('safety_dist').value)
        self.V_X = float(self.get_parameter('max_speed').value)
        self.V_W = float(self.get_parameter('max_yawrate').value)

        self.lookahead_dist = float(self.get_parameter('lookahead_dist').value)
        self.local_path_max_points = int(self.get_parameter('local_path_max_points').value)
        self.controller_frequency = float(self.get_parameter('controller_frequency').value)
        self.rotate_to_goal = bool(self.get_parameter('rotate_to_goal').value)
        self.yaw_goal_tolerance = float(self.get_parameter('yaw_goal_tolerance').value)
        self.log_interval_sec = float(self.get_parameter('log_interval_sec').value)
        self.verbose_log = bool(self.get_parameter('verbose_log').value)

        self.planner.config(max_speed=self.V_X, max_yawrate=self.V_W, base=self.robot_size + self.safety_dist)
        self.planner.to_goal_cost_gain = float(self.get_parameter('dwa.to_goal_cost_gain').value)
        self.planner.path_cost_gain = float(self.get_parameter('dwa.path_cost_gain').value)
        self.planner.heading_cost_gain = float(self.get_parameter('dwa.heading_cost_gain').value)
        self.planner.obstacle_cost_gain = float(self.get_parameter('dwa.obstacle_cost_gain').value)
        self.planner.speed_cost_gain = float(self.get_parameter('dwa.speed_cost_gain').value)
        self.planner.predict_time = float(self.get_parameter('dwa.predict_time').value)
        self.planner.v_samples = int(self.get_parameter('dwa.v_samples').value)
        self.planner.w_samples = int(self.get_parameter('dwa.w_samples').value)
        self.planner.dt = 1.0 / max(self.controller_frequency, 1e-6)

        if hasattr(self, "traj"):
            self.traj.header.frame_id = self.map_frame

        self._param_sync_needed = False

    def _param_panel_specs(self) -> dict[str, ParamSpec]:
        return {
            'map_frame': ParamSpec('map_frame', 'str', help='TF map frame'),
            'base_frame': ParamSpec('base_frame', 'str', help='TF base frame'),
            'robot_radius': ParamSpec('robot_radius', 'float', min=0.05, max=0.6, step=0.01, help='Robot radius (m)'),
            'safety_dist': ParamSpec('safety_dist', 'float', min=0.0, max=0.5, step=0.01, help='Extra safety distance (m)'),
            'max_speed': ParamSpec('max_speed', 'float', min=0.0, max=1.5, step=0.05, help='Max linear speed (m/s)'),
            'max_yawrate': ParamSpec('max_yawrate', 'float', min=0.0, max=3.0, step=0.05, help='Max angular speed (rad/s)'),
            'lookahead_dist': ParamSpec('lookahead_dist', 'float', min=0.2, max=5.0, step=0.1, help='Lookahead distance along global path (m)'),
            'controller_frequency': ParamSpec('controller_frequency', 'float', min=1.0, max=30.0, step=1.0, help='Control loop frequency (Hz)'),
            'arrive_dist': ParamSpec('arrive_dist', 'float', min=0.05, max=1.5, step=0.05, help='Goal arrival distance (m)'),
            'laser_threshold': ParamSpec('laser_threshold', 'float', min=0.2, max=10.0, step=0.1, help='Laser obstacle threshold (m)'),
            'laser_frame': ParamSpec('laser_frame', 'str', help='Laser frame override (empty uses LaserScan.header.frame_id)'),
            'laser_yaw_offset': ParamSpec('laser_yaw_offset', 'float', min=-3.2, max=3.2, step=0.05, help='Extra yaw offset for laser frame (rad)'),
            'scan_stale_timeout': ParamSpec('scan_stale_timeout', 'float', min=0.0, max=5.0, step=0.1, help='Stop if scan older than this (s)'),
            'cmd_smoothing_alpha': ParamSpec('cmd_smoothing_alpha', 'float', min=0.0, max=1.0, step=0.05, help='Command low-pass alpha (0..1)'),
            'cmd_max_accel': ParamSpec('cmd_max_accel', 'float', min=0.0, max=5.0, step=0.1, help='Command accel limit (m/s^2)'),
            'cmd_max_dyawrate': ParamSpec('cmd_max_dyawrate', 'float', min=0.0, max=20.0, step=0.2, help='Command yaw accel limit (rad/s^2)'),
            'rotate_to_goal': ParamSpec('rotate_to_goal', 'bool', help='Rotate to goal heading at end'),
            'yaw_goal_tolerance': ParamSpec('yaw_goal_tolerance', 'float', min=0.05, max=1.5, step=0.05, help='Yaw tolerance (rad)'),
            'log_interval_sec': ParamSpec('log_interval_sec', 'float', min=0.1, max=10.0, step=0.1, help='Log throttle (s)'),
            'verbose_log': ParamSpec('verbose_log', 'bool', help='Enable status logs'),
            'dwa.to_goal_cost_gain': ParamSpec('dwa.to_goal_cost_gain', 'float', min=0.0, max=10.0, step=0.05, help='Goal critic weight'),
            'dwa.path_cost_gain': ParamSpec('dwa.path_cost_gain', 'float', min=0.0, max=10.0, step=0.05, help='Path critic weight'),
            'dwa.heading_cost_gain': ParamSpec('dwa.heading_cost_gain', 'float', min=0.0, max=10.0, step=0.05, help='Heading critic weight'),
            'dwa.obstacle_cost_gain': ParamSpec('dwa.obstacle_cost_gain', 'float', min=0.0, max=10.0, step=0.05, help='Obstacle critic weight'),
            'dwa.speed_cost_gain': ParamSpec('dwa.speed_cost_gain', 'float', min=0.0, max=10.0, step=0.05, help='Speed critic weight'),
            'dwa.predict_time': ParamSpec('dwa.predict_time', 'float', min=0.5, max=5.0, step=0.1, help='Trajectory rollout time (s)'),
            'dwa.v_samples': ParamSpec('dwa.v_samples', 'int', min=2, max=50, step=1, help='Velocity samples'),
            'dwa.w_samples': ParamSpec('dwa.w_samples', 'int', min=2, max=80, step=1, help='Yawrate samples'),
        }

    def _start_param_panel(self) -> None:
        host = str(self.get_parameter('param_panel_host').value)
        port = int(self.get_parameter('param_panel_port').value)
        open_browser = bool(self.get_parameter('param_panel_open_browser').value)
        self._param_panel = ParamWebPanel(
            self,
            title="Local Planner Parameter Panel",
            host=host,
            port=port,
            specs=self._param_panel_specs(),
            open_browser=open_browser,
        )
        self._param_panel.start()

    def _param_panel_tick(self) -> None:
        if self._param_panel is not None:
            updates = self._param_panel.drain_updates()
            if updates:
                ok = self._param_panel.apply_updates(updates)
                if ok:
                    self.get_logger().info(f"Applied params: {sorted(updates.keys())}")

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
                if self._param_sync_needed:
                    self._sync_params_locked()
                    sleep_dt = 1.0 / max(self.controller_frequency, 1.0)
                    self.get_logger().info("Parameters updated (applies to next control cycle).")

                if len(self.path.poses) < 2:
                    if self.verbose_log:
                        now = time.time()
                        if now - self._last_log_time >= max(self.log_interval_sec, 0.1):
                            self._last_log_time = now
                            self.get_logger().info("Waiting for /course_agv/global_path ...")
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

                now = time.time()
                scan_age = (now - self._last_scan_time) if self._last_scan_time > 0 else float("inf")
                if scan_age > max(self.scan_stale_timeout, 0.0):
                    if self.verbose_log and now - self._last_log_time >= max(self.log_interval_sec, 0.1):
                        self._last_log_time = now
                        self.get_logger().info(
                            f"Scan stale: scan_age={scan_age:.2f}s > timeout={self.scan_stale_timeout:.2f}s; "
                            "publishing zero cmd"
                        )
                    self.publish_velocity(zero=True)
                    self.lock.release()
                    time.sleep(sleep_dt)
                    continue

                self.plan_once()

                now = time.time()
                if self.verbose_log and now - self._last_log_time >= max(self.log_interval_sec, 0.1):
                    self._last_log_time = now
                    ob_n = int(getattr(self, "plan_ob", np.empty((0, 2))).shape[0])
                    gx, gy = getattr(self, "plan_goal", (float("nan"), float("nan")))
                    min_ob = float(self._dbg.get("min_ob_dist", float("nan")))
                    min_ob_a = float(self._dbg.get("min_ob_a", float("nan")))
                    goal_d = float(self._dbg.get("goal_d", float("nan")))
                    goal_a = float(self._dbg.get("goal_a", float("nan")))
                    near = int(self._dbg.get("nearest", -1))
                    gi = int(self._dbg.get("goal_index", -1))
                    look = float(self._dbg.get("lookahead", float("nan")))
                    path_len = int(self._dbg.get("path_len", 0))
                    scan_age = (now - self._last_scan_time) if self._last_scan_time > 0 else float("inf")
                    self.get_logger().info(
                        f"state: x={self.x:.2f} y={self.y:.2f} yaw={self.yaw:.2f} "
                        f"goal_r=({gx:.2f},{gy:.2f}) goal_d={goal_d:.2f} goal_a={goal_a:.2f} "
                        f"global_goal_dist={self.goal_dis:.2f} "
                        f"path_len={path_len} nearest={near} goal_index={gi} lookahead={look:.2f} "
                        f"ob={ob_n} min_ob={min_ob:.2f} min_ob_a={min_ob_a:.2f} scan_age={scan_age:.2f}s "
                        f"cmd(v={self.vx:.2f}, w={self.vw:.2f})"
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
        goal_d = float("nan")
        goal_a = float("nan")
        if hasattr(self, "plan_goal"):
            goal_d = float(math.hypot(self.plan_goal[0], self.plan_goal[1]))
            goal_a = float(math.atan2(self.plan_goal[1], self.plan_goal[0]))

        min_ob = float("nan")
        if hasattr(self, "plan_ob") and getattr(self.plan_ob, "size", 0) > 0:
            d = np.hypot(self.plan_ob[:, 0], self.plan_ob[:, 1])
            if d.size > 0:
                min_ob = float(np.min(d))
                idx = int(np.argmin(d))
                self._dbg["min_ob_a"] = float(math.atan2(float(self.plan_ob[idx, 1]), float(self.plan_ob[idx, 0])))
            else:
                self._dbg["min_ob_a"] = float("nan")
        else:
            self._dbg["min_ob_a"] = float("nan")

        self._dbg["goal_d"] = goal_d
        self._dbg["goal_a"] = goal_a
        self._dbg["min_ob_dist"] = min_ob
        self._dbg["lookahead"] = float(self.lookahead_dist)
        self._dbg["path_len"] = int(len(getattr(self.path, "poses", []) or []))
        self._dbg["goal_index"] = int(getattr(self, "goal_index", -1))

        u = self.planner.planning(
            pose=pose_robot_frame, 
            velocity=velocity, 
            goal=self.plan_goal, 
            points_cloud=self.plan_ob.tolist(),
            path_points=self.plan_path_points,
        )

        # Apply velocity limits + smoothing + rate limiting to reduce "stuttering"
        target_v = max(min(float(u[0]), self.V_X), -self.V_X)
        target_w = max(min(float(u[1]), self.V_W), -self.V_W)

        alpha = float(max(min(self.cmd_smoothing_alpha, 1.0), 0.0))
        blended_v = (1.0 - alpha) * float(self.vx) + alpha * target_v
        blended_w = (1.0 - alpha) * float(self.vw) + alpha * target_w

        dt_ctrl = 1.0 / max(self.controller_frequency, 1e-6)
        max_dv = max(float(self.cmd_max_accel), 0.0) * dt_ctrl
        max_dw = max(float(self.cmd_max_dyawrate), 0.0) * dt_ctrl

        dv = blended_v - float(self.vx)
        dw = blended_w - float(self.vw)
        if max_dv > 0.0:
            dv = max(min(dv, max_dv), -max_dv)
        if max_dw > 0.0:
            dw = max(min(dw, max_dw), -max_dw)

        self.vx = float(self.vx) + float(dv)
        self.vw = float(self.vw) + float(dw)
        self._last_cmd_time = time.time()
        
        # Debug logging (can be removed later)
        if abs(self.vx) < 0.01 and abs(self.vw) < 0.01:
            if self.verbose_log:
                self.get_logger().info(
                    f"Low command (near zero): goal_r=({self.plan_goal[0]:.2f},{self.plan_goal[1]:.2f}) "
                    f"min_ob={min_ob:.2f} v_in={velocity[0]:.2f} w_in={velocity[1]:.2f} "
                    f"goal_d={goal_d:.2f} goal_a={goal_a:.2f}"
                )
        
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
            self._last_pose_time = time.time()

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
        self._dbg["nearest"] = int(nearest)

        goal_index = min(nearest + 1, len(self.path.poses) - 1)
        acc = 0.0
        for i in range(nearest, len(self.path.poses) - 1):
            seg = math.hypot(pts[i + 1, 0] - pts[i, 0], pts[i + 1, 1] - pts[i, 1])
            acc += seg
            if acc >= self.lookahead_dist:
                goal_index = i + 1
                break

        self.goal_index = goal_index
        self._dbg["goal_index"] = int(goal_index)
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

    def _laser_to_base_transform(self, scan_frame: str) -> tuple[float, float, float, bool]:
        now = time.time()
        if now - self._laser_tf_cache_time < 0.5:
            tx, ty, yaw = self._laser_tf_cache
            return tx, ty, yaw, True

        tx, ty = 0.0, 0.0
        yaw_off = float(self.laser_yaw_offset)
        tf_ok = False
        base_frame = (self.base_frame or "").lstrip("/")
        scan_frame = (scan_frame or "").lstrip("/")
        if scan_frame and base_frame and scan_frame != base_frame:
            try:
                trans = self.tf_buffer.lookup_transform(
                    self.base_frame,
                    scan_frame,
                    rclpy.time.Time(),
                    rclpy.duration.Duration(seconds=0.02),
                )
                tx = float(trans.transform.translation.x)
                ty = float(trans.transform.translation.y)
                q = trans.transform.rotation
                siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
                cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
                yaw_off = float(math.atan2(siny_cosp, cosy_cosp) + yaw_off)
                tf_ok = True
            except TransformException:
                tf_ok = False

        self._laser_tf_cache_time = now
        self._laser_tf_cache = (tx, ty, yaw_off)
        return tx, ty, yaw_off, tf_ok

    def laser_callback(self, msg):
        self._last_scan_time = time.time()

        scan_frame = (self.laser_frame or getattr(msg.header, "frame_id", "") or self.base_frame)
        tx, ty, yaw_off, tf_ok = self._laser_to_base_transform(scan_frame)

        now = time.time()
        if self.verbose_log and now - self._last_laser_tf_log_time >= max(self.log_interval_sec, 0.1):
            self._last_laser_tf_log_time = now
            self.get_logger().info(
                f"laser: frame={(scan_frame or '(none)')} -> base={self.base_frame} tf_ok={tf_ok} "
                f"yaw_off={yaw_off:.3f} t=({tx:.3f},{ty:.3f}) threshold={self.threshold:.2f}"
            )

        c0 = math.cos(yaw_off)
        s0 = math.sin(yaw_off)
        angle_min = msg.angle_min
        angle_increment = msg.angle_increment

        local_ob: list[tuple[float, float]] = []
        for i, r in enumerate(msg.ranges):
            if not (msg.range_min <= r <= msg.range_max):
                continue
            if r >= self.threshold:
                continue
            a = angle_min + angle_increment * i
            xl = math.cos(a) * r
            yl = math.sin(a) * r
            xb = tx + (xl * c0 - yl * s0)
            yb = ty + (xl * s0 + yl * c0)
            local_ob.append((xb, yb))

        self.laser_lock.acquire()
        self.ob = local_ob
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
