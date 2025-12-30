# -*- coding: utf-8 -*-

"""Dynamic Window Approach (DWA) local path planner.

This module implements the DWA algorithm for local motion planning,
which generates velocity commands that balance goal tracking, obstacle
avoidance, and speed optimization.
"""

import math
import numpy as np
from typing import List


class DWA:
    """Dynamic Window Approach planner for local motion planning.

    DWA is a velocity-based local planner that searches for optimal
    velocity commands within a dynamic window, considering robot dynamics,
    obstacle avoidance, and goal tracking.
    """

    def __init__(self):
        """Initialize DWA planner with default parameters."""
        # Robot physical parameters
        self.max_speed = 1.0  # Maximum linear velocity (m/s)
        self.min_speed = 0.0  # Minimum linear velocity (m/s)
        self.max_yawrate = 1.0  # Maximum angular velocity (rad/s)
        self.max_accel = 0.2  # Maximum linear acceleration (m/s^2)
        self.max_dyawrate = 2.0  # Maximum angular acceleration (rad/s^2)
        self.v_reso = 0.05  # Linear velocity resolution (m/s)
        self.yawrate_reso = 0.1  # Angular velocity resolution (rad/s)
        self.v_samples = 6  # Samples in dynamic window (v)
        self.w_samples = 12  # Samples in dynamic window (w)
        self.dt = 0.1  # Time step for trajectory simulation (s)
        self.predict_time = 2.0  # Prediction time horizon (s)
        self.to_goal_cost_gain = 0.15  # Weight for goal distance cost
        self.path_cost_gain = 1.0  # Weight for path following cost
        self.heading_cost_gain = 0.2  # Weight for heading alignment cost
        self.speed_cost_gain = 1.0  # Weight for speed cost
        self.obstacle_cost_gain = 1.0  # Weight for obstacle cost
        self.robot_radius = 0.2  # Robot radius for collision checking (m)
        self.obstacle_threshold = 0.5  # Minimum distance to obstacles (m)
        self.normalize_costs = True  # Normalize critics across candidates

    def config(self, max_speed, max_yawrate, base):
        """Configure planner parameters.

        Args:
            max_speed: Maximum linear velocity (m/s).
            max_yawrate: Maximum angular velocity (rad/s).
            base: Robot base radius (m).
        """
        self.max_speed = max_speed
        self.max_yawrate = max_yawrate
        self.robot_radius = base

    @staticmethod
    def _normalize(values: List[float]) -> List[float]:
        if not values:
            return []
        vmin = min(values)
        vmax = max(values)
        if vmax - vmin < 1e-9:
            return [0.0 for _ in values]
        return [(v - vmin) / (vmax - vmin) for v in values]

    def motion_model(self, x, u, dt):
        """Calculate next state based on motion model.

        Uses a simple unicycle model: x' = x + v*cos(theta)*dt,
        y' = y + v*sin(theta)*dt, theta' = theta + w*dt.

        Args:
            x: Current state [x, y, yaw].
            u: Control input [v, w].
            dt: Time step (s).

        Returns:
            Next state [x, y, yaw].
        """
        x[0] += u[0] * math.cos(x[2]) * dt
        x[1] += u[0] * math.sin(x[2]) * dt
        x[2] += u[1] * dt
        return x

    def calc_dynamic_window(self, v, w):
        """Calculate dynamic window based on current velocity and constraints.

        The dynamic window limits the search space to velocities that
        can be reached within one time step given acceleration constraints.

        Args:
            v: Current linear velocity (m/s).
            w: Current angular velocity (rad/s).

        Returns:
            Tuple of (v_min, v_max, w_min, w_max) defining the dynamic window.
        """
        # Velocity constraints
        vs = [self.min_speed, self.max_speed, -self.max_yawrate, self.max_yawrate]

        # Acceleration constraints
        vd = [
            v - self.max_accel * self.dt,
            v + self.max_accel * self.dt,
            w - self.max_dyawrate * self.dt,
            w + self.max_dyawrate * self.dt,
        ]

        # Dynamic window
        v_min = max(vs[0], vd[0])
        v_max = min(vs[1], vd[1])
        w_min = max(vs[2], vd[2])
        w_max = min(vs[3], vd[3])

        return v_min, v_max, w_min, w_max

    def calc_trajectory(self, x_init, v, w):
        """Simulate trajectory for given velocity command.

        Args:
            x_init: Initial state [x, y, yaw].
            v: Linear velocity (m/s).
            w: Angular velocity (rad/s).

        Returns:
            Trajectory as list of states [[x, y, yaw], ...].
        """
        x = np.array(x_init)
        trajectory = [x.copy()]
        time = 0

        while time <= self.predict_time:
            x = self.motion_model(x, [v, w], self.dt)
            trajectory.append(x.copy())
            time += self.dt

        return np.array(trajectory)

    def calc_to_goal_cost(self, trajectory, goal):
        """Calculate cost based on distance to goal.

        Args:
            trajectory: Simulated trajectory.
            goal: Goal position [x, y].

        Returns:
            Cost value (lower is better).
        """
        dx = goal[0] - trajectory[-1, 0]
        dy = goal[1] - trajectory[-1, 1]
        goal_dist = math.sqrt(dx**2 + dy**2)
        return goal_dist

    def calc_obstacle_cost(self, trajectory, obstacles):
        """Calculate cost based on proximity to obstacles.

        Args:
            trajectory: Simulated trajectory.
            obstacles: List of obstacle points [(x, y), ...].

        Returns:
            Cost value (lower is better). Returns infinity if collision detected.
        """
        if len(obstacles) == 0:
            return 0.0

        obstacles = np.asarray(obstacles, dtype=float)
        traj_xy = trajectory[:, :2]  # (T, 2)

        dx = traj_xy[:, None, 0] - obstacles[None, :, 0]
        dy = traj_xy[:, None, 1] - obstacles[None, :, 1]
        dists = np.hypot(dx, dy)

        if np.any(dists <= self.robot_radius):
            return float('inf')

        min_dist = float(np.min(dists))
        return 1.0 / max(min_dist, 1e-6)

    def calc_speed_cost(self, v):
        """Calculate cost based on speed (prefer higher speeds).

        Args:
            v: Linear velocity (m/s).

        Returns:
            Cost value (lower is better).
        """
        return self.max_speed - v

    def calc_path_cost(self, trajectory, path_points):
        """Distance from trajectory endpoint to the global path (robot frame)."""
        if not path_points:
            return 0.0
        path = np.asarray(path_points, dtype=float)
        dx = path[:, 0] - trajectory[-1, 0]
        dy = path[:, 1] - trajectory[-1, 1]
        return float(np.min(np.hypot(dx, dy)))

    def calc_heading_cost(self, trajectory, goal):
        """Heading alignment cost at trajectory endpoint (robot frame)."""
        dx = goal[0] - trajectory[-1, 0]
        dy = goal[1] - trajectory[-1, 1]
        goal_angle = math.atan2(dy, dx)
        yaw = trajectory[-1, 2]
        d = goal_angle - yaw
        while d > math.pi:
            d -= 2.0 * math.pi
        while d < -math.pi:
            d += 2.0 * math.pi
        return abs(d)

    def planning(self, pose, velocity, goal, points_cloud, path_points=None):
        """Plan optimal velocity command using DWA.

        Args:
            pose: Current robot pose [x, y, yaw] in robot frame.
            velocity: Current velocity [v, w].
            goal: Goal position [x, y] in robot frame.
            points_cloud: List of obstacle points [(x, y), ...] in robot frame.
            path_points: Optional list of path points [(x, y), ...] in robot frame.

        Returns:
            Tuple of (v, w) representing optimal velocity command.
        """
        # Filter out obstacles that are too far away (likely invalid)
        valid_obstacles = []
        for obs in points_cloud:
            dist = math.sqrt(obs[0]**2 + obs[1]**2)
            if dist < 10.0:  # Only consider obstacles within 10m
                valid_obstacles.append(obs)

        # Initialize with current velocity
        v = velocity[0]
        w = velocity[1]

        # Calculate dynamic window
        v_min, v_max, w_min, w_max = self.calc_dynamic_window(v, w)

        # Ensure dynamic window is valid
        if v_min >= v_max:
            v_min = 0.0
            v_max = self.max_speed * 0.5  # Use half max speed as fallback
        if w_min >= w_max:
            w_min = -self.max_yawrate * 0.5
            w_max = self.max_yawrate * 0.5

        # Calculate simple goal-seeking velocity as fallback
        goal_dist = math.sqrt(goal[0]**2 + goal[1]**2)
        goal_angle = math.atan2(goal[1], goal[0])
        
        # Initialize fallback velocities
        fallback_v = 0.0
        fallback_w = 0.0
        if goal_dist > 0.05:  # If goal is not too close
            fallback_v = min(goal_dist * 0.3, self.max_speed * 0.3)
            fallback_w = goal_angle * 0.5
            fallback_w = max(min(fallback_w, self.max_yawrate), -self.max_yawrate)

        # Initialize best cost and velocity
        # Default to a simple goal-seeking behavior if no valid trajectory found
        best_cost = float("inf")
        best_u = [fallback_v, fallback_w]

        # Search for optimal velocity in dynamic window
        valid_trajectories = 0

        vs = np.linspace(v_min, v_max, max(int(self.v_samples), 2))
        ws = np.linspace(w_min, w_max, max(int(self.w_samples), 2))

        candidates = []
        to_goal_costs = []
        path_costs = []
        heading_costs = []
        speed_costs = []
        obstacle_costs = []

        for test_v in vs:
            for test_w in ws:
                # Simulate trajectory
                trajectory = self.calc_trajectory(pose, test_v, test_w)

                # Calculate costs
                to_goal_cost = self.calc_to_goal_cost(trajectory, goal)
                path_cost = self.calc_path_cost(trajectory, path_points)
                heading_cost = self.calc_heading_cost(trajectory, goal)
                speed_cost = self.calc_speed_cost(test_v)
                obstacle_cost = self.calc_obstacle_cost(trajectory, valid_obstacles)

                # Skip if collision detected
                if obstacle_cost == float('inf'):
                    continue

                valid_trajectories += 1
                candidates.append((float(test_v), float(test_w)))
                to_goal_costs.append(float(to_goal_cost))
                path_costs.append(float(path_cost))
                heading_costs.append(float(heading_cost))
                speed_costs.append(float(speed_cost))
                obstacle_costs.append(float(obstacle_cost))

        if valid_trajectories > 0:
            if self.normalize_costs:
                to_goal_costs_n = self._normalize(to_goal_costs)
                path_costs_n = self._normalize(path_costs)
                heading_costs_n = self._normalize(heading_costs)
                speed_costs_n = self._normalize(speed_costs)
                obstacle_costs_n = self._normalize(obstacle_costs)
            else:
                to_goal_costs_n = to_goal_costs
                path_costs_n = path_costs
                heading_costs_n = heading_costs
                speed_costs_n = speed_costs
                obstacle_costs_n = obstacle_costs

            for i, (test_v, test_w) in enumerate(candidates):
                final_cost = (
                    self.to_goal_cost_gain * to_goal_costs_n[i]
                    + self.path_cost_gain * path_costs_n[i]
                    + self.heading_cost_gain * heading_costs_n[i]
                    + self.speed_cost_gain * speed_costs_n[i]
                    + self.obstacle_cost_gain * obstacle_costs_n[i]
                )

                if final_cost < best_cost:
                    best_cost = final_cost
                    best_u = [test_v, test_w]

        # If no valid trajectories found, use fallback
        if valid_trajectories == 0:
            best_u = [fallback_v, fallback_w]

        # If we get stuck with (almost) zero linear velocity, rotate towards goal.
        if abs(best_u[0]) < 0.01 and goal_dist > 0.1:
            best_u[0] = 0.0
            if abs(best_u[1]) < 0.01:
                best_u[1] = max(min(goal_angle, self.max_yawrate), -self.max_yawrate)

        return best_u[0], best_u[1]
