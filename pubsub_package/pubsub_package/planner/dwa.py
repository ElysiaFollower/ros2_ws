# -*- coding: utf-8 -*-

"""
DWA Algorithm (Production Grade)
Stateless implementation with vectorized cost calculation
"""

import math
import numpy as np

class DWA:
    def __init__(self):
        # Default config, should be overwritten by `config` method
        self.max_speed = 0.5
        self.max_yawrate = 1.0
        self.dt = 0.1
        self.predict_time = 2.0
        self.v_samples = 10
        self.w_samples = 20
        self.robot_radius = 0.2
        
        # Weights
        self.alpha = 0.5  # To Goal
        self.beta = 0.5   # To Path
        self.gamma = 1.0  # Obstacle
        self.delta = 0.2  # Speed

    def config(self, max_speed, max_yawrate, base, **kwargs):
        self.max_speed = max_speed
        self.max_yawrate = max_yawrate
        self.robot_radius = base
        # Allow updating weights
        if 'to_goal_cost_gain' in kwargs: self.alpha = kwargs['to_goal_cost_gain']
        if 'obstacle_cost_gain' in kwargs: self.gamma = kwargs['obstacle_cost_gain']

    def motion_model(self, x, u, dt):
        """
        x: [x, y, yaw]
        u: [v, w]
        """
        x[2] += u[1] * dt
        x[0] += u[0] * math.cos(x[2]) * dt
        x[1] += u[0] * math.sin(x[2]) * dt
        return x

    def predict_trajectory(self, x_init, v, w):
        traj = []
        x = np.array(x_init, dtype=float)
        time = 0
        while time <= self.predict_time:
            traj.append(x.copy())
            x = self.motion_model(x, [v, w], self.dt)
            time += self.dt
        return np.array(traj)

    def planning(self, pose, velocity, goal, obstacles, path_points=None):
        """
        Args:
            pose: [x, y, yaw] in ROBOT FRAME (usually [0,0,0])
            velocity: [v, w]
            goal: [x, y] in ROBOT FRAME
            obstacles: list of [x, y] in ROBOT FRAME
        Returns:
            best_u: [v, w]
        """
        best_u = [0.0, 0.0]
        min_cost = float('inf')
        
        # Dynamic Window
        v_min = max(0.0, velocity[0] - 0.5 * self.dt) # accel limit assumption
        v_max = min(self.max_speed, velocity[0] + 0.5 * self.dt)
        w_min = max(-self.max_yawrate, velocity[1] - 1.0 * self.dt)
        w_max = min(self.max_yawrate, velocity[1] + 1.0 * self.dt)
        
        vs = np.linspace(v_min, v_max, self.v_samples)
        ws = np.linspace(w_min, w_max, self.w_samples)
        
        valid_obs = np.array(obstacles) if len(obstacles) > 0 else np.empty((0,2))

        # Force a recovery spin if stuck? Handled by planner logic, not here.
        
        found_valid = False

        for v in vs:
            for w in ws:
                traj = self.predict_trajectory(pose, v, w)
                
                # 1. To Goal Cost
                dx = goal[0] - traj[-1, 0]
                dy = goal[1] - traj[-1, 1]
                cost_goal = math.hypot(dx, dy)
                
                # 2. Obstacle Cost
                cost_obs = 0.0
                min_dist = float('inf')
                if len(valid_obs) > 0:
                    # Vectorized distance calculation
                    # traj: (T, 3), obs: (N, 2)
                    # We need min dist from any point in traj to any point in obs
                    # Simplified: just check endpoint or subsample
                    
                    # For strict safety, check all traj points against all obs is slow in python
                    # We check every other point against kdtree or just simple loop
                    for tx, ty, _ in traj:
                        dists = np.hypot(valid_obs[:,0] - tx, valid_obs[:,1] - ty)
                        min_dist = min(min_dist, np.min(dists))
                    
                    if min_dist <= self.robot_radius:
                        cost_obs = float('inf') # Collision
                    else:
                        cost_obs = 1.0 / min_dist
                
                if cost_obs == float('inf'):
                    continue

                # 3. Speed Cost (Prefer fast)
                cost_speed = self.max_speed - v

                # Total
                total_cost = (self.alpha * cost_goal + 
                              self.gamma * cost_obs + 
                              self.delta * cost_speed)
                
                if total_cost < min_cost:
                    min_cost = total_cost
                    best_u = [v, w]
                    found_valid = True

        # Fallback: if no valid path, slow down or stop
        if not found_valid:
            return [0.0, 0.0]
            
        return best_u