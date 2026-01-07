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
        self.alpha = 0.8  # To Goal (increased)
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
        v_min = 0.0 # No backward motion for differential drive standard
        v_max = min(self.max_speed, velocity[0] + 0.2) # simple accel limit
        w_min = max(-self.max_yawrate, velocity[1] - 1.0)
        w_max = min(self.max_yawrate, velocity[1] + 1.0)
        
        vs = np.linspace(v_min, v_max, self.v_samples)
        ws = np.linspace(w_min, w_max, self.w_samples)
        
        valid_obs = np.array(obstacles) if (obstacles is not None and len(obstacles) > 0) else np.empty((0,2))
        
        found_valid = False

        for v in vs:
            for w in ws:
                traj = self.predict_trajectory(pose, v, w)
                
                # 1. To Goal Cost (Euclidean distance from end of trajectory)
                dx = goal[0] - traj[-1, 0]
                dy = goal[1] - traj[-1, 1]
                cost_goal = math.hypot(dx, dy)
                
                # 2. Obstacle Cost
                cost_obs = 0.0
                if len(valid_obs) > 0:
                    # Check distance from every point in trajectory to every obstacle
                    # Optimized: just check if any point is too close
                    min_dist = float('inf')
                    
                    # Simple loop for safety (can be vectorized further)
                    for tx, ty, _ in traj:
                         # Distances from current traj point to all obstacles
                         dists = np.hypot(valid_obs[:,0] - tx, valid_obs[:,1] - ty)
                         current_min = np.min(dists)
                         min_dist = min(min_dist, current_min)

                    if min_dist <= self.robot_radius:
                        cost_obs = float('inf') # Collision
                    else:
                        cost_obs = 1.0 / min_dist
                
                if cost_obs == float('inf'):
                    continue

                # 3. Speed Cost (Prefer high speed)
                cost_speed = self.max_speed - v

                # Total Cost
                total_cost = (self.alpha * cost_goal + 
                              self.gamma * cost_obs + 
                              self.delta * cost_speed)
                
                if total_cost < min_cost:
                    min_cost = total_cost
                    best_u = [v, w]
                    found_valid = True

        # Fallback: if no valid path, return zero velocity
        if not found_valid:
            return [0.0, 0.0]
            
        return best_u