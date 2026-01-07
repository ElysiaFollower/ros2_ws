#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
import random
import numpy as np

class RRT_star:
    """
    RRT* Path Planner optimized for Grid Maps.
    Directly checks collision on the occupancy grid instead of using KDTree on points.
    """

    class Node:
        def __init__(self, x, y):
            self.x = x
            self.y = y
            self.path_x = []
            self.path_y = []
            self.parent = None
            self.cost = 0.0

    def __init__(self, grid_map, resolution, origin_x, origin_y, 
                 robot_radius=0.2, safe_dist=0.05,
                 expand_dis=1.0, path_resolution=0.1, 
                 goal_sample_rate=5, max_iter=1000):
        """
        Args:
            grid_map: 2D numpy array (0=free, 100=occupied, -1=unknown)
            resolution: Map resolution (m/cell)
            origin_x, origin_y: Map origin coordinates
            robot_radius: Robot physical radius (m)
            safe_dist: Extra safety margin (m)
            expand_dis: RRT Step size (m) - increased for speed
            max_iter: Max iterations - reduced for speed
        """
        self.grid = grid_map
        self.resolution = resolution
        self.origin_x = origin_x
        self.origin_y = origin_y
        self.height, self.width = grid_map.shape
        
        # Calculate bounds
        self.minx = origin_x
        self.maxx = origin_x + self.width * resolution
        self.miny = origin_y
        self.maxy = origin_y + self.height * resolution

        # Collision parameters
        self.collision_radius = robot_radius + safe_dist
        self.expand_dis = expand_dis
        self.path_resolution = path_resolution
        self.goal_sample_rate = goal_sample_rate
        self.max_iter = max_iter
        
        self.node_list = []

    def plan(self, sx, sy, gx, gy):
        """Plan path from (sx, sy) to (gx, gy)."""
        # 1. Bounds check
        if not self._is_valid_pos(sx, sy):
            print(f"[RRT] Error: Start ({sx:.2f}, {sy:.2f}) out of map bounds")
            return False, []
        if not self._is_valid_pos(gx, gy):
            print(f"[RRT] Error: Goal ({gx:.2f}, {gy:.2f}) out of map bounds")
            return False, []
        
        # 2. Collision check for start/goal
        # We allow start to be slightly invalid (to escape obstacles), but goal must be valid
        if self._is_in_collision(gx, gy):
            print("[RRT] Error: Goal is in collision or unknown area!")
            return False, []

        self.start = self.Node(sx, sy)
        self.end = self.Node(gx, gy)
        self.node_list = [self.start]

        for i in range(self.max_iter):
            # Random Sampling
            rnd = self._get_random_node()
            
            # Find nearest node
            nearest_ind = self._get_nearest_node_index(self.node_list, rnd)
            nearest_node = self.node_list[nearest_ind]

            # Steer
            new_node = self._steer(nearest_node, rnd, self.expand_dis)

            if self._check_collision_path(new_node):
                near_inds = self._find_near_nodes(new_node)
                
                # Choose Parent
                new_node = self._choose_parent(new_node, near_inds)
                
                if new_node:
                    self.node_list.append(new_node)
                    # Rewire
                    self._rewire(new_node, near_inds)

            # Early Exit: Check if goal is reached regularly
            if i % 50 == 0:
                dist_to_goal = math.hypot(self.node_list[-1].x - self.end.x, 
                                        self.node_list[-1].y - self.end.y)
                if dist_to_goal <= self.expand_dis:
                     final_node = self._steer(self.node_list[-1], self.end)
                     if self._check_collision_path(final_node):
                         return True, self._generate_final_course(len(self.node_list)-1)

        return False, []

    def _is_valid_pos(self, x, y):
        return (self.minx <= x <= self.maxx) and (self.miny <= y <= self.maxy)

    def _world_to_grid(self, x, y):
        gx = int((x - self.origin_x) / self.resolution)
        gy = int((y - self.origin_y) / self.resolution)
        return gx, gy

    def _is_in_collision(self, x, y):
        """Fast O(1) grid check."""
        gx, gy = self._world_to_grid(x, y)
        if not (0 <= gx < self.width and 0 <= gy < self.height):
            return True # Out of bounds is collision
        
        # Grid values: 0 (free), 100 (occupied), -1 (unknown)
        val = self.grid[gy, gx]
        if val == -1 or val > 50: 
            return True
            
        return False

    def _check_collision_path(self, node):
        """Check collision along the path segments."""
        if node is None: return False
        for x, y in zip(node.path_x, node.path_y):
            if self._is_in_collision(x, y):
                return False
        return True

    def _steer(self, from_node, to_node, extend_length=float("inf")):
        new_node = self.Node(from_node.x, from_node.y)
        d, theta = self._calc_dist_angle(new_node, to_node)
        
        new_node.path_x = [new_node.x]
        new_node.path_y = [new_node.y]
        
        if extend_length > d:
            extend_length = d
            
        n_expand = math.floor(extend_length / self.path_resolution)
        
        curr_x = new_node.x
        curr_y = new_node.y
        
        for _ in range(n_expand):
            curr_x += self.path_resolution * math.cos(theta)
            curr_y += self.path_resolution * math.sin(theta)
            new_node.path_x.append(curr_x)
            new_node.path_y.append(curr_y)
            
        d, _ = self._calc_dist_angle(self.Node(curr_x, curr_y), to_node)
        if d <= self.path_resolution:
            new_node.path_x.append(to_node.x)
            new_node.path_y.append(to_node.y)
            new_node.x = to_node.x
            new_node.y = to_node.y

        new_node.parent = from_node
        return new_node

    def _get_random_node(self):
        if random.randint(0, 100) > self.goal_sample_rate:
            rnd = self.Node(
                random.uniform(self.minx, self.maxx),
                random.uniform(self.miny, self.maxy))
        else:
            rnd = self.Node(self.end.x, self.end.y)
        return rnd

    def _get_nearest_node_index(self, node_list, rnd_node):
        dlist = [(node.x - rnd_node.x)**2 + (node.y - rnd_node.y)**2 for node in node_list]
        return dlist.index(min(dlist))

    def _find_near_nodes(self, new_node):
        nnode = len(self.node_list) + 1
        r = 50.0 * math.sqrt(math.log(nnode) / nnode)  # Connect radius
        r = min(r, self.expand_dis * 2.0)
        dlist = [(node.x - new_node.x)**2 + (node.y - new_node.y)**2 for node in self.node_list]
        return [i for i, d in enumerate(dlist) if d <= r**2]

    def _choose_parent(self, new_node, near_inds):
        if not near_inds: return None
        
        costs = []
        valid_inds = []
        
        for i in near_inds:
            near_node = self.node_list[i]
            t_node = self._steer(near_node, new_node)
            if t_node and self._check_collision_path(t_node):
                costs.append(self._calc_cost(near_node, new_node))
                valid_inds.append(i)
        
        if not costs: return None
        
        min_cost = min(costs)
        min_ind = valid_inds[costs.index(min_cost)]
        
        parent_node = self.node_list[min_ind]
        new_node = self._steer(parent_node, new_node)
        new_node.cost = min_cost
        return new_node

    def _rewire(self, new_node, near_inds):
        for i in near_inds:
            near_node = self.node_list[i]
            edge_node = self._steer(new_node, near_node)
            if not edge_node: continue
            
            edge_node.cost = self._calc_cost(new_node, near_node)
            
            if self._check_collision_path(edge_node) and near_node.cost > edge_node.cost:
                # Update parent
                self.node_list[i] = edge_node
                self.node_list[i].parent = new_node

    def _calc_cost(self, from_node, to_node):
        d, _ = self._calc_dist_angle(from_node, to_node)
        return from_node.cost + d

    def _calc_dist_angle(self, from_node, to_node):
        dx = to_node.x - from_node.x
        dy = to_node.y - from_node.y
        return math.hypot(dx, dy), math.atan2(dy, dx)

    def _generate_final_course(self, goal_ind):
        path = [[self.end.x, self.end.y]]
        node = self.node_list[goal_ind]
        while node.parent is not None:
            path.append([node.x, node.y])
            node = node.parent
        path.append([node.x, node.y])
        return path[::-1] # Reverse

    def smooth_path(self, path, max_iter=50):
        """Simple path short-cutting."""
        if len(path) <= 2: return path
        
        for _ in range(max_iter):
            if len(path) <= 2: break
            # Pick two random indices
            i = random.randint(0, len(path)-2)
            j = random.randint(i+1, len(path)-1)
            
            if j - i <= 1: continue
            
            p1 = path[i]
            p2 = path[j]
            
            # Check discrete points along the line
            dist = math.hypot(p1[0]-p2[0], p1[1]-p2[1])
            steps = int(dist / self.resolution)
            collision = False
            for k in range(1, steps):
                t = k / steps
                x = p1[0] + (p2[0] - p1[0]) * t
                y = p1[1] + (p2[1] - p1[1]) * t
                if self._is_in_collision(x, y):
                    collision = True
                    break
            
            if not collision:
                # Remove intermediate points
                path = path[:i+1] + path[j:]
                
        return path