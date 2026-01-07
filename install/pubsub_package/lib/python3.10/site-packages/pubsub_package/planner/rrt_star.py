#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
RRT* Path Planning Algorithm
Based on PythonRobotics by AtsushiSakai(@Atsushi_twi)
Adapted for ROS2 navigation framework
"""

import random
import math
import numpy as np

try:
    from scipy.spatial import cKDTree as KDTree  # type: ignore
except Exception:  # pragma: no cover
    KDTree = None


class RRT_star:
    """
    RRT* Path Planning Class
    """

    class Node:
        """RRT* Node"""
        def __init__(self, x, y):
            self.x = x
            self.y = y
            self.path_x = []
            self.path_y = []
            self.parent = None
            self.cost = 0.0

    def __init__(self, minx, maxx, miny, maxy, obstacles, robot_size=0.1, safe_dist=0.05,
                 expand_dis=0.5, path_resolution=0.05, goal_sample_rate=10, max_iter=5000,
                 connect_circle_dist=2.0, search_until_max_iter=False):
        """
        Initialize RRT* planner

        Args:
            minx, maxx, miny, maxy: Map boundaries
            obstacles: List of obstacle points [(x, y), ...]
            robot_size: Robot radius for collision checking
            safe_dist: Additional safety distance
            expand_dis: Maximum extension distance per step
            path_resolution: Resolution of path interpolation
            goal_sample_rate: Probability (%) of sampling goal point
            max_iter: Maximum iterations
            connect_circle_dist: Base distance for rewiring neighbors
            search_until_max_iter: If True, continue search until max_iter
        """
        self.minx = minx
        self.maxx = maxx
        self.miny = miny
        self.maxy = maxy
        self.obstacles = obstacles
        self.robot_radius = robot_size + safe_dist
        self.expand_dis = expand_dis
        self.path_resolution = path_resolution
        self.goal_sample_rate = goal_sample_rate
        self.max_iter = max_iter
        self.connect_circle_dist = connect_circle_dist
        self.search_until_max_iter = search_until_max_iter
        self.node_list = []

        self._obstacle_points = self._normalize_obstacles(obstacles)
        self._obstacle_kdtree = KDTree(self._obstacle_points) if (KDTree and len(self._obstacle_points) > 0) else None

    @staticmethod
    def _normalize_obstacles(obstacles):
        if not obstacles:
            return np.empty((0, 2), dtype=float)

        points = []
        for obs in obstacles:
            if obs is None:
                continue
            if len(obs) < 2:
                continue
            ox, oy = float(obs[0]), float(obs[1])
            if not np.isfinite(ox) or not np.isfinite(oy):
                continue
            points.append((ox, oy))

        if not points:
            return np.empty((0, 2), dtype=float)

        return np.asarray(points, dtype=float)

    def plan(self, sx, sy, gx, gy):
        """
        Plan path from start to goal

        Args:
            sx, sy: Start position
            gx, gy: Goal position

        Returns:
            (is_found, path): Tuple of success flag and path list [(x, y), ...]
        """
        if sx < self.minx or sx > self.maxx or sy < self.miny or sy > self.maxy:
            return False, []
        if gx < self.minx or gx > self.maxx or gy < self.miny or gy > self.maxy:
            return False, []
        if self._is_point_collision(sx, sy) or self._is_point_collision(gx, gy):
            return False, []

        self.start = self.Node(sx, sy)
        self.end = self.Node(gx, gy)
        self.goal_node = self.Node(gx, gy)

        self.node_list = [self.start]

        for i in range(self.max_iter):
            rnd = self._get_random_node()
            nearest_ind = self._get_nearest_node_index(self.node_list, rnd)
            new_node = self._steer(self.node_list[nearest_ind], rnd, self.expand_dis)
            near_node = self.node_list[nearest_ind]

            new_node.cost = near_node.cost + math.hypot(new_node.x - near_node.x,
                                                         new_node.y - near_node.y)

            if self._check_collision(new_node):
                near_inds = self._find_near_nodes(new_node)
                node_with_updated_parent = self._choose_parent(new_node, near_inds)
                if node_with_updated_parent:
                    self._rewire(node_with_updated_parent, near_inds)
                    self.node_list.append(node_with_updated_parent)
                else:
                    self.node_list.append(new_node)

            if not self.search_until_max_iter:
                last_index = self._search_best_goal_node()
                if last_index is not None:
                    path = self._generate_final_course(last_index)
                    return True, path

        # Reached max iteration, try to find best path
        last_index = self._search_best_goal_node()
        if last_index is not None:
            path = self._generate_final_course(last_index)
            return True, path

        return False, []

    def _steer(self, from_node, to_node, extend_length=float("inf")):
        """Steer from one node towards another"""
        new_node = self.Node(from_node.x, from_node.y)
        d, theta = self._calc_distance_and_angle(new_node, to_node)

        new_node.path_x = [new_node.x]
        new_node.path_y = [new_node.y]

        if extend_length > d:
            extend_length = d

        n_expand = math.floor(extend_length / self.path_resolution)

        for _ in range(n_expand):
            new_node.x += self.path_resolution * math.cos(theta)
            new_node.y += self.path_resolution * math.sin(theta)
            new_node.path_x.append(new_node.x)
            new_node.path_y.append(new_node.y)

        d, _ = self._calc_distance_and_angle(new_node, to_node)
        if d <= self.path_resolution:
            new_node.path_x.append(to_node.x)
            new_node.path_y.append(to_node.y)
            new_node.x = to_node.x
            new_node.y = to_node.y

        new_node.parent = from_node

        return new_node

    def _generate_final_course(self, goal_ind):
        """Generate final path from goal to start"""
        path = [[self.end.x, self.end.y]]
        node = self.node_list[goal_ind]
        while node.parent is not None:
            path.append([node.x, node.y])
            node = node.parent
        path.append([node.x, node.y])
        path.reverse()  # Reverse to get start -> goal order
        return path

    def _calc_dist_to_goal(self, x, y):
        """Calculate distance to goal"""
        dx = x - self.end.x
        dy = y - self.end.y
        return math.hypot(dx, dy)

    def _get_random_node(self):
        """Get random node, with goal biasing"""
        if random.randint(0, 100) > self.goal_sample_rate:
            rnd = self.Node(
                random.uniform(self.minx, self.maxx),
                random.uniform(self.miny, self.maxy))
        else:
            rnd = self.Node(self.end.x, self.end.y)
        return rnd

    @staticmethod
    def _get_nearest_node_index(node_list, rnd_node):
        """Find index of nearest node"""
        dlist = [(node.x - rnd_node.x) ** 2 + (node.y - rnd_node.y) ** 2
                 for node in node_list]
        minind = dlist.index(min(dlist))
        return minind

    def _check_collision(self, node):
        """Check if node path is collision-free"""
        if node is None:
            return False

        if not node.path_x:
            return False

        if self._obstacle_kdtree is not None:
            pts = np.column_stack((node.path_x, node.path_y))
            dists, _ = self._obstacle_kdtree.query(pts, k=1)
            return bool(np.all(dists > self.robot_radius))

        for (px, py) in zip(node.path_x, node.path_y):
            for (ox, oy) in self.obstacles:
                d = math.hypot(ox - px, oy - py)
                if d <= self.robot_radius:
                    return False  # Collision

        return True  # Safe

    def _check_if_inside_play_area(self, node):
        """Check if node is inside the map boundaries"""
        if node.x < self.minx or node.x > self.maxx or \
           node.y < self.miny or node.y > self.maxy:
            return False
        return True

    @staticmethod
    def _calc_distance_and_angle(from_node, to_node):
        """Calculate distance and angle between two nodes"""
        dx = to_node.x - from_node.x
        dy = to_node.y - from_node.y
        d = math.hypot(dx, dy)
        theta = math.atan2(dy, dx)
        return d, theta

    def _choose_parent(self, new_node, near_inds):
        """Choose best parent for new node from nearby nodes"""
        if not near_inds:
            return None

        costs = []
        for i in near_inds:
            near_node = self.node_list[i]
            t_node = self._steer(near_node, new_node)
            if t_node and self._check_collision(t_node):
                costs.append(self._calc_new_cost(near_node, new_node))
            else:
                costs.append(float("inf"))

        min_cost = min(costs)

        if min_cost == float("inf"):
            return None

        min_ind = near_inds[costs.index(min_cost)]
        new_node = self._steer(self.node_list[min_ind], new_node)
        new_node.cost = min_cost

        return new_node

    def _search_best_goal_node(self):
        """Search for best node near goal"""
        dist_to_goal_list = [
            self._calc_dist_to_goal(n.x, n.y) for n in self.node_list
        ]
        goal_inds = [i for i, dist in enumerate(dist_to_goal_list) if dist <= self.expand_dis]

        safe_goal_inds = []
        for goal_ind in goal_inds:
            t_node = self._steer(self.node_list[goal_ind], self.goal_node)
            if self._check_collision(t_node):
                safe_goal_inds.append(goal_ind)

        if not safe_goal_inds:
            return None

        safe_goal_costs = [
            self.node_list[i].cost + self._calc_dist_to_goal(self.node_list[i].x, self.node_list[i].y)
            for i in safe_goal_inds
        ]

        min_cost = min(safe_goal_costs)
        for i, cost in zip(safe_goal_inds, safe_goal_costs):
            if cost == min_cost:
                return i

        return None

    def _find_near_nodes(self, new_node):
        """Find all nodes within rewiring radius"""
        nnode = len(self.node_list) + 1
        r = self.connect_circle_dist * math.sqrt(math.log(nnode) / nnode)
        r = min(r, self.expand_dis)

        dist_list = [(node.x - new_node.x) ** 2 + (node.y - new_node.y) ** 2
                     for node in self.node_list]
        near_inds = [i for i, dist in enumerate(dist_list) if dist <= r ** 2]
        return near_inds

    def _rewire(self, new_node, near_inds):
        """Rewire nearby nodes if cheaper path through new_node"""
        for i in near_inds:
            near_node = self.node_list[i]
            edge_node = self._steer(new_node, near_node)
            if not edge_node:
                continue
            edge_node.cost = self._calc_new_cost(new_node, near_node)

            no_collision = self._check_collision(edge_node)
            improved_cost = near_node.cost > edge_node.cost

            if no_collision and improved_cost:
                for node in self.node_list:
                    if node.parent == self.node_list[i]:
                        node.parent = edge_node
                self.node_list[i] = edge_node
                self._propagate_cost_to_leaves(self.node_list[i])

    def _calc_new_cost(self, from_node, to_node):
        """Calculate cost from from_node to to_node"""
        d, _ = self._calc_distance_and_angle(from_node, to_node)
        return from_node.cost + d

    def _propagate_cost_to_leaves(self, parent_node):
        """Propagate cost updates to child nodes"""
        for node in self.node_list:
            if node.parent == parent_node:
                node.cost = self._calc_new_cost(parent_node, node)
                self._propagate_cost_to_leaves(node)

    # ==================== Path Smoothing ====================

    def smooth_path(self, path, max_iter=1000):
        """
        Smooth a given path by iteratively replacing segments with shortcut connections.

        Args:
            path: Original path as list of [x, y] coordinates
            max_iter: Number of smoothing iterations

        Returns:
            Smoothed path as list of [x, y] coordinates
        """
        if len(path) <= 2:
            return path

        smoothed_path = [p[:] for p in path]  # Deep copy

        for _ in range(max_iter):
            path_length = self._get_path_length(smoothed_path)
            if path_length < 1e-6:
                break

            # Sample two random points along the path
            pick_points = sorted([random.uniform(0, path_length), 
                                  random.uniform(0, path_length)])

            first = self._get_target_point(smoothed_path, pick_points[0])
            second = self._get_target_point(smoothed_path, pick_points[1])

            if first[2] <= 0 or second[2] <= 0:
                continue
            if (second[2] + 1) > len(smoothed_path):
                continue
            if second[2] == first[2]:
                continue

            # Collision check for the shortcut
            if not self._line_collision_check(first[:2], second[:2]):
                continue

            # Create new path with shortcut
            new_path = []
            new_path.extend(smoothed_path[:first[2] + 1])
            new_path.append([first[0], first[1]])
            new_path.append([second[0], second[1]])
            new_path.extend(smoothed_path[second[2] + 1:])
            smoothed_path = new_path

        return smoothed_path

    @staticmethod
    def _get_path_length(path):
        """Calculate total length of a path"""
        length = 0
        for i in range(len(path) - 1):
            dx = path[i + 1][0] - path[i][0]
            dy = path[i + 1][1] - path[i][1]
            length += math.hypot(dx, dy)
        return length

    @staticmethod
    def _get_target_point(path, target_length):
        """
        Get a point at a specific distance along the path.

        Returns:
            [x, y, segment_index]
        """
        length = 0
        ti = 0
        last_pair_len = 0

        for i in range(len(path) - 1):
            dx = path[i + 1][0] - path[i][0]
            dy = path[i + 1][1] - path[i][1]
            d = math.hypot(dx, dy)
            length += d
            if length >= target_length:
                ti = i
                last_pair_len = d
                break

        if last_pair_len == 0:
            return [path[-1][0], path[-1][1], len(path) - 1]

        part_ratio = (length - target_length) / last_pair_len
        x = path[ti][0] + (path[ti + 1][0] - path[ti][0]) * (1 - part_ratio)
        y = path[ti][1] + (path[ti + 1][1] - path[ti][1]) * (1 - part_ratio)

        return [x, y, ti]

    def _line_collision_check(self, first, second, sample_step=0.1):
        """
        Check if line segment between two points is collision-free.

        Args:
            first: Start point [x, y]
            second: End point [x, y]
            sample_step: Distance between sampling points

        Returns:
            True if collision-free, False otherwise
        """
        x1, y1 = first[0], first[1]
        x2, y2 = second[0], second[1]

        dx = x2 - x1
        dy = y2 - y1
        length = math.hypot(dx, dy)

        if length == 0:
            return not self._is_point_collision(x1, y1)

        steps = int(length / sample_step) + 1

        if self._obstacle_kdtree is not None:
            ts = np.linspace(0.0, 1.0, steps + 1)
            xs = x1 + ts * dx
            ys = y1 + ts * dy
            pts = np.column_stack((xs, ys))
            dists, _ = self._obstacle_kdtree.query(pts, k=1)
            return bool(np.all(dists > self.robot_radius))

        for i in range(steps + 1):
            t = i / steps
            x = x1 + t * dx
            y = y1 + t * dy

            if self._is_point_collision(x, y):
                return False

        return True

    def _is_point_collision(self, x, y):
        """
        Check if a single point collides with any obstacle.

        Args:
            x, y: Point coordinates

        Returns:
            True if collision, False otherwise
        """
        if self._obstacle_kdtree is not None:
            dist, _ = self._obstacle_kdtree.query([[x, y]], k=1)
            return bool(dist[0] <= self.robot_radius)

        for (ox, oy) in self.obstacles:
            d = math.hypot(ox - x, oy - y)
            if d <= self.robot_radius:
                return True
        return False
