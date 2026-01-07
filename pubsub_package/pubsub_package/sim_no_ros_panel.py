#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import base64
import json
import math
import threading
import time
import traceback
import webbrowser
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np

from pubsub_package.planner.dwa import DWA
from pubsub_package.planner.rrt_star import RRT_star
from pubsub_package.sim_no_ros import (
    MapSpec,
    cast_ray,
    footprint_collision,
    is_occupied,
    load_map_yaml,
    procedural_map,
)


def _normalize_angle(angle: float) -> float:
    while angle > math.pi:
        angle -= 2.0 * math.pi
    while angle < -math.pi:
        angle += 2.0 * math.pi
    return angle


def _inflate_bool_grid(occ: np.ndarray, radius_cells: int) -> np.ndarray:
    r = int(max(radius_cells, 0))
    if r == 0:
        return occ
    inflated = np.zeros_like(occ, dtype=bool)

    offsets = []
    r2 = r * r
    for dy in range(-r, r + 1):
        for dx in range(-r, r + 1):
            if dx * dx + dy * dy <= r2:
                offsets.append((dy, dx))

    rows, cols = occ.shape
    for dy, dx in offsets:
        src_r0 = max(0, -dy)
        src_r1 = min(rows, rows - dy)
        src_c0 = max(0, -dx)
        src_c1 = min(cols, cols - dx)

        dst_r0 = src_r0 + dy
        dst_r1 = src_r1 + dy
        dst_c0 = src_c0 + dx
        dst_c1 = src_c1 + dx

        inflated[dst_r0:dst_r1, dst_c0:dst_c1] |= occ[src_r0:src_r1, src_c0:src_c1]
    return inflated


def _iter_obstacles_from_grid(spec: MapSpec, stride: int = 2):
    stride = max(int(stride), 1)
    occ = np.argwhere(spec.grid >= 50)
    if stride > 1:
        occ = occ[::stride]
    for iy, ix in occ:
        x = spec.origin_x + (float(ix) + 0.5) * spec.resolution
        y = spec.origin_y + (float(iy) + 0.5) * spec.resolution
        yield (x, y)


def _world_to_robot(x: float, y: float, yaw: float, gx: float, gy: float) -> Tuple[float, float]:
    dx = gx - x
    dy = gy - y
    c = math.cos(-yaw)
    s = math.sin(-yaw)
    return (c * dx - s * dy, s * dx + c * dy)


def _closest_path_index(path: np.ndarray, x: float, y: float) -> int:
    dx = path[:, 0] - x
    dy = path[:, 1] - y
    return int(np.argmin(dx * dx + dy * dy))


def _lookahead_goal(path: np.ndarray, start_idx: int, lookahead_dist: float) -> int:
    if len(path) < 2:
        return 0
    acc = 0.0
    goal_idx = min(start_idx + 1, len(path) - 1)
    for i in range(start_idx, len(path) - 1):
        seg = math.hypot(path[i + 1, 0] - path[i, 0], path[i + 1, 1] - path[i, 1])
        acc += seg
        if acc >= lookahead_dist:
            goal_idx = i + 1
            break
    return goal_idx


def _laser_points_robot_frame(
    spec: MapSpec,
    x: float,
    y: float,
    yaw: float,
    *,
    scan_count: int,
    angle_min: float,
    angle_max: float,
    range_min: float,
    range_max: float,
    obstacle_threshold: float,
) -> list[tuple[float, float]]:
    scan_count = max(int(scan_count), 1)
    angle_inc = (angle_max - angle_min) / float(scan_count)
    step = max(spec.resolution * 0.5, 0.02)

    points: list[tuple[float, float]] = []
    for i in range(scan_count):
        a = angle_min + float(i) * angle_inc
        d = cast_ray(spec, x, y, yaw + a, range_min, range_max, step)
        if not math.isfinite(d):
            continue
        if d >= obstacle_threshold:
            continue
        points.append((math.cos(a) * d, math.sin(a) * d))
    return points


def run_episode(spec: MapSpec, params: Dict[str, Any]) -> Dict[str, Any]:
    start = params["start"]  # [x,y,yaw]
    goal = params["goal"]  # [x,y]

    robot_radius = float(params["robot_radius"])
    safety_dist = float(params["safety_dist"])
    inflation_radius = float(params["inflation_radius"])

    collision_radius = max(robot_radius + safety_dist, 0.0)
    plan_inflation_radius = max(collision_radius + inflation_radius, 0.0)

    start_collision = footprint_collision(spec, float(start[0]), float(start[1]), collision_radius)
    goal_collision = footprint_collision(spec, float(goal[0]), float(goal[1]), collision_radius)
    if start_collision or goal_collision:
        return {
            "success": False,
            "path": [],
            "traj": [],
            "error": "Start/Goal in collision",
            "debug": {
                "start_collision": bool(start_collision),
                "goal_collision": bool(goal_collision),
                "robot_radius": float(robot_radius),
                "safety_dist": float(safety_dist),
                "collision_radius": float(collision_radius),
            },
        }

    occ = (spec.grid != 0)
    inflation_cells = int(math.ceil(plan_inflation_radius / max(spec.resolution, 1e-6)))
    occ_inflated = _inflate_bool_grid(occ, inflation_cells)
    grid2 = spec.grid.copy()
    grid2[occ_inflated] = 100
    spec2 = MapSpec(
        resolution=spec.resolution,
        origin_x=spec.origin_x,
        origin_y=spec.origin_y,
        origin_yaw=spec.origin_yaw,
        grid=grid2,
    )

    start_in_inflated = is_occupied(spec2, float(start[0]), float(start[1]))
    goal_in_inflated = is_occupied(spec2, float(goal[0]), float(goal[1]))
    if start_in_inflated or goal_in_inflated:
        return {
            "success": False,
            "path": [],
            "traj": [],
            "error": "Start/Goal in inflated area",
            "debug": {
                "start_in_inflated": bool(start_in_inflated),
                "goal_in_inflated": bool(goal_in_inflated),
                "inflation_cells": int(inflation_cells),
                "plan_inflation_radius": float(plan_inflation_radius),
                "collision_radius": float(collision_radius),
            },
        }

    obstacles = list(_iter_obstacles_from_grid(spec2, stride=int(params["obstacle_stride"])))
    obstacles.append((-9999.0, -9999.0))

    rrt = RRT_star(
        minx=spec2.minx,
        maxx=spec2.maxx,
        miny=spec2.miny,
        maxy=spec2.maxy,
        obstacles=obstacles,
        robot_size=0.0,
        safe_dist=max(float(spec.resolution), 1e-6),
        expand_dis=float(params["rrt.expand_dis"]),
        path_resolution=float(params["rrt.path_resolution"]),
        goal_sample_rate=int(params["rrt.goal_sample_rate"]),
        max_iter=int(params["rrt.max_iter"]),
        connect_circle_dist=float(params["rrt.connect_circle_dist"]),
        search_until_max_iter=bool(params["rrt.search_until_max_iter"]),
    )

    found, path = rrt.plan(start[0], start[1], goal[0], goal[1])
    if not found or not path:
        return {
            "success": False,
            "path": [],
            "traj": [],
            "error": "RRT* failed",
            "debug": {
                "start_in_inflated": bool(start_in_inflated),
                "goal_in_inflated": bool(goal_in_inflated),
                "inflation_cells": int(inflation_cells),
                "plan_inflation_radius": float(plan_inflation_radius),
                "collision_radius": float(collision_radius),
                "obstacle_points": int(len(obstacles)),
                "rrt": {
                    "expand_dis": float(params["rrt.expand_dis"]),
                    "path_resolution": float(params["rrt.path_resolution"]),
                    "goal_sample_rate": int(params["rrt.goal_sample_rate"]),
                    "max_iter": int(params["rrt.max_iter"]),
                    "connect_circle_dist": float(params["rrt.connect_circle_dist"]),
                    "search_until_max_iter": bool(params["rrt.search_until_max_iter"]),
                },
            },
        }

    if int(params["rrt.smooth_iter"]) > 0:
        path = rrt.smooth_path(path, max_iter=int(params["rrt.smooth_iter"]))

    path_np = np.asarray(path, dtype=float)

    dwa = DWA()
    max_speed = float(params["max_speed"])
    max_yawrate = float(params["max_yawrate"])
    dwa.config(max_speed=max_speed, max_yawrate=max_yawrate, base=collision_radius)
    dwa.to_goal_cost_gain = float(params["dwa.to_goal_cost_gain"])
    dwa.path_cost_gain = float(params["dwa.path_cost_gain"])
    dwa.heading_cost_gain = float(params["dwa.heading_cost_gain"])
    dwa.obstacle_cost_gain = float(params["dwa.obstacle_cost_gain"])
    dwa.speed_cost_gain = float(params["dwa.speed_cost_gain"])
    dwa.predict_time = float(params["dwa.predict_time"])
    dwa.v_samples = int(params["dwa.v_samples"])
    dwa.w_samples = int(params["dwa.w_samples"])

    dt = float(params["dt"])
    max_steps = int(params["max_steps"])
    arrive = float(params["arrive_dist"])
    lookahead = float(params["lookahead_dist"])
    local_path_max_points = int(params["local_path_max_points"])
    obstacle_threshold = float(params["laser_threshold"])

    x, y, yaw = float(start[0]), float(start[1]), float(start[2])
    v, w = 0.0, 0.0
    traj = [[x, y, yaw]]

    for _ in range(max_steps):
        if math.hypot(goal[0] - x, goal[1] - y) <= arrive:
            return {"success": True, "path": path, "traj": traj, "error": ""}

        if footprint_collision(spec, x, y, collision_radius):
            return {"success": False, "path": path, "traj": traj, "error": "Collision"}

        near_idx = _closest_path_index(path_np, x, y)
        goal_idx = _lookahead_goal(path_np, near_idx, lookahead)
        gx, gy = float(path_np[goal_idx, 0]), float(path_np[goal_idx, 1])
        goal_rx, goal_ry = _world_to_robot(x, y, yaw, gx, gy)

        end_idx = min(near_idx + max(local_path_max_points, 2), len(path_np))
        local_path = [_world_to_robot(x, y, yaw, float(path_np[i, 0]), float(path_np[i, 1])) for i in range(near_idx, end_idx)]

        points = _laser_points_robot_frame(
            spec,
            x,
            y,
            yaw,
            scan_count=int(params["scan_count"]),
            angle_min=-math.pi,
            angle_max=math.pi,
            range_min=0.05,
            range_max=float(params["scan_range_max"]),
            obstacle_threshold=obstacle_threshold,
        )

        v, w = dwa.planning(
            pose=(0.0, 0.0, 0.0),
            velocity=(v, w),
            goal=(goal_rx, goal_ry),
            points_cloud=points,
            path_points=local_path,
        )

        yaw = _normalize_angle(yaw + float(w) * dt)
        nx = x + float(v) * math.cos(yaw) * dt
        ny = y + float(v) * math.sin(yaw) * dt
        if not footprint_collision(spec, nx, ny, collision_radius):
            x, y = nx, ny
        traj.append([x, y, yaw])

    return {"success": False, "path": path, "traj": traj, "error": "Max steps reached"}


def _default_params() -> Dict[str, Any]:
    return {
        "start": [0.0, 0.0, 0.0],
        "goal": [2.0, 0.0],
        "dt": 0.1,
        "max_steps": 2500,
        "arrive_dist": 0.2,
        "robot_radius": 0.2,
        "safety_dist": 0.05,
        "inflation_radius": 0.12,
        "max_speed": 0.5,
        "max_yawrate": 1.0,
        "lookahead_dist": 1.5,
        "local_path_max_points": 30,
        "scan_count": 360,
        "scan_range_max": 8.0,
        "laser_threshold": 1.5,
        "obstacle_stride": 2,
        "rrt.expand_dis": 0.6,
        "rrt.path_resolution": 0.05,
        "rrt.goal_sample_rate": 10,
        "rrt.max_iter": 6000,
        "rrt.connect_circle_dist": 2.0,
        "rrt.search_until_max_iter": False,
        "rrt.smooth_iter": 200,
        "dwa.to_goal_cost_gain": 1.2,
        "dwa.path_cost_gain": 0.4,
        "dwa.heading_cost_gain": 0.05,
        "dwa.obstacle_cost_gain": 0.03,
        "dwa.speed_cost_gain": 1.5,
        "dwa.predict_time": 1.5,
        "dwa.v_samples": 12,
        "dwa.w_samples": 26,
    }


@dataclass
class Job:
    id: str
    status: str  # "idle" | "running" | "done" | "error"
    started_ts: float = 0.0
    finished_ts: float = 0.0
    result: Optional[Dict[str, Any]] = None
    error: str = ""
    traceback: str = ""


class SimPanelApp:
    def __init__(self, spec: MapSpec):
        self.spec = spec
        self.params = _default_params()
        self._lock = threading.Lock()

        self.job = Job(id="0", status="idle")
        self.last_result: Optional[Dict[str, Any]] = None

    def set_params(self, patch: Dict[str, Any]) -> None:
        with self._lock:
            for k, v in patch.items():
                if k in ("start", "goal"):
                    self.params[k] = v
                else:
                    if k in self.params:
                        self.params[k] = v

    def start_run(self) -> str:
        with self._lock:
            job_id = str(int(time.time() * 1000))
            self.job = Job(id=job_id, status="running", started_ts=time.time())
            params = json.loads(json.dumps(self.params))
        t = threading.Thread(target=self._run_job, args=(job_id, params), daemon=True)
        t.start()
        return job_id

    def _run_job(self, job_id: str, params: Dict[str, Any]) -> None:
        try:
            result = run_episode(self.spec, params)
            with self._lock:
                if self.job.id != job_id:
                    return
                self.job.status = "done"
                self.job.finished_ts = time.time()
                self.job.result = result
                self.last_result = result
        except Exception as e:
            tb = traceback.format_exc()
            with self._lock:
                if self.job.id != job_id:
                    return
                self.job.status = "error"
                self.job.finished_ts = time.time()
                self.job.error = str(e)
                self.job.traceback = tb

    def state(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "params": self.params,
                "job": {
                    "id": self.job.id,
                    "status": self.job.status,
                    "started_ts": self.job.started_ts,
                    "finished_ts": self.job.finished_ts,
                    "error": self.job.error,
                },
                "last_result": self._result_summary(self.last_result),
                "map": {
                    "resolution": self.spec.resolution,
                    "origin_x": self.spec.origin_x,
                    "origin_y": self.spec.origin_y,
                    "width": int(self.spec.width),
                    "height": int(self.spec.height),
                },
            }

    @staticmethod
    def _result_summary(result: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        if result is None:
            return None
        return {
            "success": bool(result.get("success")),
            "error": str(result.get("error", "")),
            "path_len": int(len(result.get("path", []) or [])),
            "traj_len": int(len(result.get("traj", []) or [])),
        }

    def job_result(self, job_id: str) -> Dict[str, Any]:
        with self._lock:
            if self.job.id != job_id:
                return {"ok": False, "error": "job not found"}
            if self.job.status in ("done", "error"):
                return {
                    "ok": True,
                    "job": {
                        "id": self.job.id,
                        "status": self.job.status,
                        "error": self.job.error,
                        "started_ts": self.job.started_ts,
                        "finished_ts": self.job.finished_ts,
                    },
                    "result": self.job.result if self.job.result is not None else None,
                    "traceback": self.job.traceback,
                }
            return {"ok": True, "job": {"id": self.job.id, "status": self.job.status}}

    def map_payload(self, downsample: int = 2) -> Dict[str, Any]:
        ds = max(int(downsample), 1)
        occ = (self.spec.grid >= 50).astype(np.uint8)
        unk = (self.spec.grid < 0).astype(np.uint8)
        if ds > 1:
            occ = occ[::ds, ::ds]
            unk = unk[::ds, ::ds]
        packed = np.packbits(occ.reshape(-1), bitorder="little")
        b64 = base64.b64encode(packed.tobytes()).decode("ascii")
        packed_unk = np.packbits(unk.reshape(-1), bitorder="little")
        b64_unk = base64.b64encode(packed_unk.tobytes()).decode("ascii")
        return {
            "downsample": ds,
            "width": int(occ.shape[1]),
            "height": int(occ.shape[0]),
            "resolution": float(self.spec.resolution * ds),
            "origin_x": float(self.spec.origin_x),
            "origin_y": float(self.spec.origin_y),
            "occ_packbits_le_b64": b64,
            "unk_packbits_le_b64": b64_unk,
            "packed_bits": True,
        }

    def map_overlay_payload(self, downsample: int = 2) -> Dict[str, Any]:
        ds = max(int(downsample), 1)
        with self._lock:
            rr = float(self.params.get("robot_radius", 0.2))
            sd = float(self.params.get("safety_dist", 0.0))
            ir = float(self.params.get("inflation_radius", 0.0))
        collision_radius = max(rr + sd, 0.0)
        plan_inflation_radius = max(collision_radius + ir, 0.0)

        occ = (self.spec.grid != 0).astype(bool)  # treat unknown as obstacle (matches planning)
        if ds > 1:
            occ = occ[::ds, ::ds]
        res = float(self.spec.resolution * ds)
        inflation_cells = int(math.ceil(plan_inflation_radius / max(res, 1e-6)))
        inflated = _inflate_bool_grid(occ, inflation_cells).astype(np.uint8)

        packed = np.packbits(inflated.reshape(-1), bitorder="little")
        b64 = base64.b64encode(packed.tobytes()).decode("ascii")
        return {
            "downsample": ds,
            "width": int(inflated.shape[1]),
            "height": int(inflated.shape[0]),
            "resolution": float(self.spec.resolution * ds),
            "origin_x": float(self.spec.origin_x),
            "origin_y": float(self.spec.origin_y),
            "occ_packbits_le_b64": b64,
            "packed_bits": True,
            "collision_radius": float(collision_radius),
            "plan_inflation_radius": float(plan_inflation_radius),
            "inflation_cells": int(inflation_cells),
        }

    def _planning_spec_from_params(self, params: Dict[str, Any]) -> Tuple[MapSpec, float, float, int]:
        rr = float(params.get("robot_radius", 0.2))
        sd = float(params.get("safety_dist", 0.0))
        ir = float(params.get("inflation_radius", 0.0))
        collision_radius = max(rr + sd, 0.0)
        plan_inflation_radius = max(collision_radius + ir, 0.0)

        occ = (self.spec.grid != 0).astype(bool)  # treat unknown as obstacle (matches planning)
        inflation_cells = int(math.ceil(plan_inflation_radius / max(float(self.spec.resolution), 1e-6)))
        occ_inflated = _inflate_bool_grid(occ, inflation_cells)
        grid2 = self.spec.grid.copy()
        grid2[occ_inflated] = 100
        spec2 = MapSpec(
            resolution=self.spec.resolution,
            origin_x=self.spec.origin_x,
            origin_y=self.spec.origin_y,
            origin_yaw=self.spec.origin_yaw,
            grid=grid2,
        )
        return spec2, collision_radius, plan_inflation_radius, inflation_cells

    def pose_check_payload(self) -> Dict[str, Any]:
        with self._lock:
            params = json.loads(json.dumps(self.params))

        spec2, collision_radius, plan_inflation_radius, inflation_cells = self._planning_spec_from_params(params)
        start = params["start"]
        goal = params["goal"]

        def _sample(x: float, y: float, r: float, n: int = 32) -> list[dict[str, Any]]:
            out: list[dict[str, Any]] = []
            for i in range(n):
                a = (2.0 * math.pi) * (float(i) / float(n))
                px = x + r * math.cos(a)
                py = y + r * math.sin(a)
                out.append({"x": float(px), "y": float(py), "occ": bool(is_occupied(self.spec, px, py))})
            return out

        sx, sy = float(start[0]), float(start[1])
        gx, gy = float(goal[0]), float(goal[1])
        start_center_occ = bool(is_occupied(self.spec, sx, sy))
        goal_center_occ = bool(is_occupied(self.spec, gx, gy))
        start_center_infl = bool(is_occupied(spec2, sx, sy))
        goal_center_infl = bool(is_occupied(spec2, gx, gy))

        start_samples = _sample(sx, sy, collision_radius)
        goal_samples = _sample(gx, gy, collision_radius)

        start_hit = start_center_occ or any(p["occ"] for p in start_samples)
        goal_hit = goal_center_occ or any(p["occ"] for p in goal_samples)

        return {
            "ok": True,
            "collision_radius": float(collision_radius),
            "plan_inflation_radius": float(plan_inflation_radius),
            "inflation_cells": int(inflation_cells),
            "start": {
                "x": sx,
                "y": sy,
                "yaw": float(start[2]),
                "center_occ": start_center_occ,
                "center_in_inflated": start_center_infl,
                "hit": bool(start_hit),
                "samples": start_samples,
            },
            "goal": {
                "x": gx,
                "y": gy,
                "center_occ": goal_center_occ,
                "center_in_inflated": goal_center_infl,
                "hit": bool(goal_hit),
                "samples": goal_samples,
            },
        }


INDEX_HTML = r"""<!doctype html>
<html>
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1"/>
  <title>No-ROS Navigation Simulator</title>
  <style>
    body { font-family: system-ui, -apple-system, Segoe UI, Roboto, sans-serif; margin: 16px; }
    .grid { display: grid; grid-template-columns: 700px 1fr; gap: 16px; align-items: start; }
    .card { border: 1px solid #ddd; border-radius: 12px; padding: 12px; }
    h2 { margin: 0 0 8px 0; font-size: 16px; }
    h3 { margin: 16px 0 8px 0; font-size: 14px; }
    .row { display: grid; grid-template-columns: 170px 130px 1fr; gap: 10px; align-items: center; margin: 6px 0; }
    label { font-size: 13px; color: #222; }
    input[type=number], input[type=text] { width: 100%; padding: 6px; border: 1px solid #ccc; border-radius: 8px; }
    .desc { font-size: 12px; color: #555; line-height: 1.35; }
    .actions { display: flex; gap: 10px; margin-top: 10px; }
    button { padding: 10px 12px; border: 0; border-radius: 10px; background: #1f6feb; color: #fff; cursor: pointer; }
    button.secondary { background: #6e7781; }
    .hint { font-size: 12px; color: #666; margin-top: 6px; }
    .status { font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; font-size: 12px; white-space: pre-wrap; }
    canvas { border: 1px solid #ddd; border-radius: 12px; background: #fafafa; }
    .pill { display: inline-block; padding: 2px 8px; border-radius: 999px; background: #eef2ff; color: #3730a3; font-size: 12px; margin-left: 8px; }
    .toggles { display: flex; gap: 12px; flex-wrap: wrap; margin-top: 8px; font-size: 12px; color: #222; }
    .help { border: 1px solid #eee; background: #fafafa; border-radius: 10px; padding: 10px; font-size: 12px; color: #111; line-height: 1.5; white-space: pre-wrap; }
    .help strong { font-weight: 650; }
    @media (max-width: 980px) { .grid { grid-template-columns: 1fr; } }
  </style>
</head>
<body>
  <div class="grid">
    <div class="card">
      <h2>Map <span class="pill" id="mapInfo"></span></h2>
      <canvas id="mapCanvas" width="700" height="700"></canvas>
      <div class="hint">
        Left click: set start &nbsp;|&nbsp; Right click: set goal<br/>
        Start yaw uses the "start_yaw" field.
      </div>
      <div class="toggles">
        <label><input type="checkbox" id="showInflated" checked> 显示膨胀区</label>
        <label><input type="checkbox" id="showFootprint" checked> 显示机器人半径</label>
        <label><input type="checkbox" id="showSamples" checked> 显示碰撞采样点</label>
      </div>
      <div class="hint" id="derivedInfo"></div>
      <h3>Run</h3>
      <div class="actions">
        <button id="runBtn">Run</button>
        <button id="fitBtn" class="secondary">Fit</button>
        <button id="resetBtn" class="secondary">Reset Params</button>
      </div>
      <h3>Status</h3>
      <div class="status" id="status"></div>

      <h3>ROS2 指令导出</h3>
      <div class="toggles">
        <label><input type="checkbox" id="rosReal" checked> real:=true（真机话题）</label>
        <label><input type="checkbox" id="rosIncludeSafetyInGlobal" checked> safety 合入 global.robot_radius</label>
      </div>
      <div class="actions">
        <button id="genRosBtn" class="secondary">Generate</button>
        <button id="copyRosBtn" class="secondary">Copy</button>
      </div>
      <div class="hint">生成两条命令：`global_pub`（全局）+ `local_pub`（局部）。在 Ubuntu/WSL 里直接粘贴运行。</div>
      <textarea id="rosCmd" rows="9" style="width: 100%; padding: 8px; border: 1px solid #ddd; border-radius: 10px; font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; font-size: 12px; margin-top: 8px;" readonly></textarea>
    </div>

    <div class="card">
      <h2>Parameters</h2>
      <h3>参数说明</h3>
      <div class="help">每个参数后面都标注了用途/单位；左侧地图可显示膨胀区（橙色）与机器人半径/朝向（圈+箭头）。</div>

      <h3>Start / Goal</h3>
      <div class="row"><label>start_x</label><input id="start_x" type="number" step="0.01"><div class="desc">起点 x（米，map）。左键点地图设置。</div></div>
      <div class="row"><label>start_y</label><input id="start_y" type="number" step="0.01"><div class="desc">起点 y（米，map）。左键点地图设置。</div></div>
      <div class="row"><label>start_yaw</label><input id="start_yaw" type="number" step="0.01"><div class="desc">起点朝向 yaw（弧度）。地图会画起点箭头。</div></div>
      <div class="row"><label>goal_x</label><input id="goal_x" type="number" step="0.01"><div class="desc">目标点 x（米，map）。右键点地图设置。</div></div>
      <div class="row"><label>goal_y</label><input id="goal_y" type="number" step="0.01"><div class="desc">目标点 y（米，map）。右键点地图设置。</div></div>

      <h3>Robot / Safety</h3>
      <div class="row"><label>robot_radius</label><input id="robot_radius" type="number" step="0.01"><div class="desc">机器人半径（米）。决定是否“擦墙/碰墙”。</div></div>
      <div class="row"><label>safety_dist</label><input id="safety_dist" type="number" step="0.01"><div class="desc">额外安全距离（米）。碰撞半径 = robot_radius + safety_dist（地图圈显示）。</div></div>
      <div class="row"><label>inflation_radius</label><input id="inflation_radius" type="number" step="0.01"><div class="desc">规划额外膨胀（米）。规划膨胀半径 = 碰撞半径 + inflation_radius（橙色叠加）。</div></div>

      <h3>Global (RRT*)</h3>
      <div class="row"><label>rrt.expand_dis</label><input id="rrt.expand_dis" type="number" step="0.05"><div class="desc">RRT* 每次扩展最大步长（米）。过大易穿窄道失败。</div></div>
      <div class="row"><label>rrt.path_resolution</label><input id="rrt.path_resolution" type="number" step="0.01"><div class="desc">扩展时插值分辨率（米）。越小越安全但更慢。</div></div>
      <div class="row"><label>rrt.goal_sample_rate</label><input id="rrt.goal_sample_rate" type="number" step="1"><div class="desc">目标采样概率（0-100）。越大越“直奔目标”。</div></div>
      <div class="row"><label>rrt.max_iter</label><input id="rrt.max_iter" type="number" step="100"><div class="desc">最大迭代次数。越大越可能找到路但更慢。</div></div>
      <div class="row"><label>rrt.connect_circle_dist</label><input id="rrt.connect_circle_dist" type="number" step="0.1"><div class="desc">重连邻域尺度（米）。影响优化强度与耗时。</div></div>
      <div class="row"><label>rrt.smooth_iter</label><input id="rrt.smooth_iter" type="number" step="50"><div class="desc">路径平滑次数（随机捷径）。越大越平滑但更慢。</div></div>
      <div class="row"><label>obstacle_stride</label><input id="obstacle_stride" type="number" step="1"><div class="desc">障碍点抽样步长（格）。越大越快但越不精细。</div></div>

      <h3>Local (DWA)</h3>
      <div class="row"><label>max_speed</label><input id="max_speed" type="number" step="0.05"><div class="desc">最大线速度 v（m/s）。</div></div>
      <div class="row"><label>max_yawrate</label><input id="max_yawrate" type="number" step="0.05"><div class="desc">最大角速度 w（rad/s）。</div></div>
      <div class="row"><label>lookahead_dist</label><input id="lookahead_dist" type="number" step="0.1"><div class="desc">沿全局路径的前瞻距离（米）。过大易切弯贴墙。</div></div>
      <div class="row"><label>dwa.to_goal_cost_gain</label><input id="dwa.to_goal_cost_gain" type="number" step="0.05"><div class="desc">趋向子目标权重（越大越想直奔子目标）。</div></div>
      <div class="row"><label>dwa.path_cost_gain</label><input id="dwa.path_cost_gain" type="number" step="0.05"><div class="desc">贴合全局路径权重（越大越沿路径走）。</div></div>
      <div class="row"><label>dwa.heading_cost_gain</label><input id="dwa.heading_cost_gain" type="number" step="0.05"><div class="desc">朝向对齐权重（越大越先对准方向，可能原地转）。</div></div>
      <div class="row"><label>dwa.obstacle_cost_gain</label><input id="dwa.obstacle_cost_gain" type="number" step="0.01"><div class="desc">避障权重（越大越保守，障碍附近会更慢）。</div></div>
      <div class="row"><label>dwa.speed_cost_gain</label><input id="dwa.speed_cost_gain" type="number" step="0.05"><div class="desc">偏好速度权重（越大越不愿意慢走）。</div></div>
      <div class="row"><label>dwa.predict_time</label><input id="dwa.predict_time" type="number" step="0.1"><div class="desc">轨迹预测时间（秒）。越大越稳但更慢。</div></div>
      <div class="row"><label>dwa.v_samples</label><input id="dwa.v_samples" type="number" step="1"><div class="desc">线速度采样数。越大越精细但更慢。</div></div>
      <div class="row"><label>dwa.w_samples</label><input id="dwa.w_samples" type="number" step="1"><div class="desc">角速度采样数。越大越精细但更慢。</div></div>

      <h3>Sim</h3>
      <div class="row"><label>dt</label><input id="dt" type="number" step="0.01"><div class="desc">仿真步长（秒）。</div></div>
      <div class="row"><label>max_steps</label><input id="max_steps" type="number" step="50"><div class="desc">最大仿真步数（到不了就报 Max steps reached）。</div></div>
      <div class="row"><label>arrive_dist</label><input id="arrive_dist" type="number" step="0.05"><div class="desc">到达判定距离（米）。</div></div>
      <div class="row"><label>scan_count</label><input id="scan_count" type="number" step="10"><div class="desc">激光束数量（越大越精细但更慢）。</div></div>
      <div class="row"><label>scan_range_max</label><input id="scan_range_max" type="number" step="0.5"><div class="desc">激光最大量程（米）。</div></div>
      <div class="row"><label>laser_threshold</label><input id="laser_threshold" type="number" step="0.1"><div class="desc">小于该距离的点才作为障碍点（米）。</div></div>
      <div class="row"><label>local_path_max_points</label><input id="local_path_max_points" type="number" step="1"><div class="desc">局部参考的全局路径点上限（越大更平滑但更慢）。</div></div>
    </div>
  </div>

<script>
let mapMeta = null;
let occBits = null;
let unkBits = null;
let inflMeta = null;
let inflBits = null;
let canvas = document.getElementById('mapCanvas');
let ctx = canvas.getContext('2d');
let view = {scale: 1.0, offsetX: 0.0, offsetY: 0.0};
let lastResult = null;
let pendingSetTimer = null;
let poseCheck = null;

function byId(id){ return document.getElementById(id); }

function decodeB64ToUint8(b64){
  const bin = atob(b64);
  const arr = new Uint8Array(bin.length);
  for(let i=0;i<bin.length;i++) arr[i]=bin.charCodeAt(i);
  return arr;
}

function getOcc(ix, iy){
  // ix in [0,w), iy in [0,h), row-major
  const w = mapMeta.width, h = mapMeta.height;
  const idx = iy * w + ix;
  const byte = occBits[idx >> 3];
  const bit = (byte >> (idx & 7)) & 1;
  return bit;
}

function getUnk(ix, iy){
  if(!unkBits) return 0;
  const w = mapMeta.width, h = mapMeta.height;
  const idx = iy * w + ix;
  const byte = unkBits[idx >> 3];
  const bit = (byte >> (idx & 7)) & 1;
  return bit;
}

function getInflOcc(ix, iy){
  if(!inflMeta || !inflBits) return 0;
  const w = inflMeta.width, h = inflMeta.height;
  if(ix < 0 || iy < 0 || ix >= w || iy >= h) return 0;
  const idx = iy * w + ix;
  const byte = inflBits[idx >> 3];
  const bit = (byte >> (idx & 7)) & 1;
  return bit;
}

function worldToCanvas(wx, wy){
  const res = mapMeta.resolution;
  const ox = mapMeta.origin_x;
  const oy = mapMeta.origin_y;
  const ix = (wx - ox) / res;
  const iy = (wy - oy) / res;
  const x = view.offsetX + ix * view.scale;
  const y = view.offsetY + (mapMeta.height - iy) * view.scale;
  return [x, y];
}

function canvasToWorld(cx, cy){
  const res = mapMeta.resolution;
  const ox = mapMeta.origin_x;
  const oy = mapMeta.origin_y;
  const ix = (cx - view.offsetX) / view.scale;
  const iy = mapMeta.height - ((cy - view.offsetY) / view.scale);
  const wx = ox + ix * res;
  const wy = oy + iy * res;
  return [wx, wy];
}

function fitView(){
  const w = mapMeta.width, h = mapMeta.height;
  const scale = Math.min(canvas.width / w, canvas.height / h);
  view.scale = scale;
  view.offsetX = (canvas.width - w * scale) * 0.5;
  view.offsetY = (canvas.height - h * scale) * 0.5;
  draw();
}

function draw(){
  if(!mapMeta || !occBits) return;
  ctx.clearRect(0,0,canvas.width,canvas.height);

  const w = mapMeta.width, h = mapMeta.height;
  const s = view.scale;
  const unkColor = '#bdbdbd';
  const occColor = '#111';
  for(let iy=0; iy<h; iy++){
    for(let ix=0; ix<w; ix++){
      const u = getUnk(ix,iy);
      const o = getOcc(ix,iy);
      if(!u && !o) continue;
      const x = view.offsetX + ix * s;
      // Grid (iy=0) is at map origin (bottom). Canvas y grows downward.
      const y = view.offsetY + (h - iy - 1) * s;
      ctx.fillStyle = u ? unkColor : occColor;
      ctx.fillRect(x, y, s, s);
    }
  }

  const showInfl = byId('showInflated') ? byId('showInflated').checked : false;
  if(showInfl && inflBits && inflMeta && inflMeta.width === w && inflMeta.height === h){
    ctx.fillStyle = 'rgba(245, 158, 11, 0.50)'; // orange
    for(let iy=0; iy<h; iy++){
      for(let ix=0; ix<w; ix++){
        if(getInflOcc(ix,iy) && !(getOcc(ix,iy) || getUnk(ix,iy))){
          const x = view.offsetX + ix * s;
          const y = view.offsetY + (h - iy - 1) * s;
          ctx.fillRect(x, y, s, s);
        }
      }
    }
  }

  // draw result
  if(lastResult && lastResult.path && lastResult.path.length){
    ctx.strokeStyle = '#2563eb';
    ctx.lineWidth = 2;
    ctx.beginPath();
    lastResult.path.forEach((p, i) => {
      const [x,y] = worldToCanvas(p[0], p[1]);
      if(i===0) ctx.moveTo(x,y); else ctx.lineTo(x,y);
    });
    ctx.stroke();
  }
  if(lastResult && lastResult.traj && lastResult.traj.length){
    ctx.strokeStyle = '#dc2626';
    ctx.lineWidth = 2;
    ctx.beginPath();
    lastResult.traj.forEach((p, i) => {
      const [x,y] = worldToCanvas(p[0], p[1]);
      if(i===0) ctx.moveTo(x,y); else ctx.lineTo(x,y);
    });
    ctx.stroke();
  }

  // start/goal markers
  const sx = parseFloat(byId('start_x').value||'0');
  const sy = parseFloat(byId('start_y').value||'0');
  const syaw = parseFloat(byId('start_yaw').value||'0');
  const gx = parseFloat(byId('goal_x').value||'0');
  const gy = parseFloat(byId('goal_y').value||'0');
  drawDot(sx, sy, '#16a34a');
  drawDot(gx, gy, '#ef4444');

  const showFoot = byId('showFootprint') ? byId('showFootprint').checked : false;
  if(showFoot && mapMeta){
    const rr = parseFloat(byId('robot_radius').value||'0');
    const sd = parseFloat(byId('safety_dist').value||'0');
    const ir = parseFloat(byId('inflation_radius').value||'0');
    const cr = rr + sd;
    const pr = cr + ir;
    drawCircle(sx, sy, cr, 'rgba(22, 163, 74, 0.9)', false);
    drawCircle(gx, gy, cr, 'rgba(239, 68, 68, 0.9)', false);
    drawCircle(sx, sy, pr, 'rgba(245, 158, 11, 0.9)', true);
    drawCircle(gx, gy, pr, 'rgba(245, 158, 11, 0.9)', true);
    drawArrow(sx, sy, syaw, Math.max(0.25, cr), 'rgba(22, 163, 74, 0.9)');
  }

  const showSamples = byId('showSamples') ? byId('showSamples').checked : false;
  if(showSamples && poseCheck){
    drawSamples(poseCheck.start, 'rgba(22, 163, 74, 1.0)');
    drawSamples(poseCheck.goal, 'rgba(239, 68, 68, 1.0)');
  }
}

function drawDot(wx, wy, color){
  const [x,y] = worldToCanvas(wx, wy);
  ctx.fillStyle = color;
  ctx.beginPath();
  ctx.arc(x,y,5,0,Math.PI*2);
  ctx.fill();
}

function drawCircle(wx, wy, r, strokeStyle, dashed){
  if(!mapMeta) return;
  const [x,y] = worldToCanvas(wx, wy);
  const px = (r / mapMeta.resolution) * view.scale;
  ctx.save();
  ctx.strokeStyle = strokeStyle;
  ctx.lineWidth = 2;
  if(dashed) ctx.setLineDash([6,4]);
  ctx.beginPath();
  ctx.arc(x, y, px, 0, Math.PI*2);
  ctx.stroke();
  ctx.restore();
}

function drawArrow(wx, wy, yaw, len, strokeStyle){
  const [x0,y0] = worldToCanvas(wx, wy);
  const tx = wx + len * Math.cos(yaw);
  const ty = wy + len * Math.sin(yaw);
  const [x1,y1] = worldToCanvas(tx, ty);
  ctx.save();
  ctx.strokeStyle = strokeStyle;
  ctx.lineWidth = 2;
  ctx.beginPath();
  ctx.moveTo(x0,y0);
  ctx.lineTo(x1,y1);
  ctx.stroke();

  const ang = Math.atan2(y1 - y0, x1 - x0);
  const head = 10;
  ctx.beginPath();
  ctx.moveTo(x1,y1);
  ctx.lineTo(x1 - head * Math.cos(ang - Math.PI/6), y1 - head * Math.sin(ang - Math.PI/6));
  ctx.lineTo(x1 - head * Math.cos(ang + Math.PI/6), y1 - head * Math.sin(ang + Math.PI/6));
  ctx.closePath();
  ctx.fillStyle = strokeStyle;
  ctx.fill();
  ctx.restore();
}

function drawSamples(obj, strokeStyle){
  if(!obj || !obj.samples) return;
  obj.samples.forEach(p => {
    const c = p.occ ? 'rgba(220, 38, 38, 0.95)' : 'rgba(34, 197, 94, 0.95)';
    const [x,y] = worldToCanvas(p.x, p.y);
    ctx.fillStyle = c;
    ctx.beginPath();
    ctx.arc(x,y,3,0,Math.PI*2);
    ctx.fill();
  });
  if(obj.center_occ){
    const [x,y] = worldToCanvas(obj.x, obj.y);
    ctx.strokeStyle = 'rgba(220, 38, 38, 0.95)';
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(x-6,y-6); ctx.lineTo(x+6,y+6);
    ctx.moveTo(x+6,y-6); ctx.lineTo(x-6,y+6);
    ctx.stroke();
  }
  if(obj.hit){
    const [x,y] = worldToCanvas(obj.x, obj.y);
    ctx.fillStyle = 'rgba(220, 38, 38, 0.95)';
    ctx.font = '12px ui-monospace, SFMono-Regular, Menlo, Consolas, monospace';
    ctx.fillText('collision', x + 8, y - 8);
  }
}

async function apiGet(path){
  const r = await fetch(path);
  return await r.json();
}
async function apiPost(path, obj){
  const r = await fetch(path, {method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify(obj)});
  return await r.json();
}

async function pushParamsNow(){
  await apiPost('/api/set', gatherParams());
  await loadOverlay();
  await loadPoseCheck();
}

function scheduleSet(){
  if(pendingSetTimer) clearTimeout(pendingSetTimer);
  pendingSetTimer = setTimeout(()=>{ pushParamsNow().catch(()=>{}); }, 250);
}

function fmtNum(v){
  if(typeof v !== 'number' || !isFinite(v)) return '0';
  const s = v.toFixed(6);
  return s.replace(/\.?0+$/,'');
}

function buildRosCmds(){
  const p = gatherParams();
  const real = byId('rosReal') ? byId('rosReal').checked : true;
  const includeSafety = byId('rosIncludeSafetyInGlobal') ? byId('rosIncludeSafetyInGlobal').checked : true;
  const rr = Number(p['robot_radius'] ?? 0.2);
  const sd = Number(p['safety_dist'] ?? 0.0);
  const global_rr = includeSafety ? (rr + sd) : rr;

  const g = [];
  g.push('ros2 run pubsub_package global_pub --ros-args \\\\');
  g.push(`  -p robot_radius:=${fmtNum(global_rr)} \\\\`);
  g.push(`  -p inflation_radius:=${fmtNum(Number(p['inflation_radius'] ?? 0.1))} \\\\`);
  g.push(`  -p rrt.expand_dis:=${fmtNum(Number(p['rrt.expand_dis'] ?? 0.5))} \\\\`);
  g.push(`  -p rrt.path_resolution:=${fmtNum(Number(p['rrt.path_resolution'] ?? 0.05))} \\\\`);
  g.push(`  -p rrt.goal_sample_rate:=${Math.round(Number(p['rrt.goal_sample_rate'] ?? 10))} \\\\`);
  g.push(`  -p rrt.max_iter:=${Math.round(Number(p['rrt.max_iter'] ?? 5000))} \\\\`);
  g.push(`  -p rrt.connect_circle_dist:=${fmtNum(Number(p['rrt.connect_circle_dist'] ?? 2.0))} \\\\`);
  g.push(`  -p rrt.smooth_iter:=${Math.round(Number(p['rrt.smooth_iter'] ?? 200))}`);

  const l = [];
  l.push('ros2 run pubsub_package local_pub --ros-args \\\\');
  l.push(`  -p real:=${real ? 'true' : 'false'} \\\\`);
  l.push(`  -p robot_radius:=${fmtNum(rr)} \\\\`);
  l.push(`  -p safety_dist:=${fmtNum(sd)} \\\\`);
  l.push(`  -p max_speed:=${fmtNum(Number(p['max_speed'] ?? 0.5))} \\\\`);
  l.push(`  -p max_yawrate:=${fmtNum(Number(p['max_yawrate'] ?? 0.5))} \\\\`);
  l.push(`  -p lookahead_dist:=${fmtNum(Number(p['lookahead_dist'] ?? 0.8))} \\\\`);
  l.push(`  -p local_path_max_points:=${Math.round(Number(p['local_path_max_points'] ?? 30))} \\\\`);
  l.push(`  -p arrive_dist:=${fmtNum(Number(p['arrive_dist'] ?? 0.2))} \\\\`);
  l.push(`  -p laser_threshold:=${fmtNum(Number(p['laser_threshold'] ?? 1.5))} \\\\`);
  l.push(`  -p dwa.to_goal_cost_gain:=${fmtNum(Number(p['dwa.to_goal_cost_gain'] ?? 0.5))} \\\\`);
  l.push(`  -p dwa.path_cost_gain:=${fmtNum(Number(p['dwa.path_cost_gain'] ?? 2.0))} \\\\`);
  l.push(`  -p dwa.heading_cost_gain:=${fmtNum(Number(p['dwa.heading_cost_gain'] ?? 0.3))} \\\\`);
  l.push(`  -p dwa.obstacle_cost_gain:=${fmtNum(Number(p['dwa.obstacle_cost_gain'] ?? 1.0))} \\\\`);
  l.push(`  -p dwa.speed_cost_gain:=${fmtNum(Number(p['dwa.speed_cost_gain'] ?? 0.2))} \\\\`);
  l.push(`  -p dwa.predict_time:=${fmtNum(Number(p['dwa.predict_time'] ?? 2.0))} \\\\`);
  l.push(`  -p dwa.v_samples:=${Math.round(Number(p['dwa.v_samples'] ?? 6))} \\\\`);
  l.push(`  -p dwa.w_samples:=${Math.round(Number(p['dwa.w_samples'] ?? 12))}`);

  const out = [
    '# Global planner (RRT*)',
    g.join('\\n'),
    '',
    '# Local planner (DWA)',
    l.join('\\n'),
  ].join('\\n');

  const ta = byId('rosCmd');
  if(ta) ta.value = out;
  return out;
}

const PARAM_IDS = [
  'dt','max_steps','arrive_dist',
  'robot_radius','safety_dist','inflation_radius',
  'max_speed','max_yawrate','lookahead_dist','local_path_max_points',
  'scan_count','scan_range_max','laser_threshold',
  'obstacle_stride',
  'rrt.expand_dis','rrt.path_resolution','rrt.goal_sample_rate','rrt.max_iter','rrt.connect_circle_dist','rrt.smooth_iter',
  'dwa.to_goal_cost_gain','dwa.path_cost_gain','dwa.heading_cost_gain','dwa.obstacle_cost_gain','dwa.speed_cost_gain','dwa.predict_time','dwa.v_samples','dwa.w_samples'
];

function updateDerivedInfo(){
  const rr = parseFloat(byId('robot_radius').value||'0');
  const sd = parseFloat(byId('safety_dist').value||'0');
  const ir = parseFloat(byId('inflation_radius').value||'0');
  const cr = rr + sd;
  const pr = cr + ir;
  byId('derivedInfo').textContent = `碰撞半径 = robot_radius + safety_dist = ${cr.toFixed(3)} m; 规划膨胀半径 = 碰撞半径 + inflation_radius = ${pr.toFixed(3)} m`;
}

function bindHelpAndRedraw(){
  const ids = ['start_x','start_y','start_yaw','goal_x','goal_y', ...PARAM_IDS];
  ids.forEach(id => {
    const el = byId(id);
    if(!el) return;
    el.addEventListener('input', ()=>{
      updateDerivedInfo();
      scheduleSet();
      draw();
    });
  });
  const si = byId('showInflated');
  const sf = byId('showFootprint');
  const ss = byId('showSamples');
  if(si) si.addEventListener('change', draw);
  if(sf) sf.addEventListener('change', draw);
  if(ss) ss.addEventListener('change', draw);
  updateDerivedInfo();
}

function gatherParams(){
  const p = {};
  PARAM_IDS.forEach(id => {
    const el = byId(id);
    if(!el) return;
    const v = el.value;
    if(v === '') return;
    if(id === 'rrt.goal_sample_rate' || id === 'rrt.max_iter' || id === 'rrt.smooth_iter' || id === 'obstacle_stride' ||
       id === 'dwa.v_samples' || id === 'dwa.w_samples' || id === 'scan_count' || id === 'max_steps' || id === 'local_path_max_points'){
      p[id] = parseInt(v,10);
    } else {
      p[id] = parseFloat(v);
    }
  });
  p.start = [parseFloat(byId('start_x').value), parseFloat(byId('start_y').value), parseFloat(byId('start_yaw').value)];
  p.goal = [parseFloat(byId('goal_x').value), parseFloat(byId('goal_y').value)];
  return p;
}

function applyParams(params){
  byId('start_x').value = params.start[0];
  byId('start_y').value = params.start[1];
  byId('start_yaw').value = params.start[2];
  byId('goal_x').value = params.goal[0];
  byId('goal_y').value = params.goal[1];
  Object.keys(params).forEach(k=>{
    const el = byId(k);
    if(el) el.value = params[k];
  });
}

async function refreshState(){
  const st = await apiGet('/api/state');
  applyParams(st.params);
  updateDerivedInfo();
  await loadOverlay();
  await loadPoseCheck();
  byId('status').textContent = JSON.stringify(st, null, 2);
  draw();
}

async function loadMap(){
  const mp = await apiGet('/api/map?downsample=2');
  mapMeta = mp;
  const packed = decodeB64ToUint8(mp.occ_packbits_le_b64);
  const packedUnk = mp.unk_packbits_le_b64 ? decodeB64ToUint8(mp.unk_packbits_le_b64) : null;
  // unpackbits in JS: we keep packed bits; getOcc() reads bits directly.
  occBits = packed;
  unkBits = packedUnk;
  byId('mapInfo').textContent = `${mp.width}x${mp.height} res=${mp.resolution.toFixed(3)}`;
  fitView();
}

async function loadOverlay(){
  const mp = await apiGet('/api/map_overlay?downsample=2');
  inflMeta = mp;
  inflBits = decodeB64ToUint8(mp.occ_packbits_le_b64);
}

async function loadPoseCheck(){
  const pc = await apiGet('/api/pose_check');
  poseCheck = pc && pc.ok ? pc : null;
}

canvas.addEventListener('contextmenu', (e)=> e.preventDefault());
canvas.addEventListener('mousedown', async (e)=>{
  if(!mapMeta) return;
  const rect = canvas.getBoundingClientRect();
  const cx = e.clientX - rect.left;
  const cy = e.clientY - rect.top;
  const [wx, wy] = canvasToWorld(cx, cy);
  if(e.button === 2){
    byId('goal_x').value = wx.toFixed(3);
    byId('goal_y').value = wy.toFixed(3);
  } else {
    byId('start_x').value = wx.toFixed(3);
    byId('start_y').value = wy.toFixed(3);
  }
  await apiPost('/api/set', gatherParams());
  await loadOverlay();
  draw();
});

byId('fitBtn').addEventListener('click', fitView);
byId('resetBtn').addEventListener('click', async ()=>{
  await apiPost('/api/reset', {});
  await refreshState();
});
byId('genRosBtn').addEventListener('click', ()=>{ buildRosCmds(); });
byId('copyRosBtn').addEventListener('click', async ()=>{
  const text = buildRosCmds();
  try{
    await navigator.clipboard.writeText(text);
  }catch(e){
    const ta = byId('rosCmd');
    if(ta){ ta.focus(); ta.select(); document.execCommand('copy'); }
  }
});
byId('runBtn').addEventListener('click', async ()=>{
  await apiPost('/api/set', gatherParams());
  await loadOverlay();
  await loadPoseCheck();
  const r = await apiPost('/api/run', {});
  const jobId = r.job_id;
  byId('status').textContent = `Running job ${jobId} ...`;
  while(true){
    const jr = await apiGet(`/api/job?id=${jobId}`);
    if(jr.ok && jr.job && (jr.job.status === 'done' || jr.job.status === 'error')){
      if(jr.job.status === 'done'){
        lastResult = jr.result;
      }
      byId('status').textContent = JSON.stringify(jr, null, 2);
      draw();
      break;
    }
    await new Promise(res=>setTimeout(res, 300));
  }
});

loadMap().then(refreshState);
bindHelpAndRedraw();
buildRosCmds();
</script>
</body>
</html>
"""


def _json(handler: BaseHTTPRequestHandler, code: int, payload: Dict[str, Any]) -> None:
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    handler.send_response(code)
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Cache-Control", "no-store")
    handler.end_headers()
    handler.wfile.write(body)


def serve(spec: MapSpec, host: str, port: int, open_browser: bool) -> None:
    app = SimPanelApp(spec)

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt: str, *args: Any) -> None:
            return

        def do_GET(self) -> None:  # noqa: N802
            if self.path in ("/", "/index.html"):
                body = INDEX_HTML.encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(body)
                return

            if self.path.startswith("/api/state"):
                _json(self, 200, app.state())
                return

            if self.path.startswith("/api/map_overlay"):
                qs = self.path.split("?", 1)[1] if "?" in self.path else ""
                params = dict([p.split("=", 1) for p in qs.split("&") if p and "=" in p])
                ds = int(params.get("downsample", "2"))
                _json(self, 200, app.map_overlay_payload(downsample=ds))
                return

            if self.path.startswith("/api/pose_check"):
                _json(self, 200, app.pose_check_payload())
                return

            if self.path.startswith("/api/map"):
                qs = self.path.split("?", 1)[1] if "?" in self.path else ""
                params = dict([p.split("=", 1) for p in qs.split("&") if p and "=" in p])
                ds = int(params.get("downsample", "2"))
                _json(self, 200, app.map_payload(downsample=ds))
                return

            if self.path.startswith("/api/job"):
                qs = self.path.split("?", 1)[1] if "?" in self.path else ""
                params = dict([p.split("=", 1) for p in qs.split("&") if p and "=" in p])
                job_id = params.get("id", "")
                _json(self, 200, app.job_result(job_id))
                return

            _json(self, 404, {"ok": False, "error": "not found"})

        def do_POST(self) -> None:  # noqa: N802
            length = int(self.headers.get("Content-Length", "0") or "0")
            raw = self.rfile.read(length) if length > 0 else b"{}"
            try:
                payload = json.loads(raw.decode("utf-8")) if raw else {}
            except Exception:
                payload = {}

            if self.path.startswith("/api/set"):
                if isinstance(payload, dict):
                    app.set_params(payload)
                _json(self, 200, {"ok": True})
                return

            if self.path.startswith("/api/reset"):
                app.params = _default_params()
                _json(self, 200, {"ok": True})
                return

            if self.path.startswith("/api/run"):
                job_id = app.start_run()
                _json(self, 200, {"ok": True, "job_id": job_id})
                return

            _json(self, 404, {"ok": False, "error": "not found"})

    httpd = ThreadingHTTPServer((host, int(port)), Handler)
    url = f"http://{host}:{port}/"
    print(f"No-ROS sim panel running: {url}")
    if open_browser:
        try:
            webbrowser.open(url, new=1, autoraise=True)
        except Exception:
            pass
    httpd.serve_forever()


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--map", type=str, default="", help="Path to ROS map.yaml (optional)")
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8892)
    parser.add_argument("--open-browser", action="store_true")
    args = parser.parse_args(argv)

    if args.map:
        spec = load_map_yaml(Path(args.map))
    else:
        spec = procedural_map()

    serve(spec, host=args.host, port=args.port, open_browser=args.open_browser)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
