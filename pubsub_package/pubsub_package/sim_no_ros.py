#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
No-ROS 2D simulator for running the planners on native Python.

Goals:
- Runs on Windows without ROS2 installed.
- Uses the existing planners: RRT* (global) + DWA (local).
- Loads standard ROS map.yaml + pgm if provided, otherwise uses a built-in map.
"""

from __future__ import annotations

import argparse
import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np
import yaml

from pubsub_package.planner.dwa import DWA
from pubsub_package.planner.rrt_star import RRT_star


def _normalize_angle(angle: float) -> float:
    while angle > math.pi:
        angle -= 2.0 * math.pi
    while angle < -math.pi:
        angle += 2.0 * math.pi
    return angle


def _read_pgm(path: Path) -> np.ndarray:
    data = path.read_bytes()
    idx = 0

    def _read_token() -> bytes:
        nonlocal idx
        while idx < len(data) and data[idx] in b" \t\r\n":
            idx += 1
        if idx < len(data) and data[idx] == ord("#"):
            while idx < len(data) and data[idx] not in b"\r\n":
                idx += 1
            return _read_token()
        start = idx
        while idx < len(data) and data[idx] not in b" \t\r\n":
            idx += 1
        return data[start:idx]

    magic = _read_token()
    if magic not in (b"P2", b"P5"):
        raise ValueError(f"Unsupported PGM format: {magic!r}")

    width = int(_read_token())
    height = int(_read_token())
    maxval = int(_read_token())
    if maxval <= 0 or maxval > 65535:
        raise ValueError(f"Invalid maxval: {maxval}")

    while idx < len(data) and data[idx] in b" \t\r\n":
        idx += 1

    if magic == b"P5":
        if maxval < 256:
            raw = np.frombuffer(data, dtype=np.uint8, offset=idx, count=width * height)
            return raw.reshape((height, width))
        raw = np.frombuffer(data, dtype=">u2", offset=idx, count=width * height)
        img16 = raw.reshape((height, width)).astype(np.float32)
        img = np.clip(img16 * (255.0 / float(maxval)), 0.0, 255.0).astype(np.uint8)
        return img

    tokens = data[idx:].split()
    if len(tokens) < width * height:
        raise ValueError("PGM P2 has insufficient pixel data")
    vals = np.array(tokens[: width * height], dtype=np.int32)
    vals = np.clip(vals, 0, maxval).astype(np.float32)
    img = np.clip(vals * (255.0 / float(maxval)), 0.0, 255.0).astype(np.uint8)
    return img.reshape((height, width))


@dataclass(frozen=True)
class MapSpec:
    resolution: float
    origin_x: float
    origin_y: float
    origin_yaw: float
    grid: np.ndarray  # shape (H, W), values in {0..100, -1}

    @property
    def width(self) -> int:
        return int(self.grid.shape[1])

    @property
    def height(self) -> int:
        return int(self.grid.shape[0])

    @property
    def minx(self) -> float:
        return self.origin_x

    @property
    def miny(self) -> float:
        return self.origin_y

    @property
    def maxx(self) -> float:
        return self.origin_x + self.width * self.resolution

    @property
    def maxy(self) -> float:
        return self.origin_y + self.height * self.resolution


def load_map_yaml(map_yaml: Path) -> MapSpec:
    config = yaml.safe_load(map_yaml.read_text(encoding="utf-8"))
    image_path = Path(config["image"])
    if not image_path.is_absolute():
        image_path = (map_yaml.parent / image_path).resolve()

    resolution = float(config["resolution"])
    origin = config["origin"]
    origin_x, origin_y, origin_yaw = float(origin[0]), float(origin[1]), float(origin[2])

    occupied_thresh = float(config.get("occupied_thresh", 0.65))
    free_thresh = float(config.get("free_thresh", 0.196))
    negate = int(config.get("negate", 0))

    img = _read_pgm(image_path).astype(np.float32) / 255.0  # 0..1
    occ = img if negate else (1.0 - img)

    grid = np.full(img.shape, -1, dtype=np.int16)
    grid[occ > occupied_thresh] = 100
    grid[occ < free_thresh] = 0

    # Map origin is at lower-left; PGM (0,0) is upper-left.
    grid = np.flipud(grid)

    return MapSpec(
        resolution=resolution,
        origin_x=origin_x,
        origin_y=origin_y,
        origin_yaw=origin_yaw,
        grid=grid,
    )


def procedural_map() -> MapSpec:
    w, h = 80, 60
    resolution = 0.1
    origin_x, origin_y, origin_yaw = -4.0, -3.0, 0.0
    grid = np.zeros((h, w), dtype=np.int16)

    grid[0, :] = 100
    grid[-1, :] = 100
    grid[:, 0] = 100
    grid[:, -1] = 100

    grid[20:40, 20:22] = 100
    grid[10:12, 30:60] = 100
    grid[45:55, 55:57] = 100

    return MapSpec(resolution=resolution, origin_x=origin_x, origin_y=origin_y, origin_yaw=origin_yaw, grid=grid)


def _world_to_grid(spec: MapSpec, x: float, y: float) -> Optional[Tuple[int, int]]:
    ix = int(math.floor((x - spec.origin_x) / spec.resolution))
    iy = int(math.floor((y - spec.origin_y) / spec.resolution))
    if ix < 0 or iy < 0 or iy >= spec.height or ix >= spec.width:
        return None
    return ix, iy


def is_occupied(spec: MapSpec, x: float, y: float, treat_unknown_as_obstacle: bool = True) -> bool:
    idx = _world_to_grid(spec, x, y)
    if idx is None:
        return True
    ix, iy = idx
    v = int(spec.grid[iy, ix])
    if v < 0:
        return bool(treat_unknown_as_obstacle)
    return v >= 50


def footprint_collision(spec: MapSpec, x: float, y: float, robot_radius: float) -> bool:
    angles = np.linspace(0.0, 2.0 * math.pi, 16, endpoint=False)
    for a in angles:
        px = x + robot_radius * math.cos(a)
        py = y + robot_radius * math.sin(a)
        if is_occupied(spec, px, py):
            return True
    return is_occupied(spec, x, y)


def iter_obstacles(spec: MapSpec, stride: int = 1) -> Iterable[Tuple[float, float]]:
    stride = max(int(stride), 1)
    occ = np.argwhere(spec.grid >= 50)
    if stride > 1:
        occ = occ[::stride]
    for iy, ix in occ:
        x = spec.origin_x + (float(ix) + 0.5) * spec.resolution
        y = spec.origin_y + (float(iy) + 0.5) * spec.resolution
        yield (x, y)


def cast_ray(
    spec: MapSpec,
    x: float,
    y: float,
    angle_world: float,
    range_min: float,
    range_max: float,
    step: float,
) -> float:
    c = math.cos(angle_world)
    s = math.sin(angle_world)
    dist = range_min
    while dist <= range_max:
        px = x + c * dist
        py = y + s * dist
        if is_occupied(spec, px, py):
            return dist
        dist += step
    return float("inf")


def laser_points_robot_frame(
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
) -> List[Tuple[float, float]]:
    scan_count = max(int(scan_count), 1)
    angle_inc = (angle_max - angle_min) / float(scan_count)
    step = max(spec.resolution * 0.5, 0.02)

    points: List[Tuple[float, float]] = []
    for i in range(scan_count):
        a = angle_min + float(i) * angle_inc
        d = cast_ray(spec, x, y, yaw + a, range_min, range_max, step)
        if not math.isfinite(d):
            continue
        if d >= obstacle_threshold:
            continue
        # point in robot frame (robot at 0,0,0)
        points.append((math.cos(a) * d, math.sin(a) * d))
    return points


def _closest_point_on_path(path: Sequence[Sequence[float]], x: float, y: float) -> int:
    pts = np.asarray(path, dtype=float)
    dx = pts[:, 0] - x
    dy = pts[:, 1] - y
    return int(np.argmin(dx * dx + dy * dy))


def _lookahead_goal(path: Sequence[Sequence[float]], idx: int, lookahead_points: int) -> Tuple[float, float]:
    j = min(max(idx + lookahead_points, 0), len(path) - 1)
    return float(path[j][0]), float(path[j][1])


def _world_to_robot(x: float, y: float, yaw: float, gx: float, gy: float) -> Tuple[float, float]:
    dx = gx - x
    dy = gy - y
    c = math.cos(-yaw)
    s = math.sin(-yaw)
    return (c * dx - s * dy, s * dx + c * dy)


def run_sim(
    spec: MapSpec,
    *,
    start: Tuple[float, float, float],
    goal: Tuple[float, float],
    robot_radius: float = 0.2,
    safe_dist: float = 0.05,
    max_speed: float = 0.5,
    max_yawrate: float = 1.0,
    dt: float = 0.1,
    max_steps: int = 2000,
    arrive_dist: float = 0.2,
    lookahead_points: int = 8,
    scan_count: int = 360,
    obstacle_threshold: float = 1.5,
    obstacle_stride: int = 2,
) -> Tuple[bool, List[List[float]], List[List[float]]]:
    sx, sy, syaw = start
    gx, gy = goal

    obstacles = list(iter_obstacles(spec, stride=obstacle_stride))
    obstacles.append((-9999.0, -9999.0))

    rrt = RRT_star(
        minx=spec.minx,
        maxx=spec.maxx,
        miny=spec.miny,
        maxy=spec.maxy,
        obstacles=obstacles,
        robot_size=robot_radius,
        safe_dist=safe_dist,
        expand_dis=max(spec.resolution * 6.0, 0.5),
        path_resolution=max(spec.resolution * 0.5, 0.05),
        goal_sample_rate=10,
        max_iter=4000,
    )

    found, path = rrt.plan(sx, sy, gx, gy)
    if not found or not path:
        return False, [], []

    dwa = DWA()
    dwa.config(max_speed=max_speed, max_yawrate=max_yawrate, base=robot_radius)
    dwa.dt = float(dt)

    x, y, yaw = float(sx), float(sy), float(syaw)
    v, w = 0.0, 0.0

    traj: List[List[float]] = [[x, y, yaw]]
    for _ in range(int(max_steps)):
        if math.hypot(gx - x, gy - y) <= arrive_dist:
            return True, path, traj

        if footprint_collision(spec, x, y, robot_radius):
            return False, path, traj

        near_idx = _closest_point_on_path(path, x, y)
        gpx, gpy = _lookahead_goal(path, near_idx, lookahead_points)
        goal_rx, goal_ry = _world_to_robot(x, y, yaw, gpx, gpy)

        end_idx = min(near_idx + 40, len(path))
        local_path = [
            _world_to_robot(x, y, yaw, float(path[i][0]), float(path[i][1]))
            for i in range(near_idx, end_idx)
        ]

        points = laser_points_robot_frame(
            spec,
            x,
            y,
            yaw,
            scan_count=scan_count,
            angle_min=-math.pi,
            angle_max=math.pi,
            range_min=0.05,
            range_max=8.0,
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
        if not footprint_collision(spec, nx, ny, robot_radius):
            x, y = nx, ny

        traj.append([x, y, yaw])

    return False, path, traj


def _try_plot(spec: MapSpec, path: Sequence[Sequence[float]], traj: Sequence[Sequence[float]], out_png: Optional[Path]):
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except Exception:
        return False

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.set_aspect("equal", "box")
    ax.set_title("No-ROS 2D sim (RRT* + DWA)")

    img = (spec.grid >= 50).astype(np.uint8)
    ax.imshow(
        np.flipud(img),
        cmap="gray_r",
        extent=(spec.minx, spec.maxx, spec.miny, spec.maxy),
        origin="lower",
        alpha=0.6,
    )

    if path:
        p = np.asarray(path, dtype=float)
        ax.plot(p[:, 0], p[:, 1], "b-", linewidth=2, label="global path")

    if traj:
        t = np.asarray(traj, dtype=float)
        ax.plot(t[:, 0], t[:, 1], "r-", linewidth=1.5, label="trajectory")
        ax.plot([t[0, 0]], [t[0, 1]], "go", label="start")
        ax.plot([t[-1, 0]], [t[-1, 1]], "ro", label="end")

    ax.legend(loc="best")
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.4)

    if out_png is not None:
        fig.savefig(out_png, dpi=150, bbox_inches="tight")
    else:
        plt.show()
    return True


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--map", type=str, default="", help="Path to ROS map.yaml (optional)")
    parser.add_argument("--start", type=float, nargs=3, default=(0.0, 0.0, 0.0), metavar=("X", "Y", "YAW"))
    parser.add_argument("--goal", type=float, nargs=2, default=(2.0, 0.0), metavar=("X", "Y"))
    parser.add_argument("--robot-radius", type=float, default=0.2)
    parser.add_argument("--safe-dist", type=float, default=0.05)
    parser.add_argument("--max-speed", type=float, default=0.5)
    parser.add_argument("--max-yawrate", type=float, default=1.0)
    parser.add_argument("--dt", type=float, default=0.1)
    parser.add_argument("--max-steps", type=int, default=2500)
    parser.add_argument("--arrive", type=float, default=0.2)
    parser.add_argument("--lookahead-points", type=int, default=8)
    parser.add_argument("--scan-count", type=int, default=360)
    parser.add_argument("--obstacle-threshold", type=float, default=1.5)
    parser.add_argument("--obstacle-stride", type=int, default=2)
    parser.add_argument("--out-csv", type=str, default="", help="Write trajectory to CSV")
    parser.add_argument("--out-png", type=str, default="", help="Write plot to PNG (requires matplotlib)")
    parser.add_argument("--plot", action="store_true", help="Show interactive plot (requires matplotlib)")
    args = parser.parse_args(argv)

    if args.map:
        spec = load_map_yaml(Path(args.map))
    else:
        spec = procedural_map()

    ok, path, traj = run_sim(
        spec,
        start=(float(args.start[0]), float(args.start[1]), float(args.start[2])),
        goal=(float(args.goal[0]), float(args.goal[1])),
        robot_radius=float(args.robot_radius),
        safe_dist=float(args.safe_dist),
        max_speed=float(args.max_speed),
        max_yawrate=float(args.max_yawrate),
        dt=float(args.dt),
        max_steps=int(args.max_steps),
        arrive_dist=float(args.arrive),
        lookahead_points=int(args.lookahead_points),
        scan_count=int(args.scan_count),
        obstacle_threshold=float(args.obstacle_threshold),
        obstacle_stride=int(args.obstacle_stride),
    )

    print(f"success={ok} path_len={len(path)} traj_len={len(traj)}")

    if args.out_csv:
        out_csv = Path(args.out_csv)
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        with out_csv.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["x", "y", "yaw"])
            for x, y, yaw in traj:
                w.writerow([x, y, yaw])

    out_png = Path(args.out_png) if args.out_png else None
    if args.plot or out_png is not None:
        plotted = _try_plot(spec, path, traj, out_png=out_png)
        if not plotted:
            print("matplotlib not available; skipping plot")

    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
