#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import yaml

import rclpy
from geometry_msgs.msg import PoseWithCovarianceStamped, TransformStamped, Twist
from nav_msgs.msg import OccupancyGrid, Odometry
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import LaserScan
from tf2_ros import TransformBroadcaster

try:
    from ament_index_python.packages import get_package_share_directory  # type: ignore
except Exception:  # pragma: no cover
    get_package_share_directory = None


def _yaw_to_quat(yaw: float) -> Tuple[float, float, float, float]:
    half = 0.5 * yaw
    return (0.0, 0.0, math.sin(half), math.cos(half))


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
            img = raw.reshape((height, width))
            return img
        raw = np.frombuffer(data, dtype=">u2", offset=idx, count=width * height)
        img16 = raw.reshape((height, width)).astype(np.float32)
        img = np.clip(img16 * (255.0 / float(maxval)), 0.0, 255.0).astype(np.uint8)
        return img

    # P2 ASCII
    tokens = data[idx:].split()
    if len(tokens) < width * height:
        raise ValueError("PGM P2 has insufficient pixel data")
    vals = np.array(tokens[: width * height], dtype=np.int32)
    vals = np.clip(vals, 0, maxval).astype(np.float32)
    img = np.clip(vals * (255.0 / float(maxval)), 0.0, 255.0).astype(np.uint8)
    return img.reshape((height, width))


@dataclass
class MapSpec:
    resolution: float
    origin_x: float
    origin_y: float
    origin_yaw: float
    grid: np.ndarray  # shape (H, W), values in {0..100, -1}


def _load_map_yaml(map_yaml: Path) -> MapSpec:
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
    if negate:
        occ = img
    else:
        occ = 1.0 - img

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


def _procedural_map() -> MapSpec:
    w, h = 60, 60
    resolution = 0.1
    origin_x, origin_y, origin_yaw = -3.0, -3.0, 0.0
    grid = np.zeros((h, w), dtype=np.int16)

    grid[0, :] = 100
    grid[-1, :] = 100
    grid[:, 0] = 100
    grid[:, -1] = 100

    grid[25:35, 10:12] = 100
    grid[10:12, 20:45] = 100
    grid[40:50, 35:37] = 100

    return MapSpec(
        resolution=resolution,
        origin_x=origin_x,
        origin_y=origin_y,
        origin_yaw=origin_yaw,
        grid=grid,
    )


class Sim2D(Node):
    def __init__(self):
        super().__init__("sim2d")

        self.declare_parameter("map_yaml", "")
        self.declare_parameter("frame_map", "map")
        self.declare_parameter("frame_base", "base_footprint")
        self.declare_parameter("publish_scan_topic", "/course_agv/laser/scan")
        self.declare_parameter("also_publish_scan_topic", "/scan")
        self.declare_parameter("cmd_vel_topic", "/course_agv/velocity")
        self.declare_parameter("also_subscribe_cmd_vel_topic", "/cmd_vel")
        self.declare_parameter("max_speed", 0.5)
        self.declare_parameter("max_yawrate", 1.0)
        self.declare_parameter("robot_radius", 0.2)
        self.declare_parameter("sim_rate_hz", 50.0)
        self.declare_parameter("scan_rate_hz", 10.0)
        self.declare_parameter("scan_angle_min", -math.pi)
        self.declare_parameter("scan_angle_max", math.pi)
        self.declare_parameter("scan_count", 360)
        self.declare_parameter("scan_range_min", 0.05)
        self.declare_parameter("scan_range_max", 8.0)
        self.declare_parameter("treat_unknown_as_obstacle", True)

        self.frame_map = str(self.get_parameter("frame_map").value)
        self.frame_base = str(self.get_parameter("frame_base").value)
        self.scan_topic = str(self.get_parameter("publish_scan_topic").value)
        self.scan_topic2 = str(self.get_parameter("also_publish_scan_topic").value)
        self.cmd_vel_topic = str(self.get_parameter("cmd_vel_topic").value)
        self.cmd_vel_topic2 = str(self.get_parameter("also_subscribe_cmd_vel_topic").value)

        self.max_speed = float(self.get_parameter("max_speed").value)
        self.max_yawrate = float(self.get_parameter("max_yawrate").value)
        self.robot_radius = float(self.get_parameter("robot_radius").value)
        self.treat_unknown_as_obstacle = bool(self.get_parameter("treat_unknown_as_obstacle").value)

        self.scan_angle_min = float(self.get_parameter("scan_angle_min").value)
        self.scan_angle_max = float(self.get_parameter("scan_angle_max").value)
        self.scan_count = int(self.get_parameter("scan_count").value)
        self.scan_range_min = float(self.get_parameter("scan_range_min").value)
        self.scan_range_max = float(self.get_parameter("scan_range_max").value)
        self.scan_angle_inc = (self.scan_angle_max - self.scan_angle_min) / float(self.scan_count)

        self._map: Optional[MapSpec] = None
        self._map_msg: Optional[OccupancyGrid] = None

        self.x = 0.0
        self.y = 0.0
        self.yaw = 0.0
        self.v_cmd = 0.0
        self.w_cmd = 0.0

        self.tf_broadcaster = TransformBroadcaster(self)

        map_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self.map_pub = self.create_publisher(OccupancyGrid, "/map", map_qos)
        self.odom_pub = self.create_publisher(Odometry, "/odom", 10)
        self.scan_pub = self.create_publisher(LaserScan, self.scan_topic, 10)
        self.scan_pub2 = self.create_publisher(LaserScan, self.scan_topic2, 10)

        self.create_subscription(Twist, self.cmd_vel_topic, self._cmd_vel_cb, 10)
        if self.cmd_vel_topic2 and self.cmd_vel_topic2 != self.cmd_vel_topic:
            self.create_subscription(Twist, self.cmd_vel_topic2, self._cmd_vel_cb, 10)

        self.create_subscription(PoseWithCovarianceStamped, "/initialpose", self._initialpose_cb, 10)

        sim_rate_hz = float(self.get_parameter("sim_rate_hz").value)
        scan_rate_hz = float(self.get_parameter("scan_rate_hz").value)
        self.sim_dt = 1.0 / max(sim_rate_hz, 1.0)

        self.create_timer(self.sim_dt, self._step)
        self.create_timer(1.0 / max(scan_rate_hz, 1.0), self._publish_scan)
        self.create_timer(1.0, self._publish_map)

        self._load_or_create_map()
        self._publish_map()

    def _initialpose_cb(self, msg: PoseWithCovarianceStamped) -> None:
        x = float(msg.pose.pose.position.x)
        y = float(msg.pose.pose.position.y)
        q = msg.pose.pose.orientation
        # yaw from quaternion (z,w for planar motion)
        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        yaw = math.atan2(siny_cosp, cosy_cosp)

        if not self._footprint_collision(x, y):
            self.x, self.y, self.yaw = x, y, _normalize_angle(yaw)
            self.get_logger().info(f"Set initial pose: x={self.x:.3f}, y={self.y:.3f}, yaw={self.yaw:.3f}")
        else:
            self.get_logger().warning("Rejected initial pose: in collision")

    def _load_or_create_map(self) -> None:
        map_yaml = str(self.get_parameter("map_yaml").value).strip()
        map_path: Optional[Path] = None

        if map_yaml:
            map_path = Path(map_yaml)
            if not map_path.is_absolute():
                map_path = map_path.resolve()
            if not map_path.exists():
                self.get_logger().warning(f"map_yaml not found: {map_path}, falling back to procedural map")
                map_path = None

        if map_path is None and get_package_share_directory is not None:
            try:
                share_dir = Path(get_package_share_directory("pubsub_package"))
                candidate = share_dir / "maps" / "simple_map.yaml"
                if candidate.exists():
                    map_path = candidate
            except Exception:
                map_path = None

        if map_path is not None:
            self.get_logger().info(f"Loading map: {map_path}")
            self._map = _load_map_yaml(map_path)
        else:
            self.get_logger().info("Using procedural map")
            self._map = _procedural_map()

        self._map_msg = self._to_occupancy_grid(self._map)

        # Place robot at a guaranteed free spot near origin.
        self.x = self._map.origin_x + 2.0 * self._map.resolution
        self.y = self._map.origin_y + 2.0 * self._map.resolution
        self.yaw = 0.0

    def _to_occupancy_grid(self, spec: MapSpec) -> OccupancyGrid:
        msg = OccupancyGrid()
        msg.header.frame_id = self.frame_map
        msg.info.resolution = float(spec.resolution)
        msg.info.width = int(spec.grid.shape[1])
        msg.info.height = int(spec.grid.shape[0])
        msg.info.origin.position.x = float(spec.origin_x)
        msg.info.origin.position.y = float(spec.origin_y)
        qx, qy, qz, qw = _yaw_to_quat(spec.origin_yaw)
        msg.info.origin.orientation.x = qx
        msg.info.origin.orientation.y = qy
        msg.info.origin.orientation.z = qz
        msg.info.origin.orientation.w = qw

        msg.data = spec.grid.astype(np.int8).ravel(order="C").tolist()
        return msg

    def _publish_map(self) -> None:
        if self._map_msg is None:
            return
        self._map_msg.header.stamp = self.get_clock().now().to_msg()
        self.map_pub.publish(self._map_msg)

    def _cmd_vel_cb(self, msg: Twist) -> None:
        self.v_cmd = float(np.clip(msg.linear.x, -self.max_speed, self.max_speed))
        self.w_cmd = float(np.clip(msg.angular.z, -self.max_yawrate, self.max_yawrate))

    def _world_to_grid(self, x: float, y: float) -> Optional[Tuple[int, int]]:
        if self._map is None:
            return None
        ix = int(math.floor((x - self._map.origin_x) / self._map.resolution))
        iy = int(math.floor((y - self._map.origin_y) / self._map.resolution))
        if ix < 0 or iy < 0 or iy >= self._map.grid.shape[0] or ix >= self._map.grid.shape[1]:
            return None
        return ix, iy

    def _is_occupied(self, x: float, y: float) -> bool:
        if self._map is None:
            return True
        idx = self._world_to_grid(x, y)
        if idx is None:
            return True
        ix, iy = idx
        v = int(self._map.grid[iy, ix])
        if v < 0:
            return bool(self.treat_unknown_as_obstacle)
        return v >= 50

    def _footprint_collision(self, x: float, y: float) -> bool:
        angles = np.linspace(0.0, 2.0 * math.pi, 16, endpoint=False)
        for a in angles:
            px = x + self.robot_radius * math.cos(a)
            py = y + self.robot_radius * math.sin(a)
            if self._is_occupied(px, py):
                return True
        return self._is_occupied(x, y)

    def _step(self) -> None:
        if self._map is None:
            return

        self.yaw = _normalize_angle(self.yaw + self.w_cmd * self.sim_dt)

        nx = self.x + self.v_cmd * math.cos(self.yaw) * self.sim_dt
        ny = self.y + self.v_cmd * math.sin(self.yaw) * self.sim_dt
        if not self._footprint_collision(nx, ny):
            self.x, self.y = nx, ny
        else:
            self.v_cmd = 0.0

        self._publish_tf_and_odom()

    def _publish_tf_and_odom(self) -> None:
        now = self.get_clock().now().to_msg()

        t = TransformStamped()
        t.header.stamp = now
        t.header.frame_id = self.frame_map
        t.child_frame_id = self.frame_base
        t.transform.translation.x = float(self.x)
        t.transform.translation.y = float(self.y)
        t.transform.translation.z = 0.0
        qx, qy, qz, qw = _yaw_to_quat(self.yaw)
        t.transform.rotation.x = qx
        t.transform.rotation.y = qy
        t.transform.rotation.z = qz
        t.transform.rotation.w = qw
        self.tf_broadcaster.sendTransform(t)

        odom = Odometry()
        odom.header.stamp = now
        odom.header.frame_id = self.frame_map
        odom.child_frame_id = self.frame_base
        odom.pose.pose.position.x = float(self.x)
        odom.pose.pose.position.y = float(self.y)
        odom.pose.pose.position.z = 0.0
        odom.pose.pose.orientation.x = qx
        odom.pose.pose.orientation.y = qy
        odom.pose.pose.orientation.z = qz
        odom.pose.pose.orientation.w = qw
        odom.twist.twist.linear.x = float(self.v_cmd)
        odom.twist.twist.angular.z = float(self.w_cmd)
        self.odom_pub.publish(odom)

    def _cast_ray(self, angle_world: float) -> float:
        step = max(self._map.resolution * 0.5, 0.02)
        c = math.cos(angle_world)
        s = math.sin(angle_world)

        dist = self.scan_range_min
        while dist <= self.scan_range_max:
            px = self.x + c * dist
            py = self.y + s * dist
            if self._is_occupied(px, py):
                return dist
            dist += step
        return float("inf")

    def _publish_scan(self) -> None:
        if self._map is None:
            return

        now = self.get_clock().now().to_msg()

        scan = LaserScan()
        scan.header.stamp = now
        scan.header.frame_id = self.frame_base
        scan.angle_min = float(self.scan_angle_min)
        scan.angle_max = float(self.scan_angle_max)
        scan.angle_increment = float(self.scan_angle_inc)
        scan.range_min = float(self.scan_range_min)
        scan.range_max = float(self.scan_range_max)
        scan.time_increment = 0.0
        scan.scan_time = 0.0

        ranges = []
        for i in range(self.scan_count):
            a = self.scan_angle_min + float(i) * self.scan_angle_inc
            d = self._cast_ray(self.yaw + a)
            if not math.isfinite(d):
                ranges.append(float(self.scan_range_max))
            else:
                ranges.append(float(np.clip(d, self.scan_range_min, self.scan_range_max)))

        scan.ranges = ranges
        self.scan_pub.publish(scan)
        if self.scan_topic2 and self.scan_topic2 != self.scan_topic:
            self.scan_pub2.publish(scan)


def main(args=None):
    rclpy.init(args=args)
    node = Sim2D()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
