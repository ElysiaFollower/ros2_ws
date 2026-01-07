# Global (RRT*)
ros2 run pubsub_package global_pub --ros-args \
  -p verbose_log:=true \
  -p publish_inflated_map:=true \
  -p inflated_map_topic:=/course_agv/inflated_map \
  -p robot_radius:=0.25 \
  -p inflation_radius:=0.10 \
  -p rrt.expand_dis:=0.40 \
  -p rrt.path_resolution:=0.05 \
  -p rrt.goal_sample_rate:=15 \
  -p rrt.max_iter:=6000 \
  -p rrt.connect_circle_dist:=2.0 \
  -p rrt.smooth_iter:=50

# Local (DWA)
ros2 run pubsub_package local_pub --ros-args \
  -p verbose_log:=true \
  -p real:=true \
  -p controller_frequency:=15.0 \
  -p robot_radius:=0.20 \
  -p safety_dist:=0.05 \
  -p laser_threshold:=2.5 \
  -p scan_stale_timeout:=0.5 \
  -p cmd_smoothing_alpha:=0.45 \
  -p cmd_max_accel:=1.2 \
  -p cmd_max_dyawrate:=6.0 \
  -p max_speed:=0.45 \
  -p max_yawrate:=1.2 \
  -p lookahead_dist:=1.0 \
  -p dwa.to_goal_cost_gain:=1.2 \
  -p dwa.path_cost_gain:=1.2 \
  -p dwa.heading_cost_gain:=0.05 \
  -p dwa.obstacle_cost_gain:=0.25 \
  -p dwa.speed_cost_gain:=1.2 \
  -p dwa.predict_time:=1.5 \
  -p dwa.v_samples:=10 \
  -p dwa.w_samples:=20
