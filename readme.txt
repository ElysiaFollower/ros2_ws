1、导航框架已经实现(global.py+local.py),请实现补全全局规划算法（例如rrt*）、局部规划算法（例如dwa）并将其放在planner目录下。
2、进入ros2_ws，编译：colcon build
3、source install/setup.bash（每次新开终端都需要执行此命令）
4、ros2 run pubsub_package global_pub
5、ros2 run pubsub_package local_pub
6、ros2 launch turtlebot4_navigation localization.launch.py map:=/home/ubuntu/map.yaml 
7、ros2 launch turtlebot4_viz view_robot.launch.py 
8、将话题名称改为/course_agv/goal，方法见图1-4

9、无 ROS2 的 Windows 本机仿真（纯 Python）：
   9.1 环境准备（二选一）：
   - 推荐（PowerShell 临时设置）：$env:PYTHONPATH=(Resolve-Path pubsub_package).Path
   - 或（安装为可编辑包）：pip install -e .\\pubsub_package

   9.2 基本运行（内置地图 + 输出轨迹）：
   - python -m pubsub_package.sim_no_ros --out-png .\\log\\traj.png --out-csv .\\log\\traj.csv

   9.3 内置地图 + 自定义起点终点：
   - python -m pubsub_package.sim_no_ros --start X Y YAW --goal GX GY --out-png .\\log\\case.png

   9.4 使用你自己的地图（ROS map.yaml + pgm）：
   - python -m pubsub_package.sim_no_ros --map C:\\path\\to\\map.yaml --start X Y YAW --goal GX GY --out-png .\\log\\case.png

   9.5 交互显示（需要 matplotlib）：
   - python -m pubsub_package.sim_no_ros --plot

   9.6 仿真 Pipeline（No-ROS 端到端流程）：
   - 地图阶段：读取 map.yaml+pgm（或使用内置地图）→ 转为栅格 MapSpec（resolution/origin/grid）。
   - 全局阶段：从占据栅格抽取障碍点云 → RRT* 规划得到全局路径 path[(x,y)...]。
   - 局部闭环：循环直到到达/失败：
     ① 在全局 path 上找最近点并前瞻得到局部目标；同时截取一段局部 path_points（用于“贴着全局路走”）。
     ② 在栅格上做射线投射模拟 LaserScan → 生成机器人坐标系障碍点云 points_cloud。
     ③ DWA 在动态窗口内采样(v,w)并滚动预测轨迹，用 goal/path/heading/obstacle/speed 多个代价打分选最优速度。
     ④ 用简化运动学更新(x,y,yaw)并做 footprint 碰撞检测，记录轨迹 traj。
   - 输出阶段：打印 success/path_len/traj_len；可写 CSV/PNG（轨迹与全局路径可视化）。

10、WSL(ROS2) 仿真：
   - 编译后：ros2 launch pubsub_package sim2d.launch.py real:=false

10、在 RViz 显示“膨胀图”（不依赖 Nav2 costmap）：
   - 启动 global_pub 时开启膨胀图发布（默认开启），topic 为 /course_agv/inflated_map
   - RViz2 中 Add -> Map，Topic 选择 /course_agv/inflated_map
   - 可调参数：robot_radius（机器人半径）、inflation_radius（额外膨胀距离）

11、实时参数控制面板（Ubuntu/WSL 浏览器打开，实时调整下次规划/控制使用的参数）：
   - global_pub：ros2 run pubsub_package global_pub --ros-args -p enable_param_panel:=true
     浏览器访问：http://127.0.0.1:8891/
   - local_pub：ros2 run pubsub_package local_pub --ros-args -p real:=true -p enable_param_panel:=true
     浏览器访问：http://127.0.0.1:8890/
   - 可选自动打开浏览器：加参数 -p param_panel_open_browser:=true
   - 改端口/绑定地址：-p param_panel_port:=889X -p param_panel_host:=127.0.0.1

12、Windows 无 ROS2 仿真增强版（单面板：点选起终点 + 同时调 global/local 参数 + 自动绘制路径/轨迹）：
   12.1 PowerShell 运行（推荐）：
   - $env:PYTHONPATH=(Resolve-Path pubsub_package).Path
   - python -m pubsub_package.sim_no_ros_panel --map .\\pubsub_package\\maps\\simple_map.yaml --open-browser
   12.2 操作说明：
   - 左键点击地图：设置 start（起点），右键点击地图：设置 goal（终点）
   - 点击 Run：运行一次 “RRT* 全局 + DWA 局部闭环”，自动画出蓝色全局路径与红色轨迹
   - 在右侧参数栏修改参数后再次 Run：参数会作用于下一次规划/控制
