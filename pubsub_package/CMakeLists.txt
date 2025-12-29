# Minimum CMake version required
cmake_minimum_required(VERSION 3.5)

# Project name (use your package name)
project(pubsub_package)

# Find ament_cmake package
find_package(ament_cmake REQUIRED)

# Find rosidl_default_generators package to generate service/message interfaces
find_package(rosidl_default_generators REQUIRED)

# Generate interfaces (list your .srv, .msg, or .action files here)
rosidl_generate_interfaces(${PROJECT_NAME}
  "pubsub_package/srv/Plan.srv" 
)

# Mark the package as an ament package (necessary for ROS 2 packages)
ament_package()
