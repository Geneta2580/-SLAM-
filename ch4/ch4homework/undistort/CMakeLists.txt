cmake_minimum_required(VERSION 3.12)
project(undistort)

# 设置C++标准
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 查找库
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

# 添加可执行文件
add_executable(undistort undistort_image.cpp)

# 链接库
target_link_libraries(undistort ${OpenCV_LIBS})