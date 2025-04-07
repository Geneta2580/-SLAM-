cmake_minimum_required(VERSION 3.12)
project(draw_trajectory)

# 设置C++标准
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 查找库
find_package(Eigen3 REQUIRED)
include_directories(${EIGEN3_INCLUDE_DIR})

find_package(Sophus REQUIRED)
include_directories(${Sophus_INCLUDE_DIRS})

find_package(Pangolin REQUIRED)
include_directories(${Pangolin_INCLUDE_DIRS})

# 添加可执行文件
add_executable(draw_trajectory draw_trajectory.cpp)

# 链接库
target_link_libraries(draw_trajectory     
    ${Sophus_LIBRARIES}
    ${Pangolin_LIBRARIES}
)