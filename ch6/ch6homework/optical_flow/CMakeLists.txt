cmake_minimum_required(VERSION 3.12)
project(optical_flow)

# 设置C++标准
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 查找库
find_package(OpenCV REQUIRED)
find_package(Sophus REQUIRED)
find_package(Pangolin REQUIRED)

include_directories(
    ${OpenCV_INCLUDE_DIRS}
    ${G2O_INCLUDE_DIRS}
    ${Sophus_INCLUDE_DIRS}
    "/usr/include/eigen3/"
    ${Pangolin_INCLUDE_DIRS}
)

# 添加可执行文件
add_executable(optical_flow optical_flow.cpp)

# 链接库
target_link_libraries(
    optical_flow 
    ${OpenCV_LIBS}
    ${G2O_LIBRARIES}
    ${Sophus_LIBRARIES}
    ${Pangolin_LIBRARIES}
)