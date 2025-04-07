cmake_minimum_required(VERSION 3.12)
project(direct_method)

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
add_executable(direct_method direct_method.cpp)

# 链接库
target_link_libraries(
    direct_method 
    ${OpenCV_LIBS}
    ${G2O_LIBRARIES}
    ${Sophus_LIBRARIES}
    ${Pangolin_LIBRARIES}
)