cmake_minimum_required(VERSION 3.12)
project(directBA)

# 设置C++标准
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 查找依赖库
find_package(OpenCV REQUIRED)
find_package(Sophus REQUIRED)
find_package(Pangolin REQUIRED)
find_package(g2o REQUIRED COMPONENTS 
    core
    types_sba
    types_slam3d
    solver_csparse
    solver_dense
    solver_eigen
    stuff
)

include_directories(
    ${OpenCV_INCLUDE_DIRS}
    ${G2O_INCLUDE_DIRS}
    ${Sophus_INCLUDE_DIRS}
    "/usr/include/eigen3/"
    ${Pangolin_INCLUDE_DIRS}
)

# 添加可执行文件
add_executable(directBA directBA.cpp)

# 链接库
target_link_libraries(directBA
    ${OpenCV_LIBS}
    ${Sophus_LIBRARIES}
    ${Pangolin_LIBRARIES}
    g2o_core
    g2o_types_sba
    g2o_types_slam3d
    g2o_solver_csparse
    g2o_solver_dense
    g2o_stuff
    g2o_csparse_extension
    ${CSPARSE_LIBRARIES}
)