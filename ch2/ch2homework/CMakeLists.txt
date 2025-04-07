cmake_minimum_required(VERSION 3.12)
project(LinearSolverDemo)

# 设置C++标准
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 查找Eigen3库
find_package(Eigen3 REQUIRED)

# 添加可执行文件
add_executable(ch2_homework src/main.cpp P2.cpp P3.cpp)

# 链接Eigen3库
target_link_libraries(ch2_homework Eigen3::Eigen)