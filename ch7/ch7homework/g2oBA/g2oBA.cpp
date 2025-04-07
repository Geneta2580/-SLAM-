#include <g2o/core/base_vertex.h>
#include <g2o/core/base_binary_edge.h>
#include <g2o/core/block_solver.h>
#include <g2o/core/optimization_algorithm_levenberg.h>
#include <g2o/solvers/csparse/linear_solver_csparse.h>
#include <g2o/core/robust_kernel_impl.h>
#include <iostream>
#include <memory> 

#include "common.h"
#include "sophus/se3.hpp"

using namespace Sophus;
using namespace Eigen;
using namespace std;

/// 姿态和内参的结构
struct PoseAndIntrinsics {
    PoseAndIntrinsics() {}

    /// set from given data address 从数据地址当中取值
    explicit PoseAndIntrinsics(double *data_addr) {   // 原始数据结构
        rotation = SO3d::exp(Vector3d(data_addr[0], data_addr[1], data_addr[2])); // 旋转
        translation = Vector3d(data_addr[3], data_addr[4], data_addr[5]); // 平移
        focal = data_addr[6]; // 相机内参
        k1 = data_addr[7];
        k2 = data_addr[8];
    }

    /// 将估计值放入内存，存储到对应数据地址当中
    void set_to(double *data_addr) {
        auto r = rotation.log();
        for (int i = 0; i < 3; ++i) data_addr[i] = r[i];
        for (int i = 0; i < 3; ++i) data_addr[i + 3] = translation[i];
        data_addr[6] = focal;
        data_addr[7] = k1;
        data_addr[8] = k2;
    }

    SO3d rotation; // 结构体真正的数据结构部分
    Vector3d translation = Vector3d::Zero();
    double focal = 0;
    double k1 = 0, k2 = 0;
};

/// 位姿加相机内参的顶点，9维，前三维为so3，接下去为t, f, k1, k2
class VertexPoseAndIntrinsics : public g2o::BaseVertex<9, PoseAndIntrinsics> {  // 维数，顶点类型
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

    VertexPoseAndIntrinsics() {} // 不接入外部参数值

    virtual void setToOriginImpl() override {
        _estimate = PoseAndIntrinsics(); // 顶点初始值的估计，设置初始数据
    }

    virtual void oplusImpl(const double *update) override {   // 每次更新增量的计算方式
        _estimate.rotation = SO3d::exp(Vector3d(update[0], update[1], update[2])) * _estimate.rotation; // 旋转
        _estimate.translation += Vector3d(update[3], update[4], update[5]); // 平移
        _estimate.focal += update[6]; // 后面三行都是相机内参的增量式更新
        _estimate.k1 += update[7];
        _estimate.k2 += update[8];
    }

    /// 根据估计值投影一个点
    Vector2d project(const Vector3d &point) {
        Vector3d pc = _estimate.rotation * point + _estimate.translation; // 位姿变换后的点
        pc = -pc / pc[2];  // 归一化坐标，注意这里是BAL数据的特殊性，假设成像在光心之后，所以要加上负号
        double r2 = pc.squaredNorm();
        double distortion = 1.0 + r2 * (_estimate.k1 + _estimate.k2 * r2); // 考虑相机的径向畸变模型，没有交叉项
        return Vector2d(_estimate.focal * distortion * pc[0],   // 修正畸变
                        _estimate.focal * distortion * pc[1]);
    }

    virtual bool read(istream &in) { return true; } // 补充返回值

    virtual bool write(ostream &out) const { return true; } // 补充返回值
};


// 估计三维路标点云坐标的顶点定义
class VertexPoint : public g2o::BaseVertex<3, Vector3d> {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

    VertexPoint() {}  // 初始化一下

    virtual void setToOriginImpl() override { // 三维点云初始值
        _estimate = Vector3d(0, 0, 0);
    }

    virtual void oplusImpl(const double *update) override {
        _estimate += Vector3d(update[0], update[1], update[2]); // 更新用加迭代
    }

    virtual bool read(istream &in) { return true; } // 补充返回值

    virtual bool write(ostream &out) const { return true; } // 补充返回值
};

// 路标点云和相机位姿平移内参之间的边定义
class EdgeProjection :
    public g2o::BaseBinaryEdge<2, Vector2d, VertexPoseAndIntrinsics, VertexPoint> {  // 边参数的维度，两个要优化的顶点
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

    virtual void computeError() override {
        auto v0 = (VertexPoseAndIntrinsics *) _vertices[0]; // 获取相机参数顶点参数
        auto v1 = (VertexPoint *) _vertices[1]; // 获取3D空间点路标顶点参数
        auto proj = v0->project(v1->estimate()); // 将估计的三维路标点进行投影，project函数包含在VertexPoseAndIntrinsics类中
        _error = proj - _measurement;   // 误差等于投影减去量测，重投影误差
    }

    // use numeric derivatives
    virtual bool read(istream &in) { return true; } // 补充返回值
    
    virtual bool write(ostream &out) const { return true; } // 补充返回值

};

void SolveBA(BALProblem &bal_problem);

int main(int argc, char **argv) {

    if (argc != 2) {
        cout << "usage: bundle_adjustment_g2o bal_data.txt" << endl;
        return 1;
    }

    BALProblem bal_problem(argv[1]);  // 使用BALProblem读入数据
    bal_problem.Normalize(); // 点云归一化处理
    bal_problem.Perturb(0.1, 0.5, 0.5); // 注入噪声，数值为注入噪声的方差
    bal_problem.WriteToPLYFile("initial.ply");
    SolveBA(bal_problem);
    bal_problem.WriteToPLYFile("final.ply");

    return 0;
}

void SolveBA(BALProblem &bal_problem) {
    const int point_block_size = bal_problem.point_block_size();  // 点云数据的尺寸
    const int camera_block_size = bal_problem.camera_block_size();  // 相机参数的尺寸，内参，位姿，平移
    double *points = bal_problem.mutable_points();  // 所有点云数据
    double *cameras = bal_problem.mutable_cameras(); // 所有相机参数

    // pose dimension 9, landmark is 3
    typedef g2o::BlockSolver<g2o::BlockSolverTraits<9, 3>> BlockSolverType;  // 求解器参数指定，BLOCKsolver稀疏矩阵优化器
    typedef g2o::LinearSolverCSparse<BlockSolverType::PoseMatrixType> LinearSolverType;
    // use LM
    auto solver = new g2o::OptimizationAlgorithmLevenberg(
        std::make_unique<BlockSolverType>(std::make_unique<LinearSolverType>())); // 使用LM算法下降
    g2o::SparseOptimizer optimizer;  // 构造求解器，把上面的参数传到下面
    optimizer.setAlgorithm(solver);
    optimizer.setVerbose(true);

    /// build g2o problem
    const double *observations = bal_problem.observations();
    // vertex
    vector<VertexPoseAndIntrinsics *> vertex_pose_intrinsics;
    vector<VertexPoint *> vertex_points;
    for (int i = 0; i < bal_problem.num_cameras(); ++i) { // 添加相机参数顶点
        VertexPoseAndIntrinsics *v = new VertexPoseAndIntrinsics();
        double *camera = cameras + camera_block_size * i;
        v->setId(i);   // 设置该顶点ID
        v->setEstimate(PoseAndIntrinsics(camera)); //设置该顶点初值
        optimizer.addVertex(v); // 添加顶点到优化器，后面一样
        vertex_pose_intrinsics.push_back(v); // push_back？
    }
    for (int i = 0; i < bal_problem.num_points(); ++i) {  // 添加3D点云点顶点
        VertexPoint *v = new VertexPoint();
        double *point = points + point_block_size * i;
        v->setId(i + bal_problem.num_cameras());
        v->setEstimate(Vector3d(point[0], point[1], point[2]));
        // g2o在BA中需要手动设置待Marg的顶点
        v->setMarginalized(true);
        optimizer.addVertex(v);
        vertex_points.push_back(v);
    }

    // edge
    for (int i = 0; i < bal_problem.num_observations(); ++i) { // 喂的是观测量的数据，就是像素点
        EdgeProjection *edge = new EdgeProjection; // 这里其实可以传入外部的参数，但注意private的声明
        edge->setVertex(0, vertex_pose_intrinsics[bal_problem.camera_index()[i]]); // 设置要连接的顶点，注意这里的ID和边定义里的顶点_vertices要对应
        edge->setVertex(1, vertex_points[bal_problem.point_index()[i]]); // 设置第二个顶点
        edge->setMeasurement(Vector2d(observations[2 * i + 0], observations[2 * i + 1])); // 设置量测为二维像素点
        edge->setInformation(Matrix2d::Identity()); // 设置信息矩阵(协方差矩阵之逆)，二维数据用2D矩阵
        edge->setRobustKernel(new g2o::RobustKernelHuber()); // 设置鲁棒核函数
        optimizer.addEdge(edge); // 添加边
    }

    optimizer.initializeOptimization(); //初始化
    optimizer.optimize(40); // 迭代轮次40次，执行优化

    // set to bal problem 把优化后的点云数据存到对应的结构体位置当中
    for (int i = 0; i < bal_problem.num_cameras(); ++i) {
        double *camera = cameras + camera_block_size * i;
        auto vertex = vertex_pose_intrinsics[i];
        auto estimate = vertex->estimate();
        estimate.set_to(camera);
    }
    for (int i = 0; i < bal_problem.num_points(); ++i) {
        double *point = points + point_block_size * i;
        auto vertex = vertex_points[i];
        for (int k = 0; k < 3; ++k) point[k] = vertex->estimate()[k];
    }
}
