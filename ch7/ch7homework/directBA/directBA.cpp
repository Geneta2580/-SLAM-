//
// Created by xiang on 1/4/18.
// this program shows how to perform direct bundle adjustment
//
#include <iostream>

using namespace std;

#include <g2o/core/base_unary_edge.h>
#include <g2o/core/base_binary_edge.h>
#include <g2o/core/base_vertex.h>
#include <g2o/core/block_solver.h>
#include <g2o/core/optimization_algorithm_levenberg.h>
#include <g2o/solvers/dense/linear_solver_dense.h>
#include <g2o/core/robust_kernel.h>
#include <g2o/core/robust_kernel_impl.h>
#include <g2o/types/sba/types_six_dof_expmap.h>

#include <Eigen/Core>
#include <sophus/se3.hpp>
#include <opencv2/opencv.hpp>

#include <pangolin/pangolin.h>
#include <boost/format.hpp>

typedef vector<Sophus::SE3d, Eigen::aligned_allocator<Sophus::SE3d>> VecSE3;
typedef vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d>> VecVec3d;

// global variables
string pose_file = "../poses.txt";
string points_file = "../points.txt";

// intrinsics
float fx = 277.34;
float fy = 291.402;
float cx = 312.234;
float cy = 239.777;

// bilinear interpolation
inline float GetPixelValue(const cv::Mat &img, float x, float y) {
    uchar *data = &img.data[int(y) * img.step + int(x)];
    float xx = x - floor(x);
    float yy = y - floor(y);
    return float(
            (1 - xx) * (1 - yy) * data[0] +
            xx * (1 - yy) * data[1] +
            (1 - xx) * yy * data[img.step] +
            xx * yy * data[img.step + 1]
    );
}

// g2o vertex that use sophus::SE3d as pose
class VertexSophus : public g2o::BaseVertex<6, Sophus::SE3d> {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    VertexSophus() {}

    ~VertexSophus() {}

    bool read(std::istream &is) {return true;}

    bool write(std::ostream &os) const {return true;}

    virtual void setToOriginImpl() {
        _estimate = Sophus::SE3d();
    }

    virtual void oplusImpl(const double *update_) {
        Eigen::Map<const Eigen::Matrix<double, 6, 1>> update(update_);
        setEstimate(Sophus::SE3d::exp(update) * estimate()); // 李代数位姿左乘更新
    }
};

// TODO edge of projection error, implement it
// 16x1 error, which is the errors in patch
typedef Eigen::Matrix<double,16,1> Vector16d; // 手动定义一个16d的向量
class EdgeDirectProjection : public g2o::BaseBinaryEdge<16, Vector16d, g2o::VertexPointXYZ, VertexSophus> {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

    EdgeDirectProjection(float *color, cv::Mat &target) {
        this->origColor = color;
        this->targetImg = target;
    }

    ~EdgeDirectProjection() {}

    virtual void computeError() override {
        // TODO START YOUR CODE HERE
        // compute projection error ...
        // 获取顶点参数
        const g2o::VertexPointXYZ* point_vertex = static_cast<g2o::VertexPointXYZ*>(_vertices[0]); 
        const VertexSophus* pose_vertex = static_cast<VertexSophus*>(_vertices[1]);

        // 计算位姿变换后的点云空间坐标
        Eigen::Vector3d P = point_vertex->estimate();
        Sophus::SE3d T = pose_vertex->estimate();
        Eigen::Vector3d p_cam = T * P;

        if(p_cam.z() < 0.1) { // 检查投影后的深度是否有效
            _error.setZero();
            return;
        }

        // 计算像素坐标
        float u = (fx * p_cam.x()) / p_cam.z() + cx;
        float v = (fy * p_cam.y()) / p_cam.z() + cy;

        if(u < 2 || u > targetImg.cols-3 || v < 2 || v > targetImg.rows-3) { // 检查投影后的像素点是否有效
            _error.setZero();
            return;
        }
        
        // 提取当前的图像中patch的像素值(光度、灰度)
        Eigen::Matrix<double, 16, 1> current_patch;
        int idx = 0;
        for (int dx = -2; dx <= 1; ++dx) {
            for (int dy = -2; dy <= 1; ++dy) {
                current_patch[idx] = GetPixelValue(targetImg, u + dx, v + dy);
                idx++;
            }
        }

        // 计算周围16个像素的光度误差
        Eigen::Matrix<double,16,1> ref_patch;
        for(int i = 0; i < 16; ++i) {
            ref_patch[i] = static_cast<double>(origColor[i]);  // 显式转换float到double，直接赋值
        }
        _error = ref_patch - current_patch;
        // END YOUR CODE HERE
    }

    // Let g2o compute jacobian for you

    virtual bool read(istream &in) {return true;}

    virtual bool write(ostream &out) const {return true;}

private:
    cv::Mat targetImg;  // the target image
    float *origColor = nullptr;   // 16 floats, the color of this point
};

// plot the poses and points for you, need pangolin
void Draw(const VecSE3 &poses, const VecVec3d &points);

int main(int argc, char **argv) {

    // read poses and points
    VecSE3 poses;
    VecVec3d points;
    ifstream fin(pose_file);

    while (!fin.eof()) {   // 读取相机位姿信息
        double timestamp = 0;
        fin >> timestamp;
        if (timestamp == 0) break;
        double data[7];
        for (auto &d: data) fin >> d;
        poses.push_back(Sophus::SE3d(
                Eigen::Quaterniond(data[6], data[3], data[4], data[5]),
                Eigen::Vector3d(data[0], data[1], data[2])
        ));
        if (!fin.good()) break;
    }
    fin.close();


    vector<float *> color;
    fin.open(points_file); // 读取3维点云颜色位置信息
    while (!fin.eof()) {
        double xyz[3] = {0};
        for (int i = 0; i < 3; i++) fin >> xyz[i];
        if (xyz[0] == 0) break;
        points.push_back(Eigen::Vector3d(xyz[0], xyz[1], xyz[2]));
        float *c = new float[16];
        for (int i = 0; i < 16; i++) fin >> c[i];
        color.push_back(c);

        if (fin.good() == false) break;
    }
    fin.close();

    cout << "poses: " << poses.size() << ", points: " << points.size() << endl;

    // read images
    vector<cv::Mat> images;
    boost::format fmt("../%d.png"); // 注意图片路径也要改
    for (int i = 0; i < 7; i++) {
        images.push_back(cv::imread((fmt % i).str(), 0));
    }

    // build optimization problem
    typedef g2o::BlockSolver<g2o::BlockSolverTraits<6,3>> DirectBlock;
    auto linearSolver = std::make_unique<g2o::LinearSolverDense<DirectBlock::PoseMatrixType>>();
    auto solver_ptr = std::make_unique<DirectBlock>(std::move(linearSolver));
    auto solver = new g2o::OptimizationAlgorithmLevenberg(std::move(solver_ptr)); // L-M
    g2o::SparseOptimizer optimizer;
    optimizer.setAlgorithm(solver);
    optimizer.setVerbose(true);

    // TODO add vertices, edges into the graph optimizer
    // START YOUR CODE HERE

    // vertex
    vector<VertexSophus *> pose_vertices;
    vector<g2o::VertexPointXYZ *> point_vertices;
    
    // vertex1
    for(size_t i = 0; i < poses.size(); ++i){
        VertexSophus *v = new VertexSophus(); // 优化顶点相机位姿，优化后的结果存放在这里
        v->setId(i);
        v->setEstimate(poses[i]); // 从每帧图像给定的输入位姿开始优化
        if (i == 0) v->setFixed(true); // 固定第一帧位姿
        optimizer.addVertex(v); // 加入顶点
        pose_vertices.push_back(v);
    }

    // vertex2 g2o::VertexPointXYZ顶点默认采用加法更新
    for(size_t i = 0; i < points.size(); ++i){
        g2o::VertexPointXYZ* v = new g2o::VertexPointXYZ();// 优化的点云顶点
        v->setId(poses.size() + i); // 确保每个顶点ID唯一,和之前的位姿顶点错开
        v->setEstimate(points[i]); // 设置每个点的初始优化坐标
        v->setMarginalized(true); // 稀疏化优化关键步骤
        optimizer.addVertex(v); // 加入顶点
        point_vertices.push_back(v);
    }
    
    // edges
    for (size_t i = 0; i < poses.size(); ++i) {
        for (size_t j = 0; j < points.size(); ++j) {
            EdgeDirectProjection* edge = new EdgeDirectProjection(color[j], images[i]); // 传入颜色数据和目标图像
            edge->setVertex(0, dynamic_cast<g2o::VertexPointXYZ*>(point_vertices[j])); // 设置动态顶点注意这里的变量是变化的
            edge->setVertex(1, dynamic_cast<VertexSophus*>(pose_vertices[i])); 
            edge->setInformation(Eigen::Matrix<double, 16, 16>::Identity() * 1e-4);
            optimizer.addEdge(edge);
        }
    }

    // END YOUR CODE HERE

    // perform optimization
    optimizer.initializeOptimization(0);
    optimizer.optimize(200);

    // TODO fetch data from the optimizer
    // START YOUR CODE HERE
    for (int i=0; i<poses.size(); ++i) {
        poses[i] = pose_vertices[i]->estimate(); // 注意结果赋值都用estimate()
    }

    for (int i=0; i<points.size(); ++i) {
        points[i] = point_vertices[i]->estimate();
    }
    // END YOUR CODE HERE

    // plot the optimized points and poses
    Draw(poses, points);

    // delete color data
    for (auto &c: color) delete[] c;
    return 0;
}

void Draw(const VecSE3 &poses, const VecVec3d &points) {
    if (poses.empty() || points.empty()) {
        cerr << "parameter is empty!" << endl;
        return;
    }

    // create pangolin window and plot the trajectory
    pangolin::CreateWindowAndBind("Trajectory Viewer", 1024, 768);
    glEnable(GL_DEPTH_TEST);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    pangolin::OpenGlRenderState s_cam(
            pangolin::ProjectionMatrix(1024, 768, 500, 500, 512, 389, 0.1, 1000),
            pangolin::ModelViewLookAt(0, -0.1, -1.8, 0, 0, 0, 0.0, -1.0, 0.0)
    );

    pangolin::View &d_cam = pangolin::CreateDisplay()
            .SetBounds(0.0, 1.0, pangolin::Attach::Pix(175), 1.0, -1024.0f / 768.0f)
            .SetHandler(new pangolin::Handler3D(s_cam));


    while (pangolin::ShouldQuit() == false) {
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        d_cam.Activate(s_cam);
        glClearColor(0.0f, 0.0f, 0.0f, 0.0f);

        // draw poses
        float sz = 0.1;
        int width = 640, height = 480;
        for (auto &Tcw: poses) {
            glPushMatrix();
            Sophus::Matrix4f m = Tcw.inverse().matrix().cast<float>();
            glMultMatrixf((GLfloat *) m.data());
            glColor3f(1, 0, 0);
            glLineWidth(2);
            glBegin(GL_LINES);
            glVertex3f(0, 0, 0);
            glVertex3f(sz * (0 - cx) / fx, sz * (0 - cy) / fy, sz);
            glVertex3f(0, 0, 0);
            glVertex3f(sz * (0 - cx) / fx, sz * (height - 1 - cy) / fy, sz);
            glVertex3f(0, 0, 0);
            glVertex3f(sz * (width - 1 - cx) / fx, sz * (height - 1 - cy) / fy, sz);
            glVertex3f(0, 0, 0);
            glVertex3f(sz * (width - 1 - cx) / fx, sz * (0 - cy) / fy, sz);
            glVertex3f(sz * (width - 1 - cx) / fx, sz * (0 - cy) / fy, sz);
            glVertex3f(sz * (width - 1 - cx) / fx, sz * (height - 1 - cy) / fy, sz);
            glVertex3f(sz * (width - 1 - cx) / fx, sz * (height - 1 - cy) / fy, sz);
            glVertex3f(sz * (0 - cx) / fx, sz * (height - 1 - cy) / fy, sz);
            glVertex3f(sz * (0 - cx) / fx, sz * (height - 1 - cy) / fy, sz);
            glVertex3f(sz * (0 - cx) / fx, sz * (0 - cy) / fy, sz);
            glVertex3f(sz * (0 - cx) / fx, sz * (0 - cy) / fy, sz);
            glVertex3f(sz * (width - 1 - cx) / fx, sz * (0 - cy) / fy, sz);
            glEnd();
            glPopMatrix();
        }

        // points
        glPointSize(2);
        glBegin(GL_POINTS);
        for (size_t i = 0; i < points.size(); i++) {
            glColor3f(0.0, points[i][2]/4, 1.0-points[i][2]/4);
            glVertex3d(points[i][0], points[i][1], points[i][2]);
        }
        glEnd();

        pangolin::FinishFrame();
        usleep(5000);   // sleep 5 ms
    }
}

