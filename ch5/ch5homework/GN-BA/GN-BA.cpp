//
// Created by xiang on 12/21/17.
//

#include <Eigen/Core>
#include <Eigen/Dense>

using namespace Eigen;

#include <vector>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <chrono>

#include "sophus/se3.hpp" 

using namespace std;

typedef vector<Vector3d, Eigen::aligned_allocator<Vector3d>> VecVector3d;
typedef vector<Vector2d, Eigen::aligned_allocator<Vector2d>> VecVector2d;
typedef Matrix<double, 6, 1> Vector6d;

string p3d_file = "../p3d.txt";
string p2d_file = "../p2d.txt";

int main(int argc, char **argv) {

    chrono::steady_clock::time_point t1 = chrono::steady_clock::now();
    VecVector2d p2d;
    VecVector3d p3d;
    Matrix3d K;
    double fx = 520.9, fy = 521.0, cx = 325.1, cy = 249.7;
    K << fx, 0, cx, 0, fy, cy, 0, 0, 1;

    // load points in to p3d and p2d 读入3D、2D点信息
    // START YOUR CODE HERE

    std::ifstream p3dfile(p3d_file);
    std::ifstream p2dfile(p2d_file);

    double x, y, z;
    double u, v;

    while(p3dfile >> x >> y >> z)
    {
        Vector3d p3d1(x, y, z);
        p3d.push_back(p3d1);
    }

    while(p2dfile >> u >> v)
    {
        Vector2d p2d1(u, v);
        p2d.push_back(p2d1);
    }

    // END YOUR CODE HERE
    assert(p3d.size() == p2d.size());

    int iterations = 100;
    double cost = 0, lastCost = 0;
    int nPoints = p3d.size();
    cout << "points: " << nPoints << endl;

    Sophus::SE3d T_esti; // estimated pose

    for (int iter = 0; iter < iterations; iter++) {

        Matrix<double, 6, 6> H = Matrix<double, 6, 6>::Zero();
        Vector6d b = Vector6d::Zero();

        cost = 0;
        // compute cost
        for (int i = 0; i < nPoints; i++) {
            // compute cost for p3d[I] and p2d[I] (计算重投影误差 根据投影模型计算的投影像素点和实际像素点之间的差值)
            // START YOUR CODE HERE 
            Eigen::Vector3d pc = T_esti * p3d[i]; // 空间点变换到相机坐标系下
            double inv_z = 1.0 / pc[2];
            double inv_z2 = inv_z * inv_z;

            Eigen::Vector2d proj(fx * pc[0] / pc[2] + cx, fy * pc[1] / pc[2] + cy);
            Eigen::Vector2d e = p2d[i] - proj;
            cost += e.squaredNorm(); 

	    // END YOUR CODE HERE

	    // compute jacobian 雅可比矩阵,书187式7.46
            Matrix<double, 2, 6> J;
            // START YOUR CODE HERE 
            J << -fx*inv_z, 0, fx*pc[0]*inv_z2, fx*pc[0]*pc[1]*inv_z2, -fx-fx*pc[0]*pc[0]*inv_z2, fx*pc[1]*inv_z,
            0, -fy*inv_z, fy*pc[1]*inv_z2, fy+fy*pc[1]*pc[1]*inv_z2, -fy*pc[0]*pc[1]*inv_z2, -fy*pc[0]*inv_z;
	    // END YOUR CODE HERE

            H += J.transpose() * J;  // GN法用J阵来近似H整
            b += -J.transpose() * e;
        }

	// solve dx 
        Vector6d dx;

        // START YOUR CODE HERE 
        dx = H.ldlt().solve(b);  // 通过方程解出

        // END YOUR CODE HERE

        if (isnan(dx[0])) {
            cout << "result is nan!" << endl;
            break;
        }

        if (iter > 0 && cost >= lastCost) {
            // cost increase, update is not good
            cout << "cost: " << cost << ", last cost: " << lastCost << endl;
            break;
        }

        // update your estimation
        // START YOUR CODE HERE 
        T_esti = Sophus::SE3d::exp(dx)*T_esti;

        // END YOUR CODE HERE
        
        lastCost = cost;

        cout << "iteration " << iter << " cost=" << cout.precision(12) << cost << endl;
    }

    cout << "estimated pose: \n" << T_esti.matrix() << endl;
    chrono::steady_clock::time_point t2 = chrono::steady_clock::now();
    chrono::duration<double> time_used = chrono::duration_cast < chrono::duration < double >> (t2 - t1);
    cout << "用时：" << time_used.count() << " 秒。" << endl;     
    return 0;
}