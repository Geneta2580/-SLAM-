//
// Created by 高翔 on 2017/12/19.
// 本程序演示如何从Essential矩阵计算R,t
//

#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <cmath>

using namespace Eigen;

#include <sophus/so3.hpp>

#include <iostream>

using namespace std;

int main(int argc, char **argv) {

    // 给定Essential矩阵
    Matrix3d E;
    E << -0.0203618550523477, -0.4007110038118445, -0.03324074249824097,
            0.3939270778216369, -0.03506401846698079, 0.5857110303721015,
            -0.006788487241438284, -0.5815434272915686, -0.01438258684486258;

    // 待计算的R,t
    Matrix3d R;
    Vector3d t;

    // SVD and fix sigular values
    // START YOUR CODE HERE
    // SVD分解
    Eigen::JacobiSVD<Eigen::Matrix3d> svd(E, Eigen::ComputeFullU | Eigen::ComputeFullV);
    Eigen::Matrix3d U = svd.matrixU();
    Eigen::Vector3d sigma = svd.singularValues();
    Eigen::Matrix3d V = svd.matrixV();

    // 处理奇异值
    double Sigma1 = (sigma[0] + sigma[1]) / 2;
    double Sigma2 = (sigma[0] + sigma[1]) / 2;
    double Sigma3 = 0;

    Eigen::Matrix3d Sigma = Eigen::Matrix3d::Zero();
    Sigma(0, 0) = Sigma1;
    Sigma(1, 1) = Sigma2;
    Sigma(2, 2) = Sigma3;

    // END YOUR CODE HERE

    // set t1, t2, R1, R2 
    // START YOUR CODE HERE
    Matrix3d Rz1;
    Matrix3d Rz2;
    Matrix3d t_wedge1;
    Matrix3d t_wedge2;
    Matrix3d R1;
    Matrix3d R2;

    Sophus::SO3d rotation = Sophus::SO3d::exp(M_PI/2 * Eigen::Vector3d::UnitZ()); // 绕z轴旋转90°的旋转矩阵,z轴单位向量(李代数),然后指数映射为李群(旋转矩阵)
    
    Rz1 = rotation.matrix();
    Rz2 = Rz1.transpose();  // 正反转的转置关系

    t_wedge1 = U * Rz1 * U.transpose();
    t_wedge2 = U * Rz2 * U.transpose();

    R1 = U * Rz1 * V.transpose();
    R2 = U * Rz2 * V.transpose();   

    // END YOUR CODE HERE

    cout << "R1 = " << R1 << endl;
    cout << "R2 = " << R2 << endl;
    cout << "t1 = " << Sophus::SO3d::vee(t_wedge1) << endl;
    cout << "t2 = " << Sophus::SO3d::vee(t_wedge2) << endl;

    // check t^R=E up to scale
    Matrix3d tR = t_wedge1 * R1;
    cout << "t^R = " << tR << endl;

    Matrix3d k;
    k = tR.array() / E.array();
    cout << "coefficient = " << k << endl;

    return 0;
}