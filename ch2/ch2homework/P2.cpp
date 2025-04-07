#include "P2.h"

using namespace Eigen;
using namespace std;

int P2()
{
    const int n = 100;

    // 生成对称正定矩阵A（确保可Cholesky分解）
    MatrixXd M = MatrixXd::Random(n, n);    // 随机矩阵
    MatrixXd A = M * M.transpose();         // 构造对称半正定矩阵
    A += MatrixXd::Identity(n, n) * 1e-3;   // 添加小扰动确保正定性

    // 生成随机向量b
    VectorXd b = VectorXd::Random(n);

    // QR分解求解
    HouseholderQR<MatrixXd> qr(A);
    VectorXd x_qr = qr.solve(b);

    // Cholesky分解求解
    LLT<MatrixXd> chol(A);
    if (chol.info() != Success) {
        cerr << "Cholesky分解失败：矩阵不正定！" << endl;
        return 1;
    }
    VectorXd x_chol = chol.solve(b);

    //输出结果
    std::cout << "第二题结果:" << std::endl;
    std::cout << "QR result:" << std::endl;
    std::cout << x_qr << std::endl;

    std::cout << "Cholesky result:" << std::endl;
    std::cout << x_chol << std::endl;

    // 验证结果
    cout << "QR解与Cholesky解的差异范数: " 
         << (x_qr - x_chol).norm() << endl;
    cout << "QR残差范数: " << (A * x_qr - b).norm() << endl;
    cout << "Cholesky残差范数: " << (A * x_chol - b).norm() << endl;

    return 0;
}