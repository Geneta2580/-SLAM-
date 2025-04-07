#include "P3.h"

using namespace Eigen;
using namespace std;

int P3()
{
    // 小萝卜一号的四元数和位移向量
    Eigen::Quaterniond q1(0.55, 0.3, 0.2, 0.2);
    Eigen::Vector3d t1(0.7, 1.1, 0.2);

    // 小萝卜二号的四元数和位移向量
    Eigen::Quaterniond q2(-0.1, 0.3, -0.7, 0.2);
    Eigen::Vector3d t2(-0.1, 0.4, 0.8);

    // 点在小萝卜一号坐标系下的坐标
    Eigen::Vector3d p1(0.5, -0.1, 0.2);

    //注意四元数的归一化
    q1.normalize();
    q2.normalize();

    // 计算小萝卜一号的逆变换
    Eigen::Isometry3d T_c1w = Eigen::Isometry3d::Identity(); //这是一个4x4的矩阵，这里将其初始化为一个单位阵
    T_c1w.rotate(q1);
    T_c1w.pretranslate(t1);
    Eigen::Isometry3d T_wc1 = T_c1w.inverse();  // 世界系到小萝卜1坐标系坐标变换的逆变换即小萝卜1坐标系到世界系的坐标变换

    // 计算点在世界坐标系下的坐标
    Eigen::Vector3d p_w = T_wc1 * p1; //注意这里进行了扩展，将三维向量后面添1

    // 计算小萝卜二号的变换矩阵
    Eigen::Isometry3d T_c2w = Eigen::Isometry3d::Identity();
    T_c2w.rotate(q2);
    T_c2w.pretranslate(t2);

    // 计算点在小萝卜二号坐标系下的坐标
    Eigen::Vector3d p2 = T_c2w * p_w;

    // 输出结果
    std::cout << "第三题结果:" << std::endl;
    std::cout << "点在小萝卜二号坐标系下的坐标: " << p2.transpose() << std::endl;

    return 0;
}