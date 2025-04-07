#include <sophus/se3.hpp>
#include <string>
#include <iostream>
#include <fstream>
#include <unistd.h>

// need pangolin for plotting trajectory
#include <pangolin/pangolin.h>

using namespace std;

// path to trajectory file
string trajectory_file = "../trajectory.txt";

string trajectory_file1 = "../estimated.txt";

string trajectory_file2 = "../groundtruth.txt";

// function for plotting trajectory, don't edit this code
// start point is red and end point is blue
void DrawTrajectory(vector<Sophus::SE3d, Eigen::aligned_allocator<Sophus::SE3d>>);

int main(int argc, char **argv) {

    vector<Sophus::SE3d, Eigen::aligned_allocator<Sophus::SE3d>> poses, poses1, poses2;

    double norm, RMSE = 0.0;

    /// implement pose reading code
    // start your code here (5~10 lines)

    ifstream fin(trajectory_file);
    ifstream fin1(trajectory_file1);
    ifstream fin2(trajectory_file2);

    while(!fin.eof())
    {
        double time, tx, ty, tz, qx, qy, qz, qw;
        fin >> time >> tx >> ty >> tz >> qx >> qy >> qz >> qw; // 以空格分割，同时将每行的结果赋给每个变量
        if (fin.fail()) break;
        Eigen::Quaterniond q(qw, qx, qy, qz);
        Eigen::Vector3d t(tx, ty, tz);
        Sophus::SE3d pose(q, t); // SE3d是SE3的double精度版
        poses.push_back(pose);
    }
    fin.close();

    while(!fin1.eof())
    {
        double time1, tx1, ty1, tz1, qx1, qy1, qz1, qw1;
        fin1 >> time1 >> tx1 >> ty1 >> tz1 >> qx1 >> qy1 >> qz1 >> qw1; // 以空格分割，同时将每行的结果赋给每个变量
        if (fin1.fail()) break;
        Eigen::Quaterniond q1(qw1, qx1, qy1, qz1);
        Eigen::Vector3d t1(tx1, ty1, tz1);
        Sophus::SE3d pose1(q1, t1); // SE3d是SE3的double精度版
        poses1.push_back(pose1);
    }
    fin1.close();

    while(!fin2.eof())
    {
        double time2, tx2, ty2, tz2, qx2, qy2, qz2, qw2;
        fin2 >> time2 >> tx2 >> ty2 >> tz2 >> qx2 >> qy2 >> qz2 >> qw2; // 以空格分割，同时将每行的结果赋给每个变量
        if (fin2.fail()) break;
        Eigen::Quaterniond q2(qw2, qx2, qy2, qz2);
        Eigen::Vector3d t2(tx2, ty2, tz2);
        Sophus::SE3d pose2(q2, t2); // SE3d是SE3的double精度版
        poses2.push_back(pose2);
    }
    fin2.close();

    for (size_t i = 0; i < poses1.size(); i++)
     {
        Sophus::SE3d deltaPose = poses1[i].inverse() * poses2[i];
        Eigen::Matrix<double, 6, 1> lieAlgebra = deltaPose.log();
        norm += pow(lieAlgebra.norm(), 2);
    }
    RMSE = sqrt(norm/poses1.size());
    
    // end your code here

    // draw trajectory in pangolin
    cout << "RMSE:" << RMSE << endl;
    DrawTrajectory(poses);

    return 0;
}

/*******************************************************************************************/
void DrawTrajectory(vector<Sophus::SE3d, Eigen::aligned_allocator<Sophus::SE3d>> poses) {
    if (poses.empty()) {
        cerr << "Trajectory is empty!" << endl;
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
        glClearColor(1.0f, 1.0f, 1.0f, 1.0f);

        glLineWidth(2);
        for (size_t i = 0; i < poses.size() - 1; i++) {
            glColor3f(1 - (float) i / poses.size(), 0.0f, (float) i / poses.size());
            glBegin(GL_LINES);
            auto p1 = poses[i], p2 = poses[i + 1];
            glVertex3d(p1.translation()[0], p1.translation()[1], p1.translation()[2]);
            glVertex3d(p2.translation()[0], p2.translation()[1], p2.translation()[2]);
            glEnd();
        }
        pangolin::FinishFrame();
        usleep(5000);   // sleep 5 ms
    }

}