#pragma once

/// 从文件读入BAL dataset
class BALProblem {
public:
    /// load bal data from text file
    explicit BALProblem(const std::string &filename, bool use_quaternions = false);  // 定义显式转换（可选用四元数模式读取）

    ~BALProblem() {
        delete[] point_index_;
        delete[] camera_index_;
        delete[] observations_;
        delete[] parameters_;
    }

    /// save results to text file
    void WriteToFile(const std::string &filename) const;

    /// save results to ply pointcloud
    void WriteToPLYFile(const std::string &filename) const;

    void Normalize();  

    void Perturb(const double rotation_sigma,    // 向数据中加入噪声
                 const double translation_sigma,
                 const double point_sigma);

    int camera_block_size() const { return use_quaternions_ ? 10 : 9; }

    int point_block_size() const { return 3; }

    int num_cameras() const { return num_cameras_; }

    int num_points() const { return num_points_; }

    int num_observations() const { return num_observations_; }

    int num_parameters() const { return num_parameters_; }

    const int *point_index() const { return point_index_; }

    const int *camera_index() const { return camera_index_; }

    const double *observations() const { return observations_; }

    const double *parameters() const { return parameters_; }

    const double *cameras() const { return parameters_; }

    const double *points() const { return parameters_ + camera_block_size() * num_cameras_; }

    /// camera参数的起始地址
    double *mutable_cameras() { return parameters_; }

    double *mutable_points() { return parameters_ + camera_block_size() * num_cameras_; }

    double *mutable_camera_for_observation(int i) {
        return mutable_cameras() + camera_index_[i] * camera_block_size();
    }

    double *mutable_point_for_observation(int i) {
        return mutable_points() + point_index_[i] * point_block_size();
    }

    const double *camera_for_observation(int i) const {
        return cameras() + camera_index_[i] * camera_block_size();
    }

    const double *point_for_observation(int i) const {
        return points() + point_index_[i] * point_block_size();
    }

private:
    void CameraToAngelAxisAndCenter(const double *camera,
                                    double *angle_axis,
                                    double *center) const;

    void AngleAxisAndCenterToCamera(const double *angle_axis,
                                    const double *center,
                                    double *camera) const;

    int num_cameras_;
    int num_points_;
    int num_observations_;
    int num_parameters_;
    bool use_quaternions_;

    int *point_index_;      // 每个observation对应的point index
    int *camera_index_;     // 每个observation对应的camera index
    double *observations_;
    double *parameters_;
};
