#pragma once

#include <opencv/highgui.h>

#include <Eigen/Core>
#include <eigen3/Eigen/Dense>
#include <list>
#include <opencv2/core/eigen.hpp>
#include <opencv2/opencv.hpp>
#include <thread>
#include <vector>

#include "factor/optimization.h"
#include "parameter.h"

using namespace std;
class FeaturePerFrame {
 public:
  Eigen::Vector2d obs;
  FeaturePerFrame(Eigen::Vector2d &_obs) : obs(_obs) {}
};

class FeaturePerId {
 public:
  int feature_id;
  int start_frame;
  double estimate_depth;
  bool is_triangulate = false;
  int solve_flag;
  vector<FeaturePerFrame> feature_per_frame;

  FeaturePerId(int _feature_id, int _start_frame)
      : feature_id(_feature_id), start_frame(_start_frame) {}
  int endframe();
};

class FeatureManager {
 public:
  bool checkParallax(const std::vector<std::pair<int, cv::Point2f>> &points,
                     int frame_count);
  double compemsateParallax(const FeaturePerId &it_per_id, int frame_count);
  double compemsateParallax(const vector<vector<cv::Point2f>> &points);
  void toPixel(vector<vector<cv::Point2f>> &points);
  void toCamera(vector<vector<cv::Point2f>> &points);

  void Initialization(int frame_count, bool &result, cv::Mat &m_K,
                      Eigen::Matrix3d Rwc[], Eigen::Vector3d twc[]);
  bool extractPoints(int frame_i, int frame_j,
                     vector<vector<cv::Point2f>> &points,
                     vector<int> &points_id);
  void slidWindowOld(Eigen::Matrix3d back_R, Eigen::Vector3d back_t,
                     Eigen::Matrix3d Rwc[], Eigen::Vector3d twc[]);
  void slidWindowNew();
  void triangulatePoint(Eigen::Matrix3d Rwc[], Eigen::Vector3d twc[]);

  Eigen::VectorXd getFeatureDepth();
  int getFeatureCount();
  void setDepth(const Eigen::VectorXd &dep);

  void checkTransformation(const Eigen::Matrix3d Rwc[],
                           const Eigen::Vector3d twc[]);
  void removeFailures();

  void BASolvetrans(Eigen::Matrix3d Rwc[], Eigen::Vector3d twc[]);

  list<FeaturePerId> feature;
};