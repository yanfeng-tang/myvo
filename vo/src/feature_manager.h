#pragma once

#include <opencv/highgui.h>

#include <Eigen/Core>
#include <eigen3/Eigen/Dense>
#include <list>
#include <opencv2/core/eigen.hpp>
#include <opencv2/opencv.hpp>
#include <thread>
#include <vector>

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
  vector<FeaturePerFrame> feature_per_frame;

  FeaturePerId(int _feature_id, int _start_frame)
      : feature_id(_feature_id), start_frame(_start_frame) {}
};

class FeatureManager {
 public:
  bool checkParallax(const std::vector<std::pair<int, cv::Point2f>> &points,
                     int frame_count);
  double compemsateParallax(const FeaturePerId &it_per_id, int frame_count);
  double compemsateParallax(const vector<vector<cv::Point2f>> &points);
  void toPixel(vector<vector<cv::Point2f>> &points);

  void Initialization(int frame_count, bool &result, cv::Mat &m_K,
                      Eigen::Matrix3d Rwc[], Eigen::Vector3d twc[]);
  bool extractPoints(int frame_i, int frame_j,
                     vector<vector<cv::Point2f>> &points);

  list<FeaturePerId> feature;
};