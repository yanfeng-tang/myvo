#pragma once

#include <ceres/ceres.h>
#include <opencv/highgui.h>

#include <vector>

#include "camodocal/camera_models/CameraFactory.h"
#include "camodocal/camera_models/CataCamera.h"
#include "camodocal/camera_models/EquidistantCamera.h"
#include "camodocal/camera_models/PinholeCamera.h"
#include "factor/optimization.h"
#include "factor/projection_factor.h"
#include "feature_manager.h"
#include "parameter.h"
using namespace std;

class Estimator {
 public:
  void ProcessImage(const vector<pair<int, cv::Point2f>> &points);
  void readIntrinsicParameter(std::string &calib_file);
  void slidWindow();
  void solveOdometry();
  void optimization();
  void vector2double();
  void double2vector();
  void checkQ();
  void checkResidual(int cam_i, int cam_j, Eigen::Vector3d obs_i, Eigen::Vector3d obs_j);

  int frame_count = 0;
  enum MarginalizationFlag { MARGIN_OLD = 0, MARGIN_NEW = 1 };
  enum SolverFlag { INITIAL = 0, NON_LINEAR = 1 };
  SolverFlag solver_flag;
  MarginalizationFlag marginalization_flag;
  camodocal::CameraPtr m_camera;
  cv::Mat un_map1, un_map2;
  cv::Mat m_K = cv::Mat::eye(3, 3, CV_32F);

  Eigen::Matrix3d Rwc[(WINDOW_SIZE + 1)];
  Eigen::Vector3d twc[(WINDOW_SIZE + 1)];
  FeatureManager f_manager;

  double para_pose[WINDOW_SIZE + 1][SIZE_POSE];
  // double para_poseT[WINDOW_SIZE + 1][SIZE_POSET];
  double para_feature[FEATURE_SIZE][SIZE_POINT];
};