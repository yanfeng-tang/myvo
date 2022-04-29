#pragma once

#include <eigen3/Eigen/Dense>
#include <opencv2/opencv.hpp>
#include <queue>

#include "camodocal/camera_models/CameraFactory.h"
#include "camodocal/camera_models/CataCamera.h"
#include "camodocal/camera_models/EquidistantCamera.h"
#include "camodocal/camera_models/PinholeCamera.h"
#include "parameters.h"

using namespace std;
using namespace camodocal;
using namespace Eigen;
using namespace cv;

void reduceVector(vector<cv::Point2f> &pts, vector<uchar> &status);
void reduceVector(vector<int> &ids, vector<uchar> &status);
bool inBord(cv::Point2f &pts);

class FeatureTracker {
 public:
  // FeatureTracker();
  void readIntrinsicParameter(const std::string &config_file);
  void readImage(const cv::Mat &_img);
  void rejectwithF();
  void setMask(cv::Mat &mask);
  void addPoints(vector<cv::Point2f> &add_points);
  void showPointMatches(const cv::Mat &pre_img, const cv::Mat &forw_img);
  void updateID();
  void undistPoints(vector<cv::Point2f> &unpts);

  cv::Mat pre_img, cur_img, forw_img;
  vector<cv::Point2f> pre_pts, cur_pts, forw_pts;
  vector<int> ids;
  camodocal::CameraPtr m_camera;
  int n_id = 0;
};