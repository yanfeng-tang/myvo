#include "estimator.h"

#include "feature_manager.h"

FeatureManager f_manager;

void Estimator::readIntrinsicParameter(std::string &calib_file) {
  ROS_INFO("read parameter form &s", calib_file.c_str());
  m_camera = camodocal::CameraFactory::instance()->generateCameraFromYamlFile(
      calib_file);
  m_K = m_camera->initUndistortRectifyMap(un_map1, un_map2);
  float fx = m_K.at<float>(0, 0);
  float fy = m_K.at<float>(1, 1);
  float cx = m_K.at<float>(0, 2);
  float cy = m_K.at<float>(1, 2);
  std::cout << "K" << fx << '\n' << fy << '\n' << cx << '\n' << cy << std::endl;
}

void Estimator::ProcessImage(const vector<pair<int, cv::Point2f>> &points) {
  /*
  1.根据视差确定关键帧
  2.初始化
  3.三角化
  4.BA
  */
  if (f_manager.checkParallax(points, frame_count))
    marginalization_flag = MARGIN_OLD;
  else
    marginalization_flag = MARGIN_NEW;

  //求解F和H
  if (/*solver_flag = INITIAL*/ 1) {
    if (frame_count == WINDOW_SIZE) {
      bool result;
      f_manager.Initialization(frame_count, result, m_K, Rwc, twc);
      if (!result) {
      }  // slidwindow
    } else {
      frame_count++;
      // std::cout << "frame count " << frame_count << std::endl;
    }
  } else {
    ROS_INFO("NON_LINEAR");
  }
}