#include "parameter.h"

std::string CAM_NAME;
int ROW;
int COL;
double FOCALLENGTH;
double MIN_PARALLAX;

void readParameter(ros::NodeHandle &n) {
  std::string config_file;
  if (n.getParam("calib_file", config_file))
    ROS_INFO("succeed load param form &s", config_file);
  else
    ROS_WARN("fail to load param");
  cv::FileStorage fsSetting(config_file, cv::FileStorage::READ);

  // std::cout << config_file << std::endl;

  if (!fsSetting.isOpened()) {
    ROS_INFO("error path to settings");
  }
  COL = fsSetting["image_width"];
  ROW = fsSetting["image_height"];
  FOCALLENGTH = 460.0;
  // WINDOW_SIZE = 10;
  MIN_PARALLAX = fsSetting["keyframe_parallax"];
  MIN_PARALLAX = MIN_PARALLAX / FOCALLENGTH;

  CAM_NAME = config_file;
}