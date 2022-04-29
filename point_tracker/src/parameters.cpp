#include "parameters.h"

#include <iostream>

std::string IMAGE_TOPIC;
std::string CAM_NAME;
int FREQUENCE;
bool PUBFRAME;
int ROW, COL;
int FOCALLENGTH;
int MAX_CNT;
int MIN_DIS;

template <typename T>
T readParam(ros::NodeHandle &n, std::string filename) {
  T ans;
  if (n.getParam(filename, ans)) {
    ROS_INFO_STREAM("Loaded" << filename << ":" << ans);
  } else {
    ROS_WARN_STREAM("Fail to load:" << filename);
    n.shutdown();
  }
  return ans;
}

void readparameters(ros::NodeHandle &n) {
  std::string config_file;
  /*
  1.判断是否打开文件
  2.读取文件
  3.释放文件
  */
  config_file = readParam<std::string>(n, "config_file");
  cv::FileStorage fsSettings(config_file, cv::FileStorage::READ);
  if (!fsSettings.isOpened()) {
    ROS_WARN_STREAM("Fail to read:" << config_file);
  } else {
    ROS_INFO_STREAM("Succeed to read:" << config_file);
  }

  fsSettings["image_topic"] >> IMAGE_TOPIC;
  FREQUENCE = fsSettings["freq"];
  ROW = fsSettings["image_height"];
  COL = fsSettings["image_width"];
  MAX_CNT = fsSettings["max_cnt"];
  MIN_DIS = fsSettings["min_dist"];
  CAM_NAME = config_file;

  fsSettings.release();

  PUBFRAME = false;
  FOCALLENGTH = 460;
}