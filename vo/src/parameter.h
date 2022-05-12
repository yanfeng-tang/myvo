#pragma once

#include <ros/ros.h>

#include <fstream>
#include <opencv2/opencv.hpp>

extern int ROW;
extern int COL;
const int WINDOW_SIZE = 10;
extern double FOCALLENGTH;
extern double MIN_PARALLAX;
extern std::string CAM_NAME;
extern cv::Mat distCoeffs;
const int MIN_OBS_SIZE = 2;
const int SIZE_POSE = 7;
// const int SIZE_POSE = 3;
const int FEATURE_SIZE = 1000;
const int SIZE_POINT = 1;
extern std::string VINS_RESULT_PATH;

void readParameter(ros::NodeHandle &n);