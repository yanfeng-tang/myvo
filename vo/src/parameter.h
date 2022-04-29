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

void readParameter(ros::NodeHandle &n);