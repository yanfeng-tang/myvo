#pragma once

#include <ros/ros.h>

#include <opencv2/highgui/highgui.hpp>

extern std::string IMAGE_TOPIC;
extern std::string CAM_NAME;
extern int FREQUENCE;
extern bool PUBFRAME;
extern int ROW;
extern int COL;
extern int FOCALLENGTH;
extern int MAX_CNT;
extern int MIN_DIS;

void readparameters(ros::NodeHandle &n);
