#pragma once

#include <geometry_msgs/PointStamped.h>
#include <nav_msgs/Odometry.h>
#include <nav_msgs/Path.h>
#include <ros/ros.h>
#include <sensor_msgs/PointCloud.h>
#include <std_msgs/Header.h>
#include <tf/transform_broadcaster.h>
#include <visualization_msgs/Marker.h>

#include <eigen3/Eigen/Dense>

#include "../estimator.h"
// #include "../parameter.h"

extern ros::Publisher pub_odometry;
extern ros::Publisher pub_point_cloud, pub_margin_cloud;

void registerPub(ros::NodeHandle &n);

void pubOdometry(const Estimator &estimator, const std_msgs::Header &header);
void pubPointCloud(const Estimator &estimator, const std_msgs::Header &header);