#include <ros/ros.h>
#include <sensor_msgs/PointCloud.h>

#include <mutex>
#include <vector>

#include "estimator.h"

Estimator estimator;

void callback(const sensor_msgs::PointCloudConstPtr &point_msg) {
  ROS_WARN("succeed receive msg");
  vector<std::pair<int, cv::Point2f>> points;
  int feature_id;
  cv::Point2f point;
  for (int i = 0; i < (int)point_msg->points.size(); i++) {
    feature_id = point_msg->channels[0].values[i];
    // printf("point id is %d", feature_id);
    point.x = point_msg->points[i].x;
    point.y = point_msg->points[i].y;
    points.push_back({feature_id, point});
  }
  estimator.ProcessImage(points);
}

int main(int argc, char **argv) {
  ros::init(argc, argv, "MyVo");
  ROS_INFO("start myvo");
  ros::NodeHandle n("~");

  readParameter(n);
  estimator.readIntrinsicParameter(CAM_NAME);

  ros::Subscriber sub_point =
      n.subscribe("/point_tracker/feature", 100, callback);
  ros::spin();
  return 0;
}