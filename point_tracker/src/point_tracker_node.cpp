
#include <cv_bridge/cv_bridge.h>
#include <message_filters/subscriber.h>
#include <ros/ros.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/PointCloud.h>

#include <cstdio>
#include <iostream>

#include "point_tracker.h"

ros::Publisher pub_feature;
FeatureTracker trackdata;
int imgCount = 1;
bool firstimg = true;
double firsttime;

/**
 * @brief ROS的图像回调函数，对新来的图像进行特征点追踪，发布
 *
 * 使用createCLAHE对图像进行自适应直方图均衡化
 * calcOpticalFlowPyrLK() LK金字塔光流法，生成tracking的特征点
 * undistroted特征点
 * 然后把追踪的特征点发布到名字为pub_img的话题下，图像发布在在pub_match下
 * 被追踪的特征点是有全局唯一的ID的，后面就方便做匹配了
 */
void img_callback(const sensor_msgs::ImageConstPtr& img_msg) {
  if (firstimg) {
    firstimg = false;
    firsttime = img_msg->header.stamp.toSec();
  }
  if (round(1.0 * imgCount / (img_msg->header.stamp.toSec() - firsttime)) <=
      FREQUENCE)
    PUBFRAME = true;
  if (abs(1.0 * imgCount / (img_msg->header.stamp.toSec() - firsttime) -
          FREQUENCE) < 0.01 * FREQUENCE) {
    firsttime = img_msg->header.stamp.toSec();
    imgCount = 0;
  } else
    PUBFRAME = false;
  // copy img
  cv_bridge::CvImageConstPtr ptr =
      cv_bridge::toCvCopy(img_msg, sensor_msgs::image_encodings::MONO8);
  cv::Mat img = ptr->image;
  trackdata.readImage(img);
  trackdata.updateID();
  vector<cv::Point2f> un_pts;
  trackdata.undistPoints(un_pts);  //转成归一化
  // trackdata.showPointMatches(trackdata.pre_img, trackdata.forw_img);
  if (PUBFRAME) {
    imgCount++;
    sensor_msgs::PointCloudPtr feature(new sensor_msgs::PointCloud);
    sensor_msgs::ChannelFloat32 id_of_feature;
    sensor_msgs::ChannelFloat32 u_of_feature;
    sensor_msgs::ChannelFloat32 v_of_feature;
    geometry_msgs::Point32 p;
    feature->header = img_msg->header;
    feature->header.frame_id = "world";
    auto& cur_pts = trackdata.cur_pts;
    auto& ids = trackdata.ids;
    for (int i = 0; i < (int)un_pts.size(); i++) {
      p.x = un_pts[i].x;
      p.y = un_pts[i].y;
      p.z = 1.0;
      feature->points.push_back(p);
      id_of_feature.values.push_back(ids[i]);
      u_of_feature.values.push_back(cur_pts[i].x);
      v_of_feature.values.push_back(cur_pts[i].y);
    }
    feature->channels.push_back(id_of_feature);
    feature->channels.push_back(u_of_feature);
    feature->channels.push_back(v_of_feature);
    pub_feature.publish(feature);
  }
}

int main(int argc, char** argv) {
  ros::init(argc, argv, "point_tracker");
  ros::NodeHandle n("~");
  cout << "start point tracker" << endl;
  readparameters(n);

  trackdata.readIntrinsicParameter(CAM_NAME);

  ros::Subscriber sub_img = n.subscribe(IMAGE_TOPIC, 100, img_callback);
  pub_feature = n.advertise<sensor_msgs::PointCloud>("feature", 100);

  ros::spin();
  return 0;
}