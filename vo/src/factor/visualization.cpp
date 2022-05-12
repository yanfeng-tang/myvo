#include "visualization.h"

using namespace Eigen;

nav_msgs::Path path;
ros::Publisher pub_point_cloud, pub_margin_cloud;
ros::Publisher pub_odometry;

void registerPub(ros::NodeHandle &n) {
  pub_odometry = n.advertise<nav_msgs::Odometry>("odometry", 1000);
  pub_point_cloud = n.advertise<sensor_msgs::PointCloud>("point_cloud", 1000);
  pub_margin_cloud =
      n.advertise<sensor_msgs::PointCloud>("history_cloud", 1000);
}

void pubOdometry(const Estimator &estimator, const std_msgs::Header &header) {
  if (estimator.solver_flag == Estimator::SolverFlag::NON_LINEAR) {
    nav_msgs::Odometry odometry;
    Eigen::Quaterniond tmp_q;
    tmp_q = Eigen::Quaterniond(estimator.Rwc[WINDOW_SIZE]);

    odometry.header = header;
    odometry.header.frame_id = "world";
    odometry.child_frame_id = "world";

    odometry.pose.pose.position.x = estimator.twc[WINDOW_SIZE].x();
    odometry.pose.pose.position.y = estimator.twc[WINDOW_SIZE].y();
    odometry.pose.pose.position.z = estimator.twc[WINDOW_SIZE].z();
    odometry.pose.pose.orientation.x =
        Quaterniond(estimator.Rwc[WINDOW_SIZE]).x();
    odometry.pose.pose.orientation.y =
        Quaterniond(estimator.Rwc[WINDOW_SIZE]).y();
    odometry.pose.pose.orientation.z =
        Quaterniond(estimator.Rwc[WINDOW_SIZE]).z();
    odometry.pose.pose.orientation.w =
        Quaterniond(estimator.Rwc[WINDOW_SIZE]).w();

    ofstream foutC(VINS_RESULT_PATH, ios::app);
    foutC.setf(ios::fixed, ios::floatfield);
    foutC.precision(9);
    foutC << header.stamp.toSec() << " ";
    foutC.precision(5);

    foutC << estimator.twc[WINDOW_SIZE].x() << " "
          << estimator.twc[WINDOW_SIZE].y() << " "
          << estimator.twc[WINDOW_SIZE].z() << " " << tmp_q.x() << " "
          << tmp_q.y() << " " << tmp_q.z() << " " << tmp_q.w() << "\n";
    foutC.close();
  }
}

void pubPointCloud(const Estimator &estimator, const std_msgs::Header &header) {
  sensor_msgs::PointCloud point_cloud;
  point_cloud.header = header;

  for (auto &it_per_id : estimator.f_manager.feature) {
    if (it_per_id.is_triangulate) {
      int cam_i = it_per_id.start_frame;
      Vector3d pts_i;
      pts_i << it_per_id.feature_per_frame[0].obs * it_per_id.estimate_depth,
          it_per_id.estimate_depth;
      Vector3d w_pts_i = estimator.Rwc[cam_i] * pts_i + estimator.twc[cam_i];

      geometry_msgs::Point32 p;
      p.x = w_pts_i(0);
      p.y = w_pts_i(1);
      p.z = w_pts_i(2);
      point_cloud.points.push_back(p);
    }

    pub_point_cloud.publish(point_cloud);
  }

  sensor_msgs::PointCloud margin_cloud;
  margin_cloud.header = header;
  for (auto &it_per_id : estimator.f_manager.feature) {
    if (!it_per_id.is_triangulate) continue;

    if (it_per_id.start_frame == 0) {
      int cam_i = it_per_id.start_frame;
      Vector3d pts_i;
      pts_i << it_per_id.feature_per_frame[0].obs * it_per_id.estimate_depth,
          it_per_id.estimate_depth;
      Vector3d w_pts_i = estimator.Rwc[cam_i] * pts_i + estimator.twc[cam_i];

      geometry_msgs::Point32 p;
      p.x = w_pts_i(0);
      p.y = w_pts_i(1);
      p.z = w_pts_i(2);
      margin_cloud.points.push_back(p);
    }
  }
  pub_margin_cloud.publish(margin_cloud);
}