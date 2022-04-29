#include "feature_manager.h"

double FeatureManager::compemsateParallax(const FeaturePerId &it_pre_id,
                                          int frame_count) {
  // last obs and first obs
  const FeaturePerFrame frame_thrid =
      it_pre_id.feature_per_frame[frame_count - 2 - it_pre_id.start_frame];
  const FeaturePerFrame frame_second =
      it_pre_id.feature_per_frame[frame_count - 1 - it_pre_id.start_frame];

  Eigen::Vector2d p_thrid = frame_thrid.obs;
  Eigen::Vector2d p_second = frame_second.obs;
  double du = p_second(0) - p_thrid(0);
  double dv = p_second(1) - p_thrid(1);

  double ans = 0;
  ans = max(ans, sqrt(du * du + dv * dv));
  return ans;
}

double FeatureManager::compemsateParallax(
    const vector<vector<cv::Point2f>> &points) {
  double ans = 0;
  double du, dv;
  for (int i = 0; i < (int)points[0].size(); i++) {
    du = points[0][i].x - points[1][i].x;
    dv = points[0][i].y - points[1][i].y;
    ans += sqrt(du * du + dv * dv);
  }
  return ans;
}

bool FeatureManager::checkParallax(
    const std::vector<std::pair<int, cv::Point2f>> &points, int frame_count) {
  int feature_id, start_frame;
  Eigen::Vector2d obs;
  int feature_track_num = 0;
  int parallax_sum = 0;
  int parallax_num = 0;
  for (auto &id_pts : points) {
    obs = Eigen::Vector2d(id_pts.second.x, id_pts.second.y);
    feature_id = id_pts.first;
    auto it = find_if(feature.begin(), feature.end(),
                      [feature_id](const FeaturePerId &it) {
                        return feature_id == it.feature_id;
                      });

    if (it == feature.end()) {
      feature.push_back(FeaturePerId(feature_id, frame_count));
      feature.back().feature_per_frame.push_back(FeaturePerFrame(obs));
    } else {
      it->feature_per_frame.push_back(FeaturePerFrame(obs));
      feature_track_num++;
    }
  }

  if (frame_count < 5 || feature_track_num < 50) return true;

  for (auto &it_per_id : feature) {
    if (it_per_id.start_frame <= frame_count - 2 &&
        (it_per_id.feature_per_frame.size() + it_per_id.start_frame - 1 >=
         frame_count - 1))
      parallax_sum += compemsateParallax(it_per_id, frame_count);
    parallax_num++;
  }

  if (parallax_num == 0) return true;
  return (parallax_sum / parallax_num) >= MIN_PARALLAX;
}

/*
1.找距离较远的两帧进行F和H矩阵的求解
*/
void FeatureManager::Initialization(int frame_count, bool &result, cv::Mat &m_K,
                                    Eigen::Matrix3d Rwc[],
                                    Eigen::Vector3d twc[]) {
  vector<vector<cv::Point2f>> points;
  vector<int> points_id;
  if (extractPoints(0, frame_count, points, points_id)) {
    result = true;
    auto points_incam = points;
    toPixel(points);
    // cv::Mat E = cv::findEssentialMat(points[0], points[1], m_K);
    cv::Mat F = cv::findFundamentalMat(points[0], points[1]);

    cv::Mat K, R, t;
    m_K.convertTo(K, F.type());
    cv::Mat E = K.t() * F * K;
    cv::recoverPose(E, points[0], points[1], m_K, R, t);  // R0to1

    Eigen::Matrix3d R10;
    Eigen::Vector3d t10;
    cv::cv2eigen(R, R10);
    cv::cv2eigen(t, t10);

    cv::Mat T0 = cv::Mat::eye(3, 4, CV_64F);
    cv::Mat T10 = cv::Mat::eye(3, 4, CV_64F);
    Eigen::Matrix<double, 3, 4> T1, T2;
    T1.setZero();
    T2.setZero();

    T1.block(0, 0, 3, 3) = Eigen::Matrix3d::Identity();
    T2.block(0, 0, 3, 3) = R10;
    T2.rightCols<1>() = t10;

    cv::eigen2cv(T1, T0);
    cv::eigen2cv(T2, T10);

    cv::Mat pts_4d;
    cv::triangulatePoints(T0, T10, points_incam[0], points_incam[1], pts_4d);

    for (int i = 0; i < (int)pts_4d.cols; i++) {
      cv::Mat point3d = pts_4d.col(i);
      point3d /= point3d.at<float>(3, 0);

      cv::Point2f point(point3d.at<float>(0, 0), point3d.at<float>(1, 0));
      //把结果赋值给FeaturePerId
    }

    //归一化

    // triangulate
    Rwc[0] = Eigen::Matrix3d::Identity();
    twc[0] = Eigen::Vector3d::Zero();
    Rwc[frame_count] = Rwc[0] * R10.transpose();
    twc[frame_count] = Rwc[0] * (-t10);

    // cv::recoverPose()
    // Tangyf  分解基础矩阵求解R,t
  } else {
    result = false;
    ROS_INFO("not enough motion");
  }
}

bool FeatureManager::extractPoints(int frame_i, int frame_j,
                                   vector<vector<cv::Point2f>> &points,
                                   vector<int> &points_id) {
  ROS_WARN("start to extract points");
  // ROS_ASSERT(points.size() == 2);
  points.resize(2);
  cv::Point2f point;
  Eigen::Vector2d obs_i, obs_j;
  for (auto &it_per_id : feature) {
    int obs_size = it_per_id.feature_per_frame.size();
    int feature_id = it_per_id.feature_id;
    if (it_per_id.start_frame <= frame_i &&
        obs_size + it_per_id.start_frame - 1 >= frame_j) {
      obs_i = it_per_id.feature_per_frame[frame_i - it_per_id.start_frame].obs;
      obs_j = it_per_id.feature_per_frame[frame_j - it_per_id.start_frame].obs;
      point.x = obs_i(0);
      point.y = obs_i(1);
      points[0].push_back(point);
      point.x = obs_j(0);
      point.y = obs_j(1);
      points[1].push_back(point);
      points_id.push_back(feature_id);
    }
  }
  if (points[0].size() >= 8) {
    double parallax_sum = compemsateParallax(points);
    if ((parallax_sum / points[0].size()) >= MIN_PARALLAX) return true;
  }
  return false;
}

void FeatureManager::toPixel(vector<vector<cv::Point2f>> &points) {
  for (int i = 0; i < (int)points[0].size(); i++) {
    points[0][i].x = FOCALLENGTH * points[0][i].x + COL / 2;
    points[0][i].y = FOCALLENGTH * points[0][i].y + ROW / 2;
    points[1][i].x = FOCALLENGTH * points[1][i].x + COL / 2;
    points[1][i].y = FOCALLENGTH * points[1][i].y + ROW / 2;
  }
}