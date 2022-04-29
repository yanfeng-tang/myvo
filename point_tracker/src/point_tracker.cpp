#include "point_tracker.h"

void reduceVector(vector<cv::Point2f> &pts, vector<uchar> &status) {
  int j = 0;
  for (size_t i = 0; i < pts.size(); i++) {
    if (status[i]) pts[j++] = pts[i];
  }
  pts.resize(j);
}

void reduceVector(vector<int> &ids, vector<uchar> &status) {
  int j = 0;
  for (size_t i = 0; i < ids.size(); i++) {
    if (status[i]) ids[j++] = ids[i];
  }
  ids.resize(j);
}

bool inBord(cv::Point2f &pts) {
  const int BORDER_SIZE = 1;
  return BORDER_SIZE < pts.x && pts.x < COL - BORDER_SIZE &&
         BORDER_SIZE < pts.y && pts.y < ROW - BORDER_SIZE;
}

void FeatureTracker::readIntrinsicParameter(const std::string &config_file) {
  ROS_INFO("parameter of camera is %s", config_file.c_str());
  m_camera = CameraFactory::instance()->generateCameraFromYamlFile(config_file);
}

void FeatureTracker::updateID() {
  for (int i = 0; i < (int)ids.size(); i++) {
    if (ids[i] == -1) ids[i] = n_id++;
  }
}

void FeatureTracker::showPointMatches(const cv::Mat &pre_img,
                                      const cv::Mat &forw_img) {
  cv::Mat img1Clone, img2Clone;
  if (pre_img.channels() != 3) {
    cv::cvtColor(pre_img, img1Clone, cv::COLOR_GRAY2BGR);
    cv::cvtColor(forw_img, img2Clone, cv::COLOR_GRAY2BGR);
  } else {
    img1Clone = pre_img;
    img2Clone = forw_img;
  }
  int totalrow = pre_img.rows > forw_img.rows ? pre_img.rows : forw_img.rows;
  cv::Mat showImg =
      cv::Mat::zeros(totalrow, pre_img.cols + forw_img.cols, img1Clone.type());
  cv::Mat leftimg(showImg, Rect(0, 0, pre_img.cols, pre_img.rows));
  cv::Mat rightimg(showImg,
                   Rect(pre_img.cols, 0, forw_img.cols, forw_img.rows));

  img1Clone.copyTo(leftimg);
  img2Clone.copyTo(rightimg);

  srand((unsigned int)time(NULL));
  cv::Scalar srandColor;
  int R = (rand() % (int)(255 + 1));
  int G = (rand() % (int)(255 + 1));
  int B = (rand() % (int)(255 + 1));

  srandColor = cv::Scalar(R, G, B);

  vector<cv::Point2f> un_pre_pts(pre_pts.size()), un_forw_pts(forw_pts.size());
  Eigen::Vector3d tmp_p;
  for (int i = 0; i < (int)pre_pts.size(); i++) {
    m_camera->liftProjective(Eigen::Vector2d(pre_pts[i].x, pre_pts[i].y),
                             tmp_p);
    un_pre_pts[i].x = FOCALLENGTH * tmp_p.x() / tmp_p.z() + COL / 2;
    un_pre_pts[i].y = FOCALLENGTH * tmp_p.y() / tmp_p.z() + ROW / 2;

    m_camera->liftProjective(Eigen::Vector2d(forw_pts[i].x, forw_pts[i].y),
                             tmp_p);
    un_forw_pts[i].x = FOCALLENGTH * tmp_p.x() / tmp_p.z() + COL / 2;
    un_forw_pts[i].y = FOCALLENGTH * tmp_p.y() / tmp_p.z() + ROW / 2;
  }

  for (int i = 0; i < (int)un_pre_pts.size(); i++) {
    // cv::circle(img1, un_pre_pts[i], 5, cv::Scalar(0, 0, 255));
    cv::circle(showImg, un_pre_pts[i], 4, srandColor);
    cv::circle(showImg,
               cv::Point2f(un_forw_pts[i].x + pre_img.cols, un_forw_pts[i].y),
               4, srandColor);
    cv::putText(showImg, to_string(ids[i]), un_pre_pts[i],
                cv::FONT_HERSHEY_COMPLEX_SMALL, 0.5, cv::Scalar(255));
    cv::putText(showImg, to_string(ids[i]),
                cv::Point2f(un_forw_pts[i].x + pre_img.cols, un_forw_pts[i].y),
                cv::FONT_HERSHEY_COMPLEX_SMALL, 0.5, cv::Scalar(255));
    cv::line(showImg, un_pre_pts[i],
             cv::Point2f(un_forw_pts[i].x + pre_img.cols, un_forw_pts[i].y),
             srandColor);
  }
  imshow("point match", showImg);
  // imshow("pre_points", img1);
  cvWaitKey(1);
}

void FeatureTracker::undistPoints(vector<cv::Point2f> &unpts) {
  Eigen::Vector3d tmp_p;
  unpts = cur_pts;
  for (int i = 0; i < (int)cur_pts.size(); i++) {
    m_camera->liftProjective(Eigen::Vector2d(cur_pts[i].x, cur_pts[i].y),
                             tmp_p);
    unpts[i].x = tmp_p.x() / tmp_p.z();
    unpts[i].y = tmp_p.y() / tmp_p.z();
  }
}

void FeatureTracker::rejectwithF() {
  // cout << "forw_pts size is: " << forw_pts.size() << endl;
  // cout << "id size is: " << ids.size() << endl;
  // ROS_ASSERT(forw_pts.size() == pre_pts.size());
  if (forw_pts.size() > 8) {
    vector<cv::Point2f> un_pre_pts(pre_pts.size()),
        un_forw_pts(forw_pts.size());
    Eigen::Vector3d temp_p;
    for (int i = 0; i < (int)pre_pts.size(); i++) {
      m_camera->liftProjective(Eigen::Vector2d(pre_pts[i].x, pre_pts[i].y),
                               temp_p);
      temp_p.x() = FOCALLENGTH * temp_p.x() / temp_p.z() + COL / 2.0;
      temp_p.y() = FOCALLENGTH * temp_p.y() / temp_p.z() + ROW / 2.0;
      un_pre_pts[i] = cv::Point2f(temp_p.x(), temp_p.y());
    }
    for (int i = 0; i < (int)forw_pts.size(); i++) {
      m_camera->liftProjective(Eigen::Vector2d(forw_pts[i].x, forw_pts[i].y),
                               temp_p);
      temp_p.x() = FOCALLENGTH * temp_p.x() / temp_p.z() + COL / 2.0;
      temp_p.y() = FOCALLENGTH * temp_p.y() / temp_p.z() + ROW / 2.0;
      un_forw_pts[i] = cv::Point2f(temp_p.x(), temp_p.y());
    }

    vector<uchar> status;
    cv::Mat F = findFundamentalMat(un_pre_pts, un_forw_pts, status);
    reduceVector(pre_pts, status);
    reduceVector(cur_pts, status);
    reduceVector(forw_pts, status);
    reduceVector(ids, status);
  }
}

void FeatureTracker::setMask(cv::Mat &mask) {
  mask = cv::Mat(forw_img.size(), CV_8UC1, cv::Scalar(255));

  for (int i = 0; i < (int)forw_pts.size(); i++) {
    if (mask.at<uchar>(forw_pts[i]) == 255)
      cv::circle(mask, forw_pts[i], MIN_DIS, -1);
  }
}

void FeatureTracker::addPoints(vector<cv::Point2f> &add_points) {
  for (auto &point : add_points) {
    forw_pts.push_back(point);
    ids.push_back(-1);
  }
}

void FeatureTracker::readImage(const cv::Mat &_img) {
  cv::Mat gray, img;
  if (_img.channels() == 3) {
    cv::cvtColor(_img, gray, COLOR_RGB2GRAY);
  } else {
    gray = _img;
  }

  // cv::cvtColor(_img, gray, COLOR_BGR2GRAY);
  cv::equalizeHist(gray, img);

  if (pre_img.empty()) {
    pre_img = cur_img = img;
  }
  forw_img = img;

  forw_pts.clear();

  if (cur_pts.size() > 0) {
    vector<uchar> status;
    vector<float> err;
    cv::calcOpticalFlowPyrLK(cur_img, forw_img, cur_pts, forw_pts, status, err);

    //判断追踪的点是否在图像内
    for (int i = 0; i < (int)status.size(); i++) {
      if (status[i] && !inBord(cur_pts[i])) status[i] = 0;
    }
    reduceVector(pre_pts, status);  // pre_pts和cur_pts是一样的
    reduceVector(cur_pts, status);
    reduceVector(forw_pts, status);
    reduceVector(ids, status);
  }

  if (PUBFRAME) {
    rejectwithF();
    //设置mask,添加特征点
    int cnt_points = MAX_CNT - static_cast<int>(forw_pts.size());
    if (cnt_points > 0) {
      cv::Mat mask;
      setMask(mask);
      vector<cv::Point2f> add_points;
      cv::goodFeaturesToTrack(forw_img, add_points, cnt_points, 0.1, MIN_DIS,
                              mask);
      addPoints(add_points);
    }
    pre_img = forw_img;
    cur_img = forw_img;
  }
  pre_pts = forw_pts;
  cur_pts = forw_pts;
}