#include "feature_manager.h"

int FeaturePerId::endframe() {
  return start_frame + feature_per_frame.size() - 1;
}

double FeatureManager::compemsateParallax(const FeaturePerId &it_pre_id,
                                          int frame_count) {
  // last obs and first obs
  const FeaturePerFrame frame_thrid =
      it_pre_id.feature_per_frame[frame_count - 2 - it_pre_id.start_frame];
  const FeaturePerFrame frame_second =
      it_pre_id.feature_per_frame[frame_count - 1 - it_pre_id.start_frame];

  Eigen::Vector2d p_thrid = frame_thrid.obs;
  Eigen::Vector2d p_second = frame_second.obs;

  // std::cout << "obs_th: " << p_thrid << std::endl;
  // std::cout << "obs_se: " << p_second << std::endl;
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
  int feature_id;
  Eigen::Vector2d obs;
  int feature_track_num = 0;
  double parallax_sum = 0;
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
  // std::cout << "parallax: " << parallax_sum / parallax_num << std::endl;
  return (parallax_sum / parallax_num) >= MIN_PARALLAX;
}

void FeatureManager::BASolvetrans(Eigen::Matrix3d Rwc[],
                                  Eigen::Vector3d twc[]) {
  double para_pose[WINDOW_SIZE + 1][SIZE_POSE];
  for (int i = 0; i <= WINDOW_SIZE; i++) {
    para_pose[i][0] = twc[i](0);
    para_pose[i][1] = twc[i](1);
    para_pose[i][2] = twc[i](2);

    Eigen::Quaterniond q(Rwc[i]);
    para_pose[i][3] = q.x();
    para_pose[i][4] = q.y();
    para_pose[i][5] = q.z();
    para_pose[i][6] = q.w();
  }

  ceres::Problem problem;
  ceres::LossFunction *loss_function = new ceres::CauchyLoss(1.0);

  ceres::LocalParameterization *local_parameterization =
      new PoseLocalParameterization();
  for (int i = 0; i <= WINDOW_SIZE; i++)
    problem.AddParameterBlock(para_pose[i], SIZE_POSE, local_parameterization);

  for (auto &it_per_id : feature) {
    if (it_per_id.is_triangulate == false) continue;
    Eigen::Vector2d obs = it_per_id.feature_per_frame[0].obs;
    int cam_i = it_per_id.start_frame, cam_j = cam_i - 1;
    ROS_ASSERT(it_per_id.estimate_depth > 0.1);
    for (auto &it_per_frame : it_per_id.feature_per_frame) {
      cam_j++;
      if (cam_j == cam_i) continue;
      Eigen::Vector2d obs_j = it_per_frame.obs;
      ceres::CostFunction *f =
          new ceres::AutoDiffCostFunction<TransCostFunctor, 2, 7, 7>(
              new TransCostFunctor(obs, obs_j));
      problem.AddResidualBlock(f, loss_function, para_pose[cam_i],
                               para_pose[cam_j]);
    }
  }

  ceres::Solver::Options options;
  options.linear_solver_type = ceres::DENSE_SCHUR;
  // options.minimizer_progress_to_stdout = true;
  ceres::Solver::Summary summary;
  ceres::Solve(options, &problem, &summary);
  // std::cout << summary.FullReport() << '\n';
  for (int i = 0; i <= WINDOW_SIZE; i++) {
    twc[i](0) = para_pose[i][0];
    twc[i](1) = para_pose[i][1];
    twc[i](2) = para_pose[i][2];

    Eigen::Quaterniond q(para_pose[i][3], para_pose[i][4], para_pose[i][5],
                         para_pose[i][6]);
    Rwc[i] = q.normalized().toRotationMatrix();
  }
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
    cv::Mat E = cv::findEssentialMat(points[0], points[1], m_K);
    // cv::Mat F = cv::findFundamentalMat(points[0], points[1]);

    // std::cout << "E: " << E << std::endl;
    cv::Mat R, t;
    // m_K.convertTo(K, F.type());
    // E = K.t() * F * K;
    // std::cout << "E1: " << E << std::endl;
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

    for (int i = 0; i < (int)points_id.size(); i++) {
      int id = points_id[i];
      auto it =
          find_if(feature.begin(), feature.end(),
                  [id](const FeaturePerId &it) { return it.feature_id == id; });
      if (it == feature.end()) {
        continue;
      } else {
        cv::Mat point3d = pts_4d.col(i);
        point3d /= point3d.at<float>(3, 0);
        if (point3d.at<float>(2, 0) > 0.1) {
          it->estimate_depth = point3d.at<float>(2, 0);
          it->is_triangulate = true;
        } else {
          it->estimate_depth = 0;
        }
      }
    }

    //归一化

    // triangulate
    Rwc[0] = Eigen::Matrix3d::Identity();
    twc[0] = Eigen::Vector3d::Zero();
    Rwc[frame_count] = Rwc[0] * R10.transpose();
    twc[frame_count] = Rwc[frame_count] * (-t10);

    for (int i = 1; i < WINDOW_SIZE; i++) {
      Rwc[i] = Eigen::Matrix3d::Identity();
      twc[i] = (i / 10.) * twc[frame_count];
    }
    BASolvetrans(Rwc, twc);

    // pnp
    /*
    1.找出3d点对应在各帧上的观测
    2.使用pnp求解
    */
    // ROS_WARN("START PNP");
    // vector<cv::Point3f> pt_in_world;
    // vector<vector<cv::Point2f>> obs_in_frame(
    //     11);  // frame size 0-10 and pt size

    // for (auto &it_per_id : feature) {
    //   if (it_per_id.is_triangulate == false) continue;
    //   Eigen::Vector2d obs = it_per_id.feature_per_frame[0].obs;
    //   ROS_ASSERT(it_per_id.estimate_depth > 0.1);
    //   double depth = it_per_id.estimate_depth;
    //   obs *= depth;
    //   pt_in_world.push_back(cv::Point3f(obs.x(), obs.y(), depth));
    //   int frame_id = -1;
    //   for (auto &it_per_frame : it_per_id.feature_per_frame) {
    //     frame_id++;
    //     obs_in_frame[frame_id].push_back(
    //         cv::Point2f(it_per_frame.obs.x(), it_per_frame.obs.y()));
    //   }
    // }  //验证过是pt3d与pt2d是对应上的
    // std::cout << "obs size: " << obs_in_frame[5].size() << std::endl;
    // std::cout << "pt size: " << pt_in_world.size() << std::endl;
    // auto obs_in_cam = obs_in_frame;
    // toCamera(obs_in_frame);

    // // cv::Mat K = (cv::Mat_<double>(3, 3) << 1, 0, 0, 0, 1, 0, 0, 0, 1);
    // // cv::Mat D;
    // ROS_ASSERT(pt_in_world.size() >= 6);
    // for (int i = 1; i < (int)obs_in_frame.size() - 1; i++) {
    //   // BA

    //   // pnp
    //   // cv::Mat Ri, ti0, Ri0;
    //   // // cv::eigen2cv(Rwc[i - 1], Ri);
    //   // // cv::eigen2cv(twc[i - 1], ti0);
    //   // if (!cv::solvePnP(pt_in_world, obs_in_frame[i], m_K, distCoeffs, Ri,
    //   //                   ti0)) {
    //   //   ROS_DEBUG("solve pnp fail");
    //   //   return;
    //   // }
    //   // Eigen::Matrix3d tmp_R;
    //   // Eigen::Vector3d tmp_t;
    //   // ROS_WARN("Succeed pnp");
    //   // cv::Rodrigues(Ri, Ri0);
    //   // cv::cv2eigen(Ri0, tmp_R);
    //   // Rwc[i] = tmp_R.transpose();
    //   // cv::cv2eigen(ti0, tmp_t);
    //   // twc[i] = Rwc[i] * (-tmp_t);
    //   // std::cout << "Rwc " << i << ": " << Rwc[i] << std::endl;
    //   // std::cout << "twc " << i << ": " << twc[i] << std::endl;
    //   // std::cout << std::endl;

    //   // DLT
    //   // Eigen::MatrixXd H(2 * pt_in_world.size(), 12);
    //   // int svd_idx = 0;
    //   // for (int j = 0; j < pt_in_world.size(); j++) {
    //   //   Eigen::Matrix<double, 4, 1> pt1, pt2, pt3, pt4;
    //   //   pt1 << pt_in_world[j].x, pt_in_world[j].y, pt_in_world[j].z, 1;
    //   //   pt2.setZero();
    //   //   pt3 = -obs_in_frame[i][j].x * pt1;
    //   //   pt4 = -obs_in_frame[i][j].y * pt1;
    //   //   H.row(svd_idx++) << pt1.transpose(), pt2.transpose(),
    //   //   pt3.transpose(); H.row(svd_idx++) << pt2.transpose(),
    //   //   pt1.transpose(), pt4.transpose();
    //   // }

    //   // ROS_ASSERT(H.rows() == svd_idx);
    //   // Eigen::JacobiSVD<Eigen::MatrixXd> svd_A(H, Eigen::ComputeThinV);
    //   // Eigen::MatrixXd V_A = svd_A.matrixV();
    //   // Eigen::MatrixXd Sigma_A = svd_A.singularValues();

    //   // double a1 = V_A(0, 11);
    //   // double a2 = V_A(1, 11);
    //   // double a3 = V_A(2, 11);
    //   // double a4 = V_A(3, 11);
    //   // double a5 = V_A(4, 11);
    //   // double a6 = V_A(5, 11);
    //   // double a7 = V_A(6, 11);
    //   // double a8 = V_A(7, 11);
    //   // double a9 = V_A(8, 11);
    //   // double a10 = V_A(9, 11);
    //   // double a11 = V_A(10, 11);
    //   // double a12 = V_A(11, 11);
    //   // Eigen::Matrix3d R_bar;
    //   // R_bar << a1, a2, a3, a5, a6, a7, a9, a10, a11;
    //   // Eigen::JacobiSVD<Eigen::MatrixXd> svd_R(
    //   //     R_bar, Eigen::ComputeFullU | Eigen::ComputeFullV);
    //   // Eigen::Matrix3d U_R = svd_R.matrixU();
    //   // Eigen::Matrix3d V_R = svd_R.matrixV();
    //   // Eigen::Vector3d V_Sigma = svd_R.singularValues();

    //   // Eigen::Matrix3d R = U_R * V_R.transpose();
    //   // double beta = 1.0 / ((V_Sigma(0) + V_Sigma(1) + V_Sigma(2)) / 3.0);

    //   // // Step 4. Compute t
    //   // Eigen::Vector3d t_bar(a4, a8, a12);
    //   // Eigen::Vector3d t = beta * t_bar;

    //   // int num_positive = 0;
    //   // int num_negative = 0;
    //   // for (int i = 0; i < pt_in_world.size(); i++) {
    //   //   const double &x = pt_in_world[i].x;
    //   //   const double &y = pt_in_world[i].y;
    //   //   const double &z = pt_in_world[i].z;

    //   //   double lambda = beta * (x * a9 + y * a10 + z * a11 + a12);
    //   //   if (lambda >= 0) {
    //   //     num_positive++;
    //   //   } else {
    //   //     num_negative++;
    //   //   }
    //   // }

    //   // if (num_positive < num_negative) {
    //   //   R = -R;
    //   //   t = -t;
    //   // }

    //   // Rwc[i] = R.transpose();
    //   // twc[i] = Rwc[i] * (-t);
    // }

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
    // int obs_size = it_per_id.feature_per_frame.size();
    int feature_id = it_per_id.feature_id;
    if (it_per_id.start_frame <= frame_i && it_per_id.endframe() >= frame_j) {
      // std::cout << "obs size: " << obs_size << std::endl;
      obs_i = it_per_id.feature_per_frame[frame_i].obs;
      obs_j = it_per_id.feature_per_frame[frame_j].obs;
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
  for (int i = 0; i < (int)points.size(); i++) {
    for (int j = 0; j < (int)points[i].size(); j++) {
      points[i][j].x = FOCALLENGTH * points[i][j].x + COL / 2;
      points[i][j].y = FOCALLENGTH * points[i][j].y + ROW / 2;
    }
  }
}

void FeatureManager::toCamera(vector<vector<cv::Point2f>> &points) {
  for (int i = 0; i < (int)points.size(); i++) {
    for (int j = 0; j < (int)points[i].size(); j++) {
      points[i][j].x = FOCALLENGTH * points[i][j].x;
      points[i][j].y = FOCALLENGTH * points[i][j].y;
    }
  }
}

void FeatureManager::slidWindowOld(Eigen::Matrix3d back_R,
                                   Eigen::Vector3d back_t,
                                   Eigen::Matrix3d Rwc[],
                                   Eigen::Vector3d twc[]) {
  // shift depth
  Eigen::Matrix3d New_R = Rwc[0].transpose() * back_R;              // R0to1
  Eigen::Vector3d New_t = Rwc[0].transpose() * (-twc[0] + back_t);  // t0to1
  // int slidfeature = 0;
  for (auto it = feature.begin(), it_next = feature.begin();
       it != feature.end(); it = it_next) {
    it_next++;
    if (it->start_frame != 0)
      it->start_frame--;
    else {
      Eigen::Vector2d obs_in_back = it->feature_per_frame[0].obs;
      it->feature_per_frame.erase(it->feature_per_frame.begin());
      if (it->feature_per_frame.size() == 0) {
        feature.erase(it);
        // slidfeature++;
        continue;
      } else {
        double depth = it->estimate_depth;
        Eigen::Vector3d pt_in_back(obs_in_back(0) * depth,
                                   obs_in_back(1) * depth, depth);
        Eigen::Vector3d pt_in_newcam;
        pt_in_newcam = New_R * pt_in_back + New_t;
        if (pt_in_newcam(2) > 0)
          it->estimate_depth = pt_in_newcam(2);
        else {
          it->estimate_depth = 0;
          it->is_triangulate = false;
        }
        // it->is_triangulate = false;
      }
    }
  }
  // std::cout << "erase feature num due to obs num: " << slidfeature <<
  // std::endl;
}

void FeatureManager::slidWindowNew() {
  for (auto it = feature.begin(), it_next = feature.begin();
       it != feature.end(); it = it_next) {
    it_next++;
    if (it->start_frame == WINDOW_SIZE)
      it->start_frame--;
    else if (it->start_frame == WINDOW_SIZE - 1 && it->is_triangulate)
      it->is_triangulate = false;
    else {
      int j = WINDOW_SIZE - 1 - it->start_frame;
      if (it->endframe() < WINDOW_SIZE - 1) continue;
      it->feature_per_frame.erase(it->feature_per_frame.begin() + j);
      if (it->feature_per_frame.size() == 0) feature.erase(it);
    }
  }
}

void FeatureManager::triangulatePoint(Eigen::Matrix3d Rwc[],
                                      Eigen::Vector3d twc[]) {
  int triangulatepoint = 0;
  for (auto &it_per_id : feature) {
    if (it_per_id.is_triangulate && it_per_id.estimate_depth > 0.1) continue;
    int obs_size = it_per_id.feature_per_frame.size();
    if (obs_size < MIN_OBS_SIZE || it_per_id.start_frame >= WINDOW_SIZE - 2)
      continue;
    int cam_i = it_per_id.start_frame, cam_j = cam_i - 1;
    Eigen::Vector2d pti = it_per_id.feature_per_frame[0].obs;
    Eigen::MatrixXd svd_A(2 * obs_size, 4);
    int svd_idx = 0;
    for (auto &it_per_frame : it_per_id.feature_per_frame) {
      cam_j++;
      Eigen::Vector2d ptj = it_per_frame.obs;
      Eigen::Matrix<double, 3, 4> T;
      Eigen::Matrix3d Rji = Rwc[cam_j].transpose() * Rwc[cam_i];
      Eigen::Vector3d tji = Rwc[cam_j].transpose() * (twc[cam_i] - twc[cam_j]);
      T.leftCols<3>() = Rji;
      T.rightCols<1>() = tji;
      svd_A.row(svd_idx++) = ptj(1) * T.row(2) - T.row(1);
      svd_A.row(svd_idx++) = -ptj(0) * T.row(2) + T.row(0);
    }

    ROS_ASSERT(svd_idx == svd_A.rows());
    Eigen::Vector4d svd_V =
        Eigen::JacobiSVD<Eigen::MatrixXd>(svd_A, Eigen::ComputeThinV)
            .matrixV()
            .rightCols<1>();

    double depth = svd_V[2] / svd_V[3];
    // visual
    // if (it_per_id.is_triangulate) {
    //   std::cout << "point id " << it_per_id.feature_id << std::endl;
    //   std::cout << "depth " << it_per_id.estimate_depth << std::endl;
    //   std::cout << "estimate depth" << depth << std::endl;
    //   std::cout << std::endl;
    // }
    if (depth > 0) {
      it_per_id.estimate_depth = depth;
      it_per_id.is_triangulate = true;
      triangulatepoint++;

      // std::cout << "point id " << it_per_id.feature_id << std::endl;
      // std::cout << "depth " << it_per_id.estimate_depth << std::endl;
      // std::cout << std::endl;
    }
  }
  std::cout << "triangulatpoint: " << triangulatepoint << std::endl;
}

Eigen::VectorXd FeatureManager::getFeatureDepth() {
  Eigen::VectorXd dep_xd(getFeatureCount());
  int dep_idx = 0;
  for (auto &it_per_id : feature) {
    if (it_per_id.is_triangulate) {
      // ROS_ASSERT(it_per_id.estimate_depth > 0.1);
      dep_xd(dep_idx++) = 1.0 / it_per_id.estimate_depth;
    }
  }
  return dep_xd;
}

int FeatureManager::getFeatureCount() {
  int count = 0;
  for (auto &it_per_id : feature) {
    if (it_per_id.is_triangulate) {
      // ROS_ASSERT(it_per_id.estimate_depth > 0.1);
      count++;
    }
  }
  return count;
}

void FeatureManager::setDepth(const Eigen::VectorXd &dep) {
  int feature_idx = -1;
  // int failfeature_num = 0;
  for (auto &it_per_id : feature) {
    if (it_per_id.is_triangulate) {
      it_per_id.estimate_depth = 1.0 / dep(++feature_idx);
      if (it_per_id.estimate_depth <= 0) {
        // std::cout << "dep: " << it_per_id.estimate_depth << std::endl;
        it_per_id.is_triangulate = false;
        it_per_id.estimate_depth = 0;
        it_per_id.solve_flag = 2;
        // failfeature_num++;
      }
    }
  }
  // std::cout << "fail feature numis: " << failfeature_num << std::endl;
}

void FeatureManager::checkTransformation(const Eigen::Matrix3d Rwc[],
                                         const Eigen::Vector3d twc[]) {
  for (auto &it_per_id : feature) {
    if (it_per_id.is_triangulate) {
      ROS_ASSERT(it_per_id.estimate_depth > 0.1);
      Eigen::Vector3d pts_i;
      pts_i << it_per_id.feature_per_frame[0].obs, 1;
      pts_i *= it_per_id.estimate_depth;

      // int frame_id = -1;
      int cam_i = it_per_id.start_frame, cam_j = cam_i - 1;
      for (auto &it_per_frame : it_per_id.feature_per_frame) {
        cam_j++;
        if (cam_j == cam_i) continue;
        Eigen::Vector3d pts_i2j =
            Rwc[cam_j].transpose() *
            (Rwc[cam_i] * pts_i + twc[cam_i] - twc[cam_j]);
        pts_i2j /= pts_i2j(2);

        std::cout << "trans: " << pts_i2j.head(2) << std::endl;
        std::cout << "obs: " << it_per_frame.obs << std::endl;
        std::cout << std::endl;
      }
    }
  }
}

void FeatureManager::removeFailures() {
  int feature_size = 0;
  for (auto it = feature.begin(), it_next = feature.begin();
       it != feature.end(); it = it_next) {
    it_next++;
    if (it->solve_flag == 2) feature.erase(it);
    if (it->is_triangulate) feature_size++;
  }
  std::cout << "feature size: " << feature_size << std::endl;
}