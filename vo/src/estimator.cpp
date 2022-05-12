#include "estimator.h"

#include "feature_manager.h"

void Estimator::readIntrinsicParameter(std::string &calib_file) {
  ROS_INFO("read parameter form &s", calib_file.c_str());
  m_camera = camodocal::CameraFactory::instance()->generateCameraFromYamlFile(
      calib_file);
  m_K = m_camera->initUndistortRectifyMap(un_map1, un_map2);
  float fx = m_K.at<float>(0, 0);
  float fy = m_K.at<float>(1, 1);
  float cx = m_K.at<float>(0, 2);
  float cy = m_K.at<float>(1, 2);
  std::cout << "K" << fx << '\n' << fy << '\n' << cx << '\n' << cy << std::endl;
  std::cout << "distCoeffs: " << distCoeffs << std::endl;
}

void Estimator::ProcessImage(const vector<pair<int, cv::Point2f>> &points) {
  /*
  1.根据视差确定关键帧
  2.初始化
  3.三角化
  4.BA
  */
  if (f_manager.checkParallax(points, frame_count))
    marginalization_flag = MARGIN_OLD;
  else
    marginalization_flag = MARGIN_NEW;

  //求解F和H
  if (solver_flag == INITIAL) {
    if (frame_count == WINDOW_SIZE) {
      bool result;
      f_manager.Initialization(frame_count, result, m_K, Rwc, twc);
      if (!result) {
        slidWindow();
      }  // slidwindow
      else {
        // ROS_WARN("succeed initia");
        solver_flag = NON_LINEAR;
        solveOdometry();
        // f_manager.checkTransformation(Rwc, twc);
        // checkQ();
        slidWindow();
        f_manager.removeFailures();
      }
    } else {
      frame_count++;
      // std::cout << "frame count " << frame_count << std::endl;
    }
  } else {
    ROS_INFO("NON_LINEAR");
    // solveOdometry();
    // slidWindow();
    // f_manager.removeFailures();
  }
}

void Estimator::slidWindow() {
  if (marginalization_flag == MARGIN_OLD) {
    // getchar();
    Eigen::Matrix3d back_R = Rwc[0];
    Eigen::Vector3d back_t = twc[0];
    for (int i = 0; i < WINDOW_SIZE; i++) {
      Rwc[i].swap(Rwc[i + 1]);
      twc[i].swap(twc[i + 1]);
    }
    // Eigen::Matrix3d delta_R =
    //     Rwc[WINDOW_SIZE - 1].transpose() * Rwc[WINDOW_SIZE - 2];
    Rwc[WINDOW_SIZE] = Rwc[WINDOW_SIZE - 1];
    twc[WINDOW_SIZE] = twc[WINDOW_SIZE - 1];
    ROS_WARN("MARGIN OLD");
    f_manager.slidWindowOld(back_R, back_t, Rwc, twc);
  } else {
    Rwc[frame_count - 1] = Rwc[frame_count];
    twc[frame_count - 1] = twc[frame_count];
    // ROS_WARN("start slidnew");
    ROS_WARN("MARGIN NEW");
    f_manager.slidWindowNew();
  }
  // ROS_WARN("end sliding");
}

void Estimator::solveOdometry() {
  f_manager.triangulatePoint(Rwc, twc);
  optimization();
  // 1.triangulate point
  // 2. BA
}

void Estimator::optimization() {
  // ceres 优化
  //边缘化
  ceres::Problem problem;
  ceres::LossFunction *loss_function = new ceres::CauchyLoss(1.0);
  vector2double();

  std::cout << "trans before optimization: " << '\n';
  for (int i = 0; i <= WINDOW_SIZE; i++) {
    for (int j = 0; j < 7; j++) {
      std::cout << para_pose[i][j] << '\n';
      std::cout << std::endl;
    }
  }
  ceres::LocalParameterization *local_parameterization =
      new PoseLocalParameterization();
  for (int i = 0; i <= WINDOW_SIZE; i++) {
    problem.AddParameterBlock(para_pose[i], SIZE_POSE, local_parameterization);
  }

  for (auto &it_per_id : f_manager.feature) {
    if (it_per_id.is_triangulate) {
      int cam_i = it_per_id.start_frame, cam_j = cam_i - 1;
      Eigen::Vector3d obs_i;
      obs_i << it_per_id.feature_per_frame[0].obs, 1;
      int feature_idx = -1;
      for (auto &it_per_frame : it_per_id.feature_per_frame) {
        cam_j++;
        if (cam_i == cam_j) continue;
        Eigen::Vector3d obs_j;
        obs_j << it_per_frame.obs, 1;
        // checkResidual(cam_i, cam_j, obs_i, obs_j);
        problem.AddResidualBlock(
            new ceres::AutoDiffCostFunction<PoseCostFunctor, 2, 7, 7, 1>(
                new PoseCostFunctor(obs_i, obs_j)),
            loss_function, para_pose[cam_i], para_pose[cam_j],
            para_feature[++feature_idx]);
      }
    }
  }

  ceres::Solver::Options options;
  options.linear_solver_type = ceres::DENSE_SCHUR;
  // options.minimizer_progress_to_stdout = true;
  // options.max_num_iterations = 8;
  ceres::Solver::Summary summary;
  ceres::Solve(options, &problem, &summary);
  // std::cout << summary.FullReport() << '\n';
  double2vector();
  std::cout << "trans after optimization: " << '\n';
  for (int i = 0; i <= WINDOW_SIZE; i++) {
    for (int j = 0; j < 7; j++) {
      std::cout << para_pose[i][j] << '\n';
      std::cout << std::endl;
    }
  }
}

void Estimator::vector2double() {
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

  int feature_count = f_manager.getFeatureCount();
  Eigen::VectorXd dep_xd = f_manager.getFeatureDepth();
  for (int i = 0; i < feature_count; i++) {
    para_feature[i][0] = dep_xd(i);
  }
}

void Estimator::double2vector() {
  for (int i = 0; i <= WINDOW_SIZE; i++) {
    twc[i](0) = para_pose[i][0];
    twc[i](1) = para_pose[i][1];
    twc[i](2) = para_pose[i][2];

    Eigen::Quaterniond q;
    q.x() = para_pose[i][3];
    q.y() = para_pose[i][4];
    q.z() = para_pose[i][5];
    q.w() = para_pose[i][6];
    Rwc[i] = q.normalized().toRotationMatrix();

    Eigen::VectorXd dep = f_manager.getFeatureDepth();
    for (int i = 0; i < f_manager.getFeatureCount(); i++) {
      dep(i) = para_feature[i][0];
    }
    f_manager.setDepth(dep);
  }
}

void Estimator::checkQ() {
  for (int i = 0; i < WINDOW_SIZE + 1; i++) {
    Eigen::Quaterniond q(Rwc[i]);
    q.x() = -q.x();
    q.y() = -q.y();
    q.z() = -q.z();

    Eigen::Matrix3d tmp_R = q.normalized().toRotationMatrix();
    tmp_R *= Rwc[i];
    std::cout << "check Q: " << tmp_R << std::endl;
  }
}

void Estimator::checkResidual(int cam_i, int cam_j, Eigen::Vector3d obs_i,
                              Eigen::Vector3d obs_j) {
  obs_i =
      Rwc[cam_j].transpose() * (Rwc[cam_i] * obs_i + twc[cam_i] - twc[cam_j]);
  Eigen::Vector2d residual = obs_i.head(2) - obs_j.head(2);
  std::cout << "residual in cam " << cam_j << ": " << residual << std::endl;
}
