#include "projection_factor.h"

Eigen::Matrix2d ProjectionFactor::sqrt_info =
    FOCALLENGTH / 1.5 * Eigen::Matrix2d::Identity();
double ProjectionFactor::sum_t;

ProjectionFactor::ProjectionFactor(const Eigen::Vector3d &_pts_i,
                                   const Eigen::Vector3d &_pts_j)
    : pts_i(_pts_i), pts_j(_pts_j) {
#ifdef UNIT_SPHERE_ERROR
  Eigen::Vector3d b1, b2;
  Eigen::Vector3d a = pts_j.normalized();
  Eigen::Vector3d tmp(0, 0, 1);
  if (a == tmp) tmp << 1, 0, 0;
  b1 = (tmp - a * (a.transpose() * tmp)).normalized();
  b2 = a.cross(b1);
  tangent_base.block<1, 3>(0, 0) = b1.transpose();
  tangent_base.block<1, 3>(1, 0) = b2.transpose();
#endif
};

bool ProjectionFactor::Evaluate(double const *const *parameters,
                                double *residuals, double **jacobians) const {
  Eigen::Vector3d Pi(parameters[0][0], parameters[0][1], parameters[0][2]);
  Eigen::Quaterniond Qi(parameters[0][6], parameters[0][3], parameters[0][4],
                        parameters[0][5]);

  Eigen::Vector3d Pj(parameters[1][0], parameters[1][1], parameters[1][2]);
  Eigen::Quaterniond Qj(parameters[1][6], parameters[1][3], parameters[1][4],
                        parameters[1][5]);

  double inv_dep_i = parameters[2][0];

  Eigen::Vector3d pts_camera_i = pts_i / inv_dep_i;
  // Eigen::Vector3d pts_imu_i = qic * pts_camera_i + tic;
  Eigen::Vector3d pts_w = Qi * pts_camera_i + Pi;
  Eigen::Vector3d pts_camera_j = Qj.inverse() * (pts_w - Pj);
  // Eigen::Vector3d pts_camera_j = qic.inverse() * (pts_imu_j - tic);
  Eigen::Map<Eigen::Vector2d> residual(residuals);

#ifdef UNIT_SPHERE_ERROR
  residual = tangent_base * (pts_camera_j.normalized() - pts_j.normalized());
#else
  double dep_j = pts_camera_j.z();
  residual = (pts_camera_j / dep_j).head<2>() - pts_j.head<2>();
#endif

  residual = sqrt_info * residual;

  if (jacobians) {
    Eigen::Matrix3d Ri = Qi.toRotationMatrix();
    Eigen::Matrix3d Rj = Qj.toRotationMatrix();
    // Eigen::Matrix3d ric = qic.toRotationMatrix();
    Eigen::Matrix<double, 2, 3> reduce(2, 3);
#ifdef UNIT_SPHERE_ERROR
    double norm = pts_camera_j.norm();
    Eigen::Matrix3d norm_jaco;
    double x1, x2, x3;
    x1 = pts_camera_j(0);
    x2 = pts_camera_j(1);
    x3 = pts_camera_j(2);
    norm_jaco << 1.0 / norm - x1 * x1 / pow(norm, 3), -x1 * x2 / pow(norm, 3),
        -x1 * x3 / pow(norm, 3), -x1 * x2 / pow(norm, 3),
        1.0 / norm - x2 * x2 / pow(norm, 3), -x2 * x3 / pow(norm, 3),
        -x1 * x3 / pow(norm, 3), -x2 * x3 / pow(norm, 3),
        1.0 / norm - x3 * x3 / pow(norm, 3);
    reduce = tangent_base * norm_jaco;
#else
    reduce << 1. / dep_j, 0, -pts_camera_j(0) / (dep_j * dep_j), 0, 1. / dep_j,
        -pts_camera_j(1) / (dep_j * dep_j);
#endif
    reduce = sqrt_info * reduce;

    if (jacobians[0]) {
      Eigen::Map<Eigen::Matrix<double, 2, 7, Eigen::RowMajor>> jacobian_pose_i(
          jacobians[0]);

      Eigen::Matrix<double, 3, 6> jaco_i;
      jaco_i.leftCols<3>() = Rj.transpose();
      jaco_i.rightCols<3>() =
          -Rj.transpose() * Ri * Utility::skewSymmetric(pts_camera_i);

      jacobian_pose_i.leftCols<6>() = reduce * jaco_i;
      jacobian_pose_i.rightCols<1>().setZero();
    }

    if (jacobians[1]) {
      Eigen::Map<Eigen::Matrix<double, 2, 7, Eigen::RowMajor>> jacobian_pose_j(
          jacobians[1]);

      Eigen::Matrix<double, 3, 6> jaco_j;
      jaco_j.leftCols<3>() = -Rj.transpose();
      jaco_j.rightCols<3>() = Utility::skewSymmetric(pts_camera_j);

      jacobian_pose_j.leftCols<6>() = reduce * jaco_j;
      jacobian_pose_j.rightCols<1>().setZero();
    }
    if (jacobians[2]) {
      Eigen::Map<Eigen::Vector2d> jacobian_feature(jacobians[2]);
#if 1
      jacobian_feature =
          reduce * Rj.transpose() * Ri * pts_i * -1.0 / (inv_dep_i * inv_dep_i);
#else
      jacobian_feature =
          reduce * ric.transpose() * Rj.transpose() * Ri * ric * pts_i;
#endif
    }
  }

  return true;
}