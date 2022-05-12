#pragma once

#include <ceres/ceres.h>
#include <ceres/rotation.h>

#include <eigen3/Eigen/Dense>

#include "utility.h"

class PoseLocalParameterization : public ceres::LocalParameterization {
  virtual bool Plus(const double *x, const double *delta,
                    double *x_plus_delta) const;
  virtual bool ComputeJacobian(const double *x, double *jacobian) const;
  virtual int GlobalSize() const { return 7; };
  virtual int LocalSize() const { return 6; };
};

struct PoseCostFunctor {
 public:
  Eigen::Vector3d obs_i, obs_j;
  PoseCostFunctor(const Eigen::Vector3d &pts_i, const Eigen::Vector3d &pts_j)
      : obs_i(pts_i), obs_j(pts_j){};
  template <typename T>
  bool operator()(const T *const param_i, const T *const param_j,
                  const T *const param_dep, T *residual) const {
    T pts_i[3], pts_j[3], Rj_inv[4], Ri[4], pts_w[3];
    pts_i[0] = T(obs_i(0)) / param_dep[0];
    pts_i[1] = T(obs_i(1)) / param_dep[0];
    pts_i[2] = 1.0 / param_dep[0];
    // std::cout << "depth: " << param_dep[0] << std::endl;

    // Eigen::Quaterniond q(param_Rj[3], param_Rj[0], param_Rj[1], param_Rj[2]);
    // Eigen::Matrix3d Rj_inv = q.toRotationMatrix().transpose();

    Ri[0] = param_i[3];
    Ri[1] = param_i[4];
    Ri[2] = param_i[5];
    Ri[3] = param_i[6];

    Rj_inv[0] = -param_j[3];
    Rj_inv[1] = -param_j[4];
    Rj_inv[2] = -param_j[5];
    Rj_inv[3] = param_j[6];

    ceres::QuaternionRotatePoint(Ri, pts_i, pts_w);

    pts_i[0] = pts_i[0] + param_i[0] - param_j[0];
    pts_i[1] = pts_i[1] + param_i[1] - param_j[1];
    pts_i[2] = pts_i[2] + param_i[2] - param_j[2];

    ceres::QuaternionRotatePoint(Rj_inv, pts_w, pts_j);

    pts_j[0] /= pts_j[2];
    pts_j[1] /= pts_j[2];

    // const T u1 = pts_w2j[0] / pts_w2j[2];
    // const T v1 = pts_w2j[1] / pts_w2j[2];

    residual[0] = pts_j[0] - T(obs_j(0));
    residual[1] = pts_j[1] - T(obs_j(1));

    // std::cout << "residual" << residual[0] << ' ' << residual[1] <<
    // std::endl; std::cout << std::endl;

    return true;
  }

  static ceres::CostFunction *create(const Eigen::Vector3d &pts_i,
                                     const Eigen::Vector3d &pts_j) {
    return (new ceres::AutoDiffCostFunction<PoseCostFunctor, 2, 7, 7, 1>(
        new PoseCostFunctor(pts_i, pts_j)));
  }
};

struct TransCostFunctor {
  Eigen::Vector2d obs_i, obs_j;
  TransCostFunctor(Eigen::Vector2d _obs_i, Eigen::Vector2d _obs_j)
      : obs_i(_obs_i), obs_j(_obs_j){};
  template <typename T>
  bool operator()(const T *const param_i, const T *const param_j,
                  T *residual) const {
    T Ri[4], Rj_inv[4], Pi[3], Pw[3], Pj[3];

    Pi[0] = T(obs_i(0));
    Pi[1] = T(obs_i(1));
    Pi[2] = T(1.0);

    Ri[0] = param_i[3];
    Ri[1] = param_i[4];
    Ri[2] = param_i[5];
    Ri[3] = param_i[6];

    ceres::QuaternionRotatePoint(Ri, Pi, Pw);
    Pw[0] = Pw[0] + param_i[0] - param_j[0];
    Pw[1] = Pw[1] + param_i[1] - param_j[1];
    Pw[2] = Pw[2] + param_i[2] - param_j[2];

    Rj_inv[0] = -param_j[3];
    Rj_inv[1] = -param_j[4];
    Rj_inv[2] = -param_j[5];
    Rj_inv[3] = param_j[6];

    ceres::QuaternionRotatePoint(Rj_inv, Pw, Pj);

    Pj[0] /= Pj[2];
    Pj[1] /= Pj[2];

    residual[0] = Pj[0] - T(obs_j(0));
    residual[1] = Pj[1] - T(obs_j(1));

    return true;
  }
};