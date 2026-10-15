#pragma once

#include <Eigen/Dense>

namespace gsolver {

  using Isometry2d = Eigen::Isometry2d;

  //! rotation matrix in 2d
  inline Eigen::Matrix2d rotation2d(const double& angle) {
    Eigen::Matrix2d R;
    R << cos(angle), -sin(angle), sin(angle), cos(angle);
    return R;
  }

  //! @brief tangent vector is [x, y, theta], same translation-first convention as expMapSE3
  inline Isometry2d expMapSE2(const Eigen::Vector3d& x) {
    Isometry2d T    = Isometry2d::Identity();
    T.translation() = x.head(2);
    T.linear()      = rotation2d(x[2]);
    return T;
  }

  inline Eigen::Vector3d logMapSE2(const Isometry2d& T) {
    Eigen::Vector3d x;
    x.head(2) = T.translation();
    x[2]      = atan2(T.linear()(1, 0), T.linear()(0, 0));
    return x;
  }

} // namespace gsolver
