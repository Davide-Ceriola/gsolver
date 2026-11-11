#pragma once

// Two independent, global, opt-in choices for the pose-graph types (SE2Pose, SE3Pose and their
// factors SE2Prior, SE2Odometry, SE3PosePrior, SE3PosePose). Both default off, which is gsolver's
// original behaviour.
//
//  true_exponential: the chart used for retractions (x exp(tau)) and factor errors (log).
//    off: translation and rotation separately, R^2 x SO(2) / R^3 x SO(3):
//         exp(t, w) = [Exp(w) | t]                     (expMapSE2 / expMapSE3)
//    on:  the group exponential of SE(2) / SE(3):
//         exp(rho, w) = [Exp(w) | V(w) rho]            (expSE2True / expSE3True)
//    Tangent order is the same (translation part, then rotation). Factor Jacobians in this mode
//    are computed by central differences of the error (exact up to O(h^2)).
//
//  exact_transport: moving a precision matrix between tangent spaces.
//    A Gaussian message is a point plus a precision in the tangent space at some point. off: a
//    precision is reused unchanged in another tangent space. on: it is transported with the
//    Jacobian J of the change of chart, Lambda' = J^-T Lambda J^-1 (J by central differences).

#include <cmath>

#include <Eigen/Dense>

#include <gsolver/maths/geometry2d.h>
#include <gsolver/maths/geometry3d.h>

namespace gsolver {

  struct LieOptions {
    static inline bool true_exponential = false;
    static inline bool exact_transport  = false;
  };

  // ---- SE(2) group exponential and logarithm, tangent (rho_x, rho_y, theta) ----
  inline Eigen::Matrix2d se2V(double th) {
    double a, b; // V = [[a, -b], [b, a]]
    if (std::abs(th) < 1e-8) {
      a = 1.0 - th * th / 6.0;
      b = th / 2.0;
    } else {
      a = std::sin(th) / th;
      b = (1.0 - std::cos(th)) / th;
    }
    Eigen::Matrix2d V;
    V << a, -b, b, a;
    return V;
  }

  inline Isometry2d expSE2True(const Eigen::Vector3d& x) {
    Isometry2d T    = Isometry2d::Identity();
    T.linear()      = rotation2d(x[2]);
    T.translation() = se2V(x[2]) * x.head<2>();
    return T;
  }

  inline Eigen::Vector3d logSE2True(const Isometry2d& T) {
    const double th = std::atan2(T.linear()(1, 0), T.linear()(0, 0));
    Eigen::Vector3d x;
    x.head<2>() = se2V(th).inverse() * T.translation();
    x[2]        = th;
    return x;
  }

  // ---- SE(3) group exponential and logarithm, tangent (rho, w) ----
  inline Eigen::Matrix3d se3V(const Eigen::Vector3d& w) {
    const double th2      = w.squaredNorm();
    const Eigen::Matrix3d W = skew(w);
    if (th2 < 1e-10) {
      return Eigen::Matrix3d::Identity() + 0.5 * W + W * W / 6.0;
    }
    const double th = std::sqrt(th2);
    return Eigen::Matrix3d::Identity() + (1.0 - std::cos(th)) / th2 * W + (th - std::sin(th)) / (th2 * th) * W * W;
  }

  inline Isometry3d expSE3True(const Vector6d& x) {
    Isometry3d T    = Isometry3d::Identity();
    T.linear()      = expMapSO3(x.tail<3>());
    T.translation() = se3V(x.tail<3>()) * x.head<3>();
    return T;
  }

  inline Vector6d logSE3True(const Isometry3d& T) {
    Vector6d x;
    x.tail<3>() = logMapSO3(T.linear());
    x.head<3>() = se3V(x.tail<3>()).inverse() * T.translation();
    return x;
  }

  // ---- The chart in use (LieOptions::true_exponential) ----
  inline Isometry2d chartExpSE2(const Eigen::Vector3d& x) {
    return LieOptions::true_exponential ? expSE2True(x) : expMapSE2(x);
  }
  inline Eigen::Vector3d chartLogSE2(const Isometry2d& T) {
    return LieOptions::true_exponential ? logSE2True(T) : logMapSE2(T);
  }
  inline Isometry3d chartExpSE3(const Vector6d& x) {
    return LieOptions::true_exponential ? expSE3True(x) : expMapSE3(x);
  }
  inline Vector6d chartLogSE3(const Isometry3d& T) {
    return LieOptions::true_exponential ? logSE3True(T) : logMapSE3(T);
  }

  // Jacobian of the change of chart y(d) = boxminus(to, boxplus(from, tau + d)) at d = 0, by
  // central differences (VariableType: SE2Pose or SE3Pose, with static boxplus / boxminus)
  template <typename VariableType>
  typename VariableType::MatrixLieDim chartChangeJacobian(const typename VariableType::EstimateType& to,
                                                          const typename VariableType::EstimateType& from,
                                                          const typename VariableType::VectorLieDim& tau) {
    using Vec          = typename VariableType::VectorLieDim;
    constexpr double h = 1e-6;
    typename VariableType::MatrixLieDim J;
    for (int k = 0; k < VariableType::LieDim; ++k) {
      Vec d = Vec::Zero();
      d[k]  = h;
      J.col(k) = (VariableType::boxminus(to, VariableType::boxplus(from, Vec(tau + d))) -
                  VariableType::boxminus(to, VariableType::boxplus(from, Vec(tau - d)))) /
                 (2.0 * h);
    }
    return J;
  }

  // A precision in the tangent space at `from` (Gaussian around from (+) tau) expressed in the
  // tangent space at `to`: J^-T Lambda J^-1 with LieOptions::exact_transport, else unchanged
  template <typename VariableType>
  typename VariableType::MatrixLieDim transportPrecision(const typename VariableType::EstimateType& to,
                                                         const typename VariableType::EstimateType& from,
                                                         const typename VariableType::VectorLieDim& tau,
                                                         const typename VariableType::MatrixLieDim& Lambda) {
    if (!LieOptions::exact_transport || Lambda.isZero(0.0)) {
      return Lambda;
    }
    const typename VariableType::MatrixLieDim Jinv = chartChangeJacobian<VariableType>(to, from, tau).inverse();
    typename VariableType::MatrixLieDim out        = Jinv.transpose() * Lambda * Jinv;
    return 0.5 * (out + out.transpose());
  }

  template <typename VariableType>
  typename VariableType::MatrixLieDim transportPrecision(const typename VariableType::EstimateType& to,
                                                         const typename VariableType::EstimateType& from,
                                                         const typename VariableType::MatrixLieDim& Lambda) {
    return transportPrecision<VariableType>(to, from, VariableType::VectorLieDim::Zero(), Lambda);
  }

} // namespace gsolver
