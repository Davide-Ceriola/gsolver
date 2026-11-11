#include <gsolver/graph/types/factors/se3_pose_pose.h>
#include <gsolver/maths/lie_options.h>

namespace gsolver {

  SE3PosePose::SE3PosePose(std::string id, const Eigen::Isometry3d& z, MatrixErrorDim Sigma) : FactorNode(id, Sigma) {
    z_ = z;
  }

  void SE3PosePose::updateErrorAndJacobian() {
    evaluate(edges_.at<0>()->variable_to_factor_message_.mu_, edges_.at<1>()->variable_to_factor_message_.mu_, z_, error_, J_);
  }

  void SE3PosePose::evaluate(const Eigen::Isometry3d& xi,
                             const Eigen::Isometry3d& xj,
                             const Eigen::Isometry3d& z,
                             VectorErrorDim& error,
                             MatrixJacobianDim& J) {
    if (LieOptions::true_exponential) {
      error              = chartLogSE3(z.inverse() * xi.inverse() * xj);
      constexpr double e = 1e-6;
      for (int k = 0; k < 12; ++k) {
        Vector6d d        = Vector6d::Zero();
        d[k % 6]          = e;
        const bool on_i   = k < 6;
        const Eigen::Isometry3d ip = on_i ? SE3Pose::boxplus(xi, d) : xi, im = on_i ? SE3Pose::boxplus(xi, Vector6d(-d)) : xi;
        const Eigen::Isometry3d jp = on_i ? xj : SE3Pose::boxplus(xj, d), jm = on_i ? xj : SE3Pose::boxplus(xj, Vector6d(-d));
        J.col(k) = (chartLogSE3(z.inverse() * ip.inverse() * jp) - chartLogSE3(z.inverse() * im.inverse() * jm)) / (2 * e);
      }
      return;
    }
    error = logMapSE3(z.inverse() * xi.inverse() * xj);
    // Same derivatives as SE3PoseVelPose, whose pose block is this factor's first variable
    J.block<6, 6>(0, 0) = DerivativeUtils::jacob_dDinvP1invP2_de1e2(z.inverse(), xi, xj, 1);
    J.block<6, 6>(0, 6) = DerivativeUtils::jacob_dDinvP1invP2_de1e2(z.inverse(), xi, xj, 2);
  }

  SE3PosePrior::SE3PosePrior(std::string id, const Eigen::Isometry3d& z, MatrixErrorDim Sigma) : FactorNode(id, Sigma) {
    z_ = z;
  }

  void SE3PosePrior::updateErrorAndJacobian() {
    evaluate(edges_.at<0>()->variable_to_factor_message_.mu_, z_, error_, J_);
  }

  void SE3PosePrior::evaluate(const Eigen::Isometry3d& x, const Eigen::Isometry3d& z, VectorErrorDim& error, MatrixJacobianDim& J) {
    if (LieOptions::true_exponential) {
      error              = chartLogSE3(z.inverse() * x);
      constexpr double e = 1e-6;
      for (int k = 0; k < 6; ++k) {
        Vector6d d = Vector6d::Zero();
        d[k]       = e;
        J.col(k) = (chartLogSE3(z.inverse() * SE3Pose::boxplus(x, d)) - chartLogSE3(z.inverse() * SE3Pose::boxplus(x, Vector6d(-d)))) / (2 * e);
      }
      return;
    }
    error = logMapSE3(z.inverse() * x);
    // As SE3PoseVelPrior: log(z^-1 * I^-1 * x exp(tau))
    J = DerivativeUtils::jacob_dDinvP1invP2_de1e2(z.inverse(), Eigen::Isometry3d::Identity(), x, 2);
  }

} // namespace gsolver
