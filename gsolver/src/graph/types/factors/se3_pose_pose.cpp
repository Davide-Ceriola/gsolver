#include <gsolver/graph/types/factors/se3_pose_pose.h>

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
    error = logMapSE3(z.inverse() * x);
    // As SE3PoseVelPrior: log(z^-1 * I^-1 * x exp(tau))
    J = DerivativeUtils::jacob_dDinvP1invP2_de1e2(z.inverse(), Eigen::Isometry3d::Identity(), x, 2);
  }

} // namespace gsolver
