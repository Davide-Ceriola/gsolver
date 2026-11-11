#pragma once

#include <Eigen/Dense>

#include <gsolver/graph/factor_node.h>
#include <gsolver/graph/types/variables/se3_pose.h>
#include <gsolver/maths/derivatives.h>
#include <gsolver/maths/geometry3d.h>

namespace gsolver {

  /**
   * @brief Relative pose between two SE3Pose variables (odometry or loop closure): the
   * pose-graph edge (g2o EDGE_SE3:QUAT).
   *
   * z is the measured pose of x_j in the frame of x_i.
   * error = log(z^-1 * x_i^-1 * x_j), translation then rotation (as logMapSE3).
   */
  class SE3PosePose : public FactorNode<6, SE3Pose, SE3Pose> {
  public:
    Eigen::Isometry3d z_;

    SE3PosePose(std::string id, const Eigen::Isometry3d& z, MatrixErrorDim Sigma);

    void updateErrorAndJacobian() override;

    // Error and Jacobian [d error / d tau_i | d error / d tau_j] at the given linearization point
    static void evaluate(const Eigen::Isometry3d& xi,
                         const Eigen::Isometry3d& xj,
                         const Eigen::Isometry3d& z,
                         VectorErrorDim& error,
                         MatrixJacobianDim& J);
  };

  /**
   * @brief Prior on an SE3Pose: error = log(z^-1 * x), translation then rotation.
   */
  class SE3PosePrior : public FactorNode<6, SE3Pose> {
  public:
    Eigen::Isometry3d z_;

    SE3PosePrior(std::string id, const Eigen::Isometry3d& z, MatrixErrorDim Sigma);

    void updateErrorAndJacobian() override;

    static void evaluate(const Eigen::Isometry3d& x, const Eigen::Isometry3d& z, VectorErrorDim& error, MatrixJacobianDim& J);
  };

} // namespace gsolver
