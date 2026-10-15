#pragma once

#include <Eigen/Dense>

#include <gsolver/graph/factor_node.h>
#include <gsolver/graph/types/variables/se2_pose.h>
#include <gsolver/maths/geometry2d.h>

namespace gsolver {

  /**
   * @brief Relative pose (odometry) between two SE2Pose variables.
   *
   * z is the measured pose of x_j in the frame of x_i.
   * error = log(z^-1 * x_i^-1 * x_j)
   */
  class SE2Odometry : public FactorNode<3, SE2Pose, SE2Pose> {
  public:
    Isometry2d z_;

    SE2Odometry(std::string id, const Isometry2d& z, MatrixErrorDim Sigma);

    void updateErrorAndJacobian() override;

    // Error and Jacobian [d error / d tau_i | d error / d tau_j] at the given linearization point
    static void evaluate(const Isometry2d& xi,
                         const Isometry2d& xj,
                         const Isometry2d& z,
                         VectorErrorDim& error,
                         MatrixJacobianDim& J);
  };

} // namespace gsolver
