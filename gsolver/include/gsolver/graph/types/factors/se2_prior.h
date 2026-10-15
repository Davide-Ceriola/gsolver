#pragma once

#include <Eigen/Dense>

#include <gsolver/graph/factor_node.h>
#include <gsolver/graph/types/variables/se2_pose.h>
#include <gsolver/maths/geometry2d.h>

namespace gsolver {

  /**
   * @brief Prior on an SE2Pose.
   *
   * error = log(z^-1 * x)
   */
  class SE2Prior : public FactorNode<3, SE2Pose> {
  public:
    Isometry2d z_;

    SE2Prior(std::string id, const Isometry2d& z, MatrixErrorDim Sigma);

    void updateErrorAndJacobian() override;

    // Error and Jacobian (d error / d tau, right perturbation) at the given linearization point
    static void evaluate(const Isometry2d& x, const Isometry2d& z, VectorErrorDim& error, MatrixJacobianDim& J);
  };

} // namespace gsolver
