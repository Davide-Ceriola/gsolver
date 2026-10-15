#pragma once

#include <Eigen/Dense>

#include <gsolver/graph/factor_node.h>
#include <gsolver/graph/types/variables/se2_pose.h>
#include <gsolver/maths/geometry2d.h>

namespace gsolver {

  /**
   * @brief Range and bearing from an observer SE2Pose x_i to the position of another SE2Pose x_j.
   *
   * z = [range, bearing in radians], in the frame of x_i.
   * p     = R_i^T (t_j - t_i)
   * error = [|p| - z_range, wrap(atan2(p_y, p_x) - z_bearing)]
   */
  class SE2RangeBearing : public FactorNode<2, SE2Pose, SE2Pose> {
  public:
    Eigen::Vector2d z_;

    SE2RangeBearing(std::string id, const Eigen::Vector2d& z, MatrixErrorDim Sigma);

    void updateErrorAndJacobian() override;

    // Error and Jacobian [d error / d tau_i | d error / d tau_j] at the given linearization point
    static void evaluate(const Isometry2d& xi,
                         const Isometry2d& xj,
                         const Eigen::Vector2d& z,
                         VectorErrorDim& error,
                         MatrixJacobianDim& J);
  };

} // namespace gsolver
