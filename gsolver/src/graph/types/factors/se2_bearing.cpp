#include <gsolver/graph/types/factors/se2_bearing.h>

namespace gsolver {

  namespace {
    // Below this squared distance the bearing is undefined; clamp to keep the Jacobian finite
    constexpr double kMinSquaredRange = 1e-12;
  } // namespace

  SE2Bearing::SE2Bearing(std::string id, const double& z, MatrixErrorDim Sigma) : FactorNode(id, Sigma) {
    z_ = z;
  }

  void SE2Bearing::updateErrorAndJacobian() {
    const Isometry2d& xi = edges_.at<0>()->variable_to_factor_message_.mu_;
    const Isometry2d& xj = edges_.at<1>()->variable_to_factor_message_.mu_;
    evaluate(xi, xj, z_, error_, J_);
  }

  void SE2Bearing::evaluate(const Isometry2d& xi, const Isometry2d& xj, const double& z, VectorErrorDim& error, MatrixJacobianDim& J) {
    const Eigen::Matrix2d Ri_t = xi.linear().transpose();
    const Eigen::Vector2d p    = Ri_t * (xj.translation() - xi.translation());
    const double r2            = std::max(p.squaredNorm(), kMinSquaredRange);

    error(0) = wrapAngle(atan2(p.y(), p.x()) - z);

    // d bearing / d p
    const Eigen::RowVector2d dh_dp(-p.y() / r2, p.x() / r2);

    J = MatrixJacobianDim::Zero();
    // perturbing x_i: p -> p - tau_t - tau_theta * perp(p), and dh_dp * perp(p) = 1
    J.block<1, 2>(0, 0) = -dh_dp;
    J(0, 2)             = -1.0;
    // perturbing x_j: p -> p + R_i^T R_j tau_t; the heading of x_j does not matter
    J.block<1, 2>(0, 3) = dh_dp * Ri_t * xj.linear();
  }

} // namespace gsolver
