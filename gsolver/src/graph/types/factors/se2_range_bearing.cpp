#include <gsolver/graph/types/factors/se2_range_bearing.h>

namespace gsolver {

  namespace {
    // Below this distance range and bearing are undefined; clamp to keep the Jacobian finite
    constexpr double kMinRange = 1e-6;
  } // namespace

  SE2RangeBearing::SE2RangeBearing(std::string id, const Eigen::Vector2d& z, MatrixErrorDim Sigma) : FactorNode(id, Sigma) {
    z_ = z;
  }

  void SE2RangeBearing::updateErrorAndJacobian() {
    const Isometry2d& xi = edges_.at<0>()->variable_to_factor_message_.mu_;
    const Isometry2d& xj = edges_.at<1>()->variable_to_factor_message_.mu_;
    evaluate(xi, xj, z_, error_, J_);
  }

  void SE2RangeBearing::evaluate(const Isometry2d& xi,
                                 const Isometry2d& xj,
                                 const Eigen::Vector2d& z,
                                 VectorErrorDim& error,
                                 MatrixJacobianDim& J) {
    const Eigen::Matrix2d Ri_t = xi.linear().transpose();
    const Eigen::Vector2d p    = Ri_t * (xj.translation() - xi.translation());
    const double r             = std::max(p.norm(), kMinRange);

    error(0) = r - z(0);
    error(1) = wrapAngle(atan2(p.y(), p.x()) - z(1));

    // d range / d p and d bearing / d p
    Eigen::Matrix2d de_dp;
    de_dp.row(0) = p.transpose() / r;
    de_dp.row(1) = Eigen::RowVector2d(-p.y(), p.x()) / (r * r);

    // Perturbing x_i: p -> p - tau_t - tau_theta * perp(p). Range is unchanged by
    // the rotation (p . perp(p) = 0); bearing changes by -tau_theta.
    // Perturbing x_j: p -> p + R_i^T R_j tau_t; the heading of x_j does not matter.
    J                   = MatrixJacobianDim::Zero();
    J.block<2, 2>(0, 0) = -de_dp;
    J(0, 2)             = 0.0;
    J(1, 2)             = -1.0;
    J.block<2, 2>(0, 3) = de_dp * Ri_t * xj.linear();
  }

} // namespace gsolver
