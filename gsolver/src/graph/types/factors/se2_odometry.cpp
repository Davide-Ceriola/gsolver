#include <gsolver/graph/types/factors/se2_odometry.h>
#include <gsolver/maths/lie_options.h>

namespace gsolver {

  SE2Odometry::SE2Odometry(std::string id, const Isometry2d& z, MatrixErrorDim Sigma) : FactorNode(id, Sigma) {
    z_ = z;
  }

  void SE2Odometry::updateErrorAndJacobian() {
    const Isometry2d& xi = edges_.at<0>()->variable_to_factor_message_.mu_;
    const Isometry2d& xj = edges_.at<1>()->variable_to_factor_message_.mu_;
    evaluate(xi, xj, z_, error_, J_);
  }

  void SE2Odometry::evaluate(const Isometry2d& xi,
                             const Isometry2d& xj,
                             const Isometry2d& z,
                             VectorErrorDim& error,
                             MatrixJacobianDim& J) {
    const Isometry2d h     = xi.inverse() * xj;
    const Isometry2d delta = z.inverse() * h;
    if (LieOptions::true_exponential) {
      // Group exponential: error in its chart, Jacobian by central differences of that error
      error              = chartLogSE2(delta);
      constexpr double e = 1e-6;
      for (int k = 0; k < 6; ++k) {
        Eigen::Vector3d d = Eigen::Vector3d::Zero();
        d[k % 3]          = e;
        const bool on_i   = k < 3;
        const Isometry2d ip = on_i ? SE2Pose::boxplus(xi, d) : xi, im = on_i ? SE2Pose::boxplus(xi, Eigen::Vector3d(-d)) : xi;
        const Isometry2d jp = on_i ? xj : SE2Pose::boxplus(xj, d), jm = on_i ? xj : SE2Pose::boxplus(xj, Eigen::Vector3d(-d));
        J.col(k) = (chartLogSE2(z.inverse() * ip.inverse() * jp) - chartLogSE2(z.inverse() * im.inverse() * jm)) / (2 * e);
      }
      return;
    }
    error                  = logMapSE2(delta);

    const Eigen::Matrix2d Rz_t = z.linear().transpose();

    J = MatrixJacobianDim::Zero();
    // d / d tau_i: (x_i exp(tau_i))^-1 ~ exp(-tau_i) to first order
    J.block<2, 2>(0, 0) = -Rz_t;
    J.block<2, 1>(0, 2) = -Rz_t * perp(h.translation());
    J(2, 2)             = -1.0;
    // d / d tau_j: same structure as the prior
    J.block<2, 2>(0, 3) = delta.linear();
    J(2, 5)             = 1.0;
  }

} // namespace gsolver
