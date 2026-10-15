#include <gsolver/graph/types/factors/se2_prior.h>

namespace gsolver {

  SE2Prior::SE2Prior(std::string id, const Isometry2d& z, MatrixErrorDim Sigma) : FactorNode(id, Sigma) {
    z_ = z;
  }

  void SE2Prior::updateErrorAndJacobian() {
    const Isometry2d& x = edges_.at<0>()->variable_to_factor_message_.mu_;
    evaluate(x, z_, error_, J_);
  }

  void SE2Prior::evaluate(const Isometry2d& x, const Isometry2d& z, VectorErrorDim& error, MatrixJacobianDim& J) {
    const Isometry2d delta = z.inverse() * x;
    error                  = logMapSE2(delta);

    // log(delta * exp(tau)) = [t_delta + R_delta * tau_t, theta_delta + tau_theta] with this chart
    J                   = MatrixJacobianDim::Identity();
    J.block<2, 2>(0, 0) = delta.linear();
  }

} // namespace gsolver
