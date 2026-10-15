#include <gsolver/graph/types/variables/se2_point.h>

namespace gsolver {

  // Type aliases from base class
  using EstimateType = SE2Point::EstimateType;
  using VectorLieDim = SE2Point::VectorLieDim;
  using MatrixLieDim = SE2Point::MatrixLieDim;

  SE2Point::SE2Point(const std::string& id, const EstimateType& mu, const MatrixLieDim& Sigma)
      : VariableNode(id, mu, Sigma) {}

  const VectorLieDim SE2Point::boxminus(const EstimateType& mu_inward) {
    return mu_inward - mu_;
  }

  const EstimateType SE2Point::boxplus(const VectorLieDim& tau_inward) {
    return EstimateType(static_cast<Eigen::Vector2d>(mu_) + tau_inward);
  }

  const EstimateType SE2Point::boxplus(const EstimateType& mu, const VectorLieDim& tau_inward) {
    return EstimateType(static_cast<Eigen::Vector2d>(mu) + tau_inward);
  }

} // namespace gsolver
