#include <gsolver/graph/types/variables/se2_pose.h>

namespace gsolver {

  using EstimateType = SE2Pose::EstimateType;
  using VectorLieDim = SE2Pose::VectorLieDim;
  using MatrixLieDim = SE2Pose::MatrixLieDim;

  SE2Pose::SE2Pose(const std::string& id, const EstimateType& mu, const MatrixLieDim& Sigma) : VariableNode(id, mu, Sigma) {
  }

  const VectorLieDim SE2Pose::boxminus(const EstimateType& mu_inward) {
    VectorLieDim tau_outward = logMapSE2(mu_.inverse() * mu_inward);
    return tau_outward;
  }

  const EstimateType SE2Pose::boxplus(const VectorLieDim& tau) {
    EstimateType mu_outward = mu_ * expMapSE2(tau);
    return mu_outward;
  }

  const EstimateType SE2Pose::boxplus(const EstimateType& mu, const VectorLieDim& tau) {
    EstimateType mu_outward = mu * expMapSE2(tau);
    return mu_outward;
  }

} // namespace gsolver
