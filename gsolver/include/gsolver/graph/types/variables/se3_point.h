#pragma once

#include <Eigen/Dense>

#include <gsolver/graph/variable_node.h>

namespace gsolver {

  class SE3Point : public VariableNode<3, Eigen::Vector3d, Edge<SE3Point>> {
  public:
    SE3Point(const std::string& id, const EstimateType& mu, const MatrixLieDim& Sigma);

    const VectorLieDim boxminus(const EstimateType& mu_inward) override;

    const EstimateType boxplus(const VectorLieDim& tau_inward) override;

    static const EstimateType boxplus(const EstimateType& mu, const VectorLieDim& tau_inward);

    // Mean as transmitted in boundary-edge payloads: x, y, z
    static constexpr int EncodedSize = 3;
    static void encode(const EstimateType& mu, double* out) {
      out[0] = mu[0], out[1] = mu[1], out[2] = mu[2];
    }
    static EstimateType decode(const double* in) {
      return EstimateType(in[0], in[1], in[2]);
    }
  };

} // namespace gsolver
