#pragma once

#include <Eigen/Dense>

#include <gsolver/graph/variable_node.h>

namespace gsolver {

  class SE2Point : public VariableNode<2, Eigen::Vector2d, Edge<SE2Point>> {
  public:
    SE2Point(const std::string& id, const EstimateType& mu, const MatrixLieDim& Sigma);

    const VectorLieDim boxminus(const EstimateType& mu_inward) override;

    const EstimateType boxplus(const VectorLieDim& tau_inward) override;

    static const EstimateType boxplus(const EstimateType& mu, const VectorLieDim& tau_inward);

    // Mean as transmitted in boundary-edge payloads: x, y
    static constexpr int EncodedSize = 2;
    static void encode(const EstimateType& mu, double* out) {
      out[0] = mu[0], out[1] = mu[1];
    }
    static EstimateType decode(const double* in) {
      return EstimateType(in[0], in[1]);
    }
  };

} // namespace gsolver
