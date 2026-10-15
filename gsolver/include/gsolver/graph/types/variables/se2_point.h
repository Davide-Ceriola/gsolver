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
  };

} // namespace gsolver
