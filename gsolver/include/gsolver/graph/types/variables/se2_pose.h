#pragma once

#include <Eigen/Dense>

#include <gsolver/graph/variable_node.h>
#include <gsolver/maths/geometry2d.h>

namespace gsolver {

  class SE2Pose : public VariableNode<3, Eigen::Isometry2d, Edge<SE2Pose>> {
  public:
    SE2Pose(const std::string& id, const EstimateType& mu, const MatrixLieDim& Sigma);

    const VectorLieDim boxminus(const EstimateType& mu_inward) override;

    const EstimateType boxplus(const VectorLieDim& tau) override;

    static const EstimateType boxplus(const EstimateType& mu, const VectorLieDim& tau);
  };

} // namespace gsolver
