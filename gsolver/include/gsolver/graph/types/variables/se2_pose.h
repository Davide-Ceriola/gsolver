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

    // Mean as transmitted in boundary-edge payloads: x, y, theta
    static constexpr int EncodedSize = 3;
    static void encode(const EstimateType& mu, double* out) {
      const Eigen::Vector3d v = logMapSE2(mu);
      out[0] = v[0], out[1] = v[1], out[2] = v[2];
    }
    static EstimateType decode(const double* in) {
      return expMapSE2(Eigen::Vector3d(in[0], in[1], in[2]));
    }
  };

} // namespace gsolver
