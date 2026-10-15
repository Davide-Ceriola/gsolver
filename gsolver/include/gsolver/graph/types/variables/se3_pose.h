#pragma once

#include <Eigen/Dense>

#include <gsolver/graph/variable_node.h>
#include <gsolver/maths/geometry3d.h>

namespace gsolver {

  class SE3Pose : public VariableNode<6, Eigen::Isometry3d, Edge<SE3Pose>> {
  public:
    SE3Pose(const std::string& id, const EstimateType& mu, const MatrixLieDim& Sigma);

    const VectorLieDim boxminus(const EstimateType& mu_inward) override;

    const EstimateType boxplus(const VectorLieDim& tau) override;

    static const EstimateType boxplus(const EstimateType& mu, const VectorLieDim& tau);

    // Mean as transmitted in boundary-edge payloads: translation, then rotation vector (log of R)
    static constexpr int EncodedSize = 6;
    static void encode(const EstimateType& mu, double* out) {
      const Eigen::Vector3d omega = logMapSO3(mu.linear());
      for (int i = 0; i < 3; ++i) {
        out[i]     = mu.translation()[i];
        out[3 + i] = omega[i];
      }
    }
    static EstimateType decode(const double* in) {
      EstimateType mu  = EstimateType::Identity();
      mu.translation() = Eigen::Vector3d(in[0], in[1], in[2]);
      mu.linear()      = expMapSO3(Eigen::Vector3d(in[3], in[4], in[5]));
      return mu;
    }
  };

} // namespace gsolver
