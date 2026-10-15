#pragma once

#include <Eigen/Dense>

#include <gsolver/graph/variable_node.h>
#include <gsolver/maths/geometry3d.h>

namespace gsolver {

  struct SE3PoseVelValue {
    Eigen::Isometry3d pose;
    Eigen::Vector<double, 6> vel;
  };

  class SE3PoseVel : public VariableNode<12, SE3PoseVelValue, Edge<SE3PoseVel>> {
  public:
    using EstimateType = SE3PoseVelValue;
    double timestamp_; // Timestamp associated with the pose

    SE3PoseVel(const std::string& id, const EstimateType& mu, const MatrixLieDim& Sigma, double timestamp);

    const VectorLieDim boxminus(const EstimateType& mu_inward) override;

    const EstimateType boxplus(const VectorLieDim& tau) override;

    static const EstimateType boxplus(const EstimateType& mu, const VectorLieDim& tau);

    // Mean as transmitted in boundary-edge payloads: translation, rotation vector (log of R), velocity
    static constexpr int EncodedSize = 12;
    static void encode(const EstimateType& mu, double* out) {
      const Eigen::Vector3d omega = logMapSO3(mu.pose.linear());
      for (int i = 0; i < 3; ++i) {
        out[i]     = mu.pose.translation()[i];
        out[3 + i] = omega[i];
      }
      for (int i = 0; i < 6; ++i) {
        out[6 + i] = mu.vel[i];
      }
    }
    static EstimateType decode(const double* in) {
      EstimateType mu;
      mu.pose               = Eigen::Isometry3d::Identity();
      mu.pose.translation() = Eigen::Vector3d(in[0], in[1], in[2]);
      mu.pose.linear()      = expMapSO3(Eigen::Vector3d(in[3], in[4], in[5]));
      for (int i = 0; i < 6; ++i) {
        mu.vel[i] = in[6 + i];
      }
      return mu;
    }
  };

} // namespace gsolver
