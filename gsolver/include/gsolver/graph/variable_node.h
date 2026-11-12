#pragma once

#include <Eigen/Dense>
#include <memory>

#include <gsolver/graph/edge.h>
#include <gsolver/graph/variable_node_base.h>
#include <gsolver/maths/symmetric.h>

namespace gsolver {

  template <int LieDim_, typename EstimateType_, typename EdgeType_>
  class VariableNode : public VariableNodeBase {
  public:
    // using VariableType = VariableNode<LieDim_, EstimateType_>; // Type for variable node
    using EstimateType = EstimateType_; // Type for mu (estimate in manifold)
    using EdgeType     = EdgeType_;     // Type for edge
    using VectorLieDim = Eigen::Matrix<double, LieDim_, 1>;
    using MatrixLieDim = Eigen::Matrix<double, LieDim_, LieDim_>;

    constexpr static int LieDim = LieDim_;

    VariableNode(const std::string& id, const EstimateType_& mu, const MatrixLieDim& Sigma);

    EstimateType mu_;    // Estimate of the variable
    MatrixLieDim Sigma_; // Covariance matrix of the variable

    std::vector<std::shared_ptr<EdgeType>> edges_; // Edges connected to this factor node

    void sendMessage(size_t outedge_id) override;
    void sendMessage(const std::string& factor_id) override;
    void sendAllMessages() override;
    void sendMessages(const std::vector<int>& mask) override;
    int lieDim() const override {
      return LieDim_;
    }
    void ruleReset() override;
    void ruleInputs(long tick, const double* edge_energy, double* out, int stride, bool update) override;
    void ruleRecordSends(const std::vector<int>& mask, long tick) override;
    void setIncomingDamping(size_t j, double d) override {
      edges_.at(j)->damping_ = d;
    }
    void updateMuAndSigma() override;
    bool hasUnprocessedMessages() const override;
    void markMessagesProcessed() override;
    std::shared_ptr<EdgeBase> addDanglingEdge(const std::string& factor_id, int remote_id) override;

    size_t numberOfEdges() const override {
      return edges_.size();
    }
    std::shared_ptr<EdgeBase> getEdgeAt(size_t i) override {
      return edges_.at(i);
    }
    void removeEdge(const std::string& factor_id) override;

  private:
    double rule_prev_logdet_ = 0.0;
    bool rule_has_logdet_    = false;
    // The belief message (send_belief_): mean, total precision and information vector at the mean
    template <typename MessageType>
    void writeBelief(MessageType& message, const MatrixLieDim& Lambda, const VectorLieDim& eta) const {
      message.mu_         = mu_;
      message.Lambda_     = Lambda;
      message.eta_        = eta;
      message.is_belief_  = true;
      message.processed_  = false;
      message.belief_mu_  = mu_;
      message.has_belief_ = true;
    }
    void beliefSums(MatrixLieDim& Lambda, VectorLieDim& eta);
    // Scratch space for sendAllMessages, kept to avoid allocating every iteration
    std::vector<VectorLieDim> tangent_eta_in_;
    std::vector<MatrixLieDim> tangent_Lambda_in_;

  public:

    // For each new VariableNode, implement the following functions:
    virtual const VectorLieDim boxminus(const EstimateType& mu_inward) = 0;
    virtual const EstimateType boxplus(const VectorLieDim& tau)        = 0;
    // static const EstimateType boxplus(const EstimateType& mu, const VectorLieDim& tau); // implemented in the derived class
  };

} // namespace gsolver