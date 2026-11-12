#pragma once

#include <stdexcept>

#include <Eigen/Dense>

#include <gsolver/graph/edge_base.h>

namespace gsolver {

  template <typename VariableType_>
  struct Message {
    using VariableType = VariableType_;
    using EstimateType = typename VariableType::EstimateType;
    using MatrixLieDim = typename VariableType::MatrixLieDim;

    bool processed_ = false; // Flag indicating if the message has been processed
    MatrixLieDim Lambda_; // Lambda matrix (precision matrix of the factor)
    EstimateType mu_;     // Estimate of the variable

    // Variable-to-factor messages also carry the sender's belief mean, used by factors that
    // linearise at the beliefs (FactorNodeBase::linearise_at_belief_); unused otherwise
    EstimateType belief_mu_;
    bool has_belief_ = false;

    // Belief messages (VariableNodeBase::send_belief_): the variable sends its whole belief
    // (mu_ = belief mean, Lambda_ = sum of all incoming precisions, eta_ = sum of all incoming
    // information vectors in the tangent space at mu_) and the factor recovers what the variable
    // would have sent it by removing its own last message (FactorNode: convertBeliefMessages)
    Eigen::Matrix<double, MatrixLieDim::RowsAtCompileTime, 1> eta_ =
      Eigen::Matrix<double, MatrixLieDim::RowsAtCompileTime, 1>::Zero();
    bool is_belief_ = false;
  };

  template <typename VariableType>
  class Edge : public EdgeBase {
  public:
    using VariableType_ = VariableType;
    using MessageType   = Message<VariableType>;

    static constexpr int LieDim      = VariableType::LieDim;
    static constexpr int PayloadSize = VariableType::EncodedSize + LieDim * (LieDim + 1) / 2;

    MessageType variable_to_factor_message_;
    MessageType factor_to_variable_message_;

    // Damping of the factor-to-variable message on this edge, set by a scheduler; negative: the
    // factor's own (MessageUpdate::damping)
    double damping_ = -1.0;

    // Factor-side scheduling (FactorNodeBase::sendRepliesIfPending):
    //  reply_pending_:  the factor has new information for this variable it has not sent yet
    //  reply_selected_: the factor's scheduler lets it reply on this edge now (1) or not (0)
    bool reply_pending_ = false;
    int reply_selected_ = 1;
    // Damping of the incoming variable-to-factor message, applied by the factor when it receives
    // a new one (FactorNodeBase::prepareIncoming); negative: none. v2f_prev_ is the last
    // (damped) incoming message.
    double v2f_damping_ = -1.0;
    MessageType v2f_prev_;
    bool has_v2f_prev_ = false;

    // The factor's linearisation point for this variable (linearise_at_belief_ mode only)
    typename VariableType::EstimateType lin_point_;
    bool has_lin_point_ = false;

    int payloadSize() const override {
      return PayloadSize;
    }

    void writeOutgoingPayload(double* payload) const override {
      const MessageType& message = outgoing();
      VariableType::encode(message.mu_, payload);
      double* lambda = payload + VariableType::EncodedSize;
      for (int r = 0; r < LieDim; ++r) {
        for (int c = r; c < LieDim; ++c) {
          *lambda++ = 0.5 * (message.Lambda_(r, c) + message.Lambda_(c, r)); // symmetric part
        }
      }
    }

    void readIncomingPayload(const double* payload) override {
      MessageType& message = incoming();
      message.mu_          = VariableType::decode(payload);
      const double* lambda = payload + VariableType::EncodedSize;
      for (int r = 0; r < LieDim; ++r) {
        for (int c = r; c < LieDim; ++c) {
          message.Lambda_(r, c) = message.Lambda_(c, r) = *lambda++;
        }
      }
    }

  private:
    const MessageType& outgoing() const {
      switch (side_) {
        case EdgeSide::FACTOR_SIDE:
          return factor_to_variable_message_;
        case EdgeSide::VARIABLE_SIDE:
          return variable_to_factor_message_;
        default:
          throw std::logic_error("Edge payloads exist only for boundary edges");
      }
    }

    MessageType& incoming() {
      switch (side_) {
        case EdgeSide::FACTOR_SIDE:
          return variable_to_factor_message_;
        case EdgeSide::VARIABLE_SIDE:
          return factor_to_variable_message_;
        default:
          throw std::logic_error("Edge payloads exist only for boundary edges");
      }
    }
  };

} // namespace gsolver
