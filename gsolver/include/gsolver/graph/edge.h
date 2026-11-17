#pragma once

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <vector>

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


    // ---- Generic rule state (core/generic_scheduler.h) ----
    using EstimateT = typename MessageType::EstimateType;
    using VectorT   = Eigen::Matrix<double, MessageType::MatrixLieDim::RowsAtCompileTime, 1>;
    // variable side
    EstimateT last_sent_mu_;
    long last_sent_tick_  = 0;
    bool has_last_sent_   = false;
    EstimateT v_osc_prev_;
    VectorT v_osc_delta_ = VectorT::Zero();
    bool v_osc_has_prev_ = false, v_osc_has_delta_ = false;
    double v_osc_cos_    = 0.0;
    // local eigenvalue of the incoming message's iteration: <change, previous change> / |previous|^2
    double v_osc_ratio_  = 0.0;
    // belief covariance at the last send, and its precision and log det (factorised lazily)
    typename MessageType::MatrixLieDim last_sent_sigma_ = MessageType::MatrixLieDim::Identity();
    typename MessageType::MatrixLieDim kl_ref_lambda_   = MessageType::MatrixLieDim::Identity();
    double kl_ref_logdet_ = 0.0;
    bool kl_ref_valid_    = false;
    // the cavity (belief without this edge's incoming message) at the last send: mean, precision,
    // log det of its covariance (what the factor received)
    EstimateT last_sent_cav_mu_;
    typename MessageType::MatrixLieDim kl_cav_lambda_ = MessageType::MatrixLieDim::Zero();
    double kl_cav_logdet_ = 0.0;
    bool kl_cav_valid_    = false;
    double prev_energy_  = -1.0;
    // memory over many ticks (exponential averages): the edge's usual change, and the direction and
    // size of its energy changes (settling: steadily down; oscillating: up and down)
    double hist_moved_ema_     = -1.0; // negative: no history yet
    double hist_trend_ema_     = 0.0;
    double hist_trend_abs_ema_ = 0.0;
    // factor side
    EstimateT f_inc_prev_;
    VectorT f_inc_delta_ = VectorT::Zero();
    bool f_inc_has_prev_ = false, f_inc_has_delta_ = false;
    double f_inc_cos_    = 0.0;
    std::vector<EstimateT> f_news_snap_;

    bool incomingFresh() const override {
      return !variable_to_factor_message_.processed_;
    }

    int lieDim() const override {
      return MessageType::MatrixLieDim::RowsAtCompileTime;
    }

    void ruleIncoming(bool fresh, double& change_per_dim, double& oscillation) override {
      const auto& m = variable_to_factor_message_;
      change_per_dim = 0.0;
      if (fresh && f_inc_has_prev_) {
        const VectorT d   = VariableType::boxminus(f_inc_prev_, m.mu_);
        const double sq   = d.dot(m.Lambda_ * d);
        change_per_dim    = std::sqrt(std::max(sq, 0.0) / lieDim());
        if (change_per_dim > 1e-3) { // below a thousandth of a sigma: no change
          if (f_inc_has_delta_) {
            const double a = std::sqrt(std::max(f_inc_delta_.dot(m.Lambda_ * f_inc_delta_), 0.0));
            if (a > 1e-12) {
              f_inc_cos_ = d.dot(m.Lambda_ * f_inc_delta_) / (std::sqrt(sq) * a);
            }
          }
          f_inc_delta_     = d;
          f_inc_has_delta_ = true;
        }
      }
      if (fresh || !f_inc_has_prev_) {
        f_inc_prev_     = m.mu_;
        f_inc_has_prev_ = true;
      }
      oscillation = f_inc_cos_;
    }

    double ruleNewsSince(size_t p) const override {
      const auto& m = variable_to_factor_message_;
      if (p >= f_news_snap_.size() || m.Lambda_.isZero(0.0)) {
        return 0.0;
      }
      const VectorT d = VariableType::boxminus(f_news_snap_[p], m.mu_);
      return std::sqrt(std::max(d.dot(m.Lambda_ * d), 0.0) / lieDim());
    }

    void ruleSnapNews(size_t p) override {
      if (p < f_news_snap_.size()) {
        f_news_snap_[p] = variable_to_factor_message_.mu_;
      }
    }

    void ruleResetFactorSide(size_t factor_arity) override {
      f_news_snap_.assign(factor_arity, variable_to_factor_message_.mu_);
      f_inc_has_prev_ = f_inc_has_delta_ = false;
      f_inc_cos_      = 0.0;
    }

    // Factor-side scheduling (FactorNodeBase::sendRepliesIfPending):
    // Damping of the incoming variable-to-factor message, applied by the factor when it receives
    // a new one (FactorNodeBase::prepareIncoming); negative: none. v2f_prev_ is the last
    // (damped) incoming message.
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
