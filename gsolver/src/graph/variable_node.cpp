#include <gsolver/graph/variable_node.h>
#include <gsolver/maths/lie_options.h>

#include <cstring>

namespace gsolver {

  template <int Dim_, typename EstimateType_, typename EdgeType_>
  VariableNode<Dim_, EstimateType_, EdgeType_>::VariableNode(const std::string& id,
                                                             const EstimateType_& mu,
                                                             const MatrixLieDim& Sigma) {
    id_    = id;
    mu_    = mu;
    Sigma_ = Sigma;
  }

  template <int Dim_, typename EstimateType_, typename EdgeType_>
  void VariableNode<Dim_, EstimateType_, EdgeType_>::sendMessage(size_t outedge_id) {
    VectorLieDim tangent_eta         = VectorLieDim::Zero();
    MatrixLieDim tangent_Lambdaprime = MatrixLieDim::Zero();
    VectorLieDim tangent_delta_mu    = VectorLieDim::Zero();

    // If there are no factors connected to the variable node, skip the message computation
    if (edges_.size() == 0) {
      return;
    }
    message_update_.prepare(id_);
    if (edges_.size() > 1 && message_update_.drop()) {
      return; // dropped: the previous message stays
    }

    if (send_belief_) {
      if (edges_.size() == 1 && initial_guess_is_prior_) {
        return; // legacy: the edge keeps its initial message, as below
      }
      MatrixLieDim Lambda;
      VectorLieDim eta;
      beliefSums(Lambda, eta);
      writeBelief(edges_[outedge_id]->variable_to_factor_message_, Lambda, eta);
      return;
    }

    // With a single factor there are no other factors to multiply: the message carries no
    // information. Legacy mode leaves the edge's initial message (the guess) in place instead.
    if (edges_.size() == 1) {
      if (!initial_guess_is_prior_) {
        edges_[outedge_id]->variable_to_factor_message_.Lambda_    = MatrixLieDim::Zero();
        edges_[outedge_id]->variable_to_factor_message_.mu_        = mu_;
        edges_[outedge_id]->variable_to_factor_message_.processed_ = false;
        edges_[outedge_id]->variable_to_factor_message_.belief_mu_  = mu_;
        edges_[outedge_id]->variable_to_factor_message_.has_belief_ = true;
      }
      return;
    }

    // Multiply inward messages from other factors
    for (size_t i = 0; i < edges_.size(); ++i) {
      if (i == outedge_id) {
        // Skip the outgoing edge
        continue;
      }

      // Access messages from edges (TODO: replace with appropriate getter methods based on your edge structure)
      const MatrixLieDim& Lambdaprime_inward = edges_[i]->factor_to_variable_message_.Lambda_;
      const EstimateType_ mu_inward          = edges_[i]->factor_to_variable_message_.mu_;

      // Project to the tangent space
      VectorLieDim tangent_delta_mu_inward = boxminus(mu_inward);

      // Compute the Lambda and eta (precision moved to the tangent space at mu_: LieOptions)
      MatrixLieDim tangent_Lambda_inward = transportPrecision<typename EdgeType_::VariableType_>(mu_, mu_inward, Lambdaprime_inward);
      VectorLieDim tangent_eta_inward    = tangent_Lambda_inward * tangent_delta_mu_inward;

      tangent_eta += tangent_eta_inward;
      tangent_Lambdaprime += tangent_Lambda_inward;
    }
    // get the perturbation to be applied to the current estimate; regularized only to solve,
    // the message keeps the true precision
    tangent_delta_mu = SpdSolver<Dim_>(MatrixLieDim(tangent_Lambdaprime + MatrixLieDim::Identity() * kInformationRegularization))
                         .solve(tangent_eta);

    // compute the new estimate in the manifold
    EstimateType_ mu_outward         = boxplus(tangent_delta_mu);
    // the precision is in the tangent space at mu_; the message is sent at mu_outward (LieOptions)
    MatrixLieDim Lambdaprime_outward =
      transportPrecision<typename EdgeType_::VariableType_>(mu_outward, mu_, tangent_delta_mu, tangent_Lambdaprime);

    edges_[outedge_id]->variable_to_factor_message_.Lambda_    = Lambdaprime_outward;
    edges_[outedge_id]->variable_to_factor_message_.mu_        = mu_outward;
    edges_[outedge_id]->variable_to_factor_message_.processed_ = false;
    edges_[outedge_id]->variable_to_factor_message_.belief_mu_  = mu_;
    edges_[outedge_id]->variable_to_factor_message_.has_belief_ = true;
  }

  template <int Dim_, typename EstimateType_, typename EdgeType_>
  void VariableNode<Dim_, EstimateType_, EdgeType_>::sendMessage(const std::string& factor_id) {
    // Legacy mode sends nothing without at least two factors, so it never needs the edge
    if (edges_.size() == 0 || (edges_.size() == 1 && initial_guess_is_prior_)) {
      return;
    }

    // Find the outgoing edge index
    const auto it = std::find_if(edges_.begin(), edges_.end(), [&](const auto& edge) {
      return edge->factor_id_ == factor_id;
    });
    if (it == edges_.end()) {
      throw std::runtime_error("Edge with the given factor_id not found in VariableNode::sendMessage");
    }
    sendMessage(static_cast<size_t>(std::distance(edges_.begin(), it)));
  }

  template <int Dim_, typename EstimateType_, typename EdgeType_>
  std::shared_ptr<EdgeBase> VariableNode<Dim_, EstimateType_, EdgeType_>::addDanglingEdge(const std::string& factor_id,
                                                                                         int remote_id) {
    auto edge          = std::make_shared<EdgeType_>();
    edge->variable_id_ = id_;
    edge->factor_id_   = factor_id;
    edge->side_        = EdgeSide::VARIABLE_SIDE;
    edge->remote_id_   = remote_id;

    // Same initial outgoing message as a local edge; nothing received yet
    edge->variable_to_factor_message_.mu_     = mu_;
    edge->variable_to_factor_message_.Lambda_ = Sigma_.inverse();
    edge->factor_to_variable_message_.mu_     = mu_;
    edge->factor_to_variable_message_.Lambda_ = MatrixLieDim::Zero();

    edges_.push_back(edge);
    return edge;
  }

  template <int Dim_, typename EstimateType_, typename EdgeType_>
  void VariableNode<Dim_, EstimateType_, EdgeType_>::removeEdge(const std::string& factor_id) {
    const auto it = std::find_if(edges_.begin(), edges_.end(), [&](const auto& edge) { return edge->factor_id_ == factor_id; });
    if (it == edges_.end()) {
      throw std::runtime_error("VariableNode::removeEdge: " + id_ + " has no edge to " + factor_id);
    }
    edges_.erase(it);
  }

  template <int Dim_, typename EstimateType_, typename EdgeType_>
  void VariableNode<Dim_, EstimateType_, EdgeType_>::sendAllMessages() {
    // Serial: this already runs inside the solver's loop over variables
    const size_t k = edges_.size();
    if (send_belief_) {
      std::vector<int> all(k, 1);
      sendMessages(all);
      return;
    }
    if (k <= 1) {
      for (size_t i = 0; i < k; ++i) {
        sendMessage(i); // nothing to multiply; see sendMessage
      }
      return;
    }

    // Every incoming message in the tangent space at the current mean, once, and their sum.
    // Each outgoing message is then the sum minus its own edge: O(k) instead of O(k^2).
    tangent_eta_in_.resize(k);
    tangent_Lambda_in_.resize(k);
    VectorLieDim total_eta    = VectorLieDim::Zero();
    MatrixLieDim total_Lambda = MatrixLieDim::Zero();
    for (size_t i = 0; i < k; ++i) {
      const auto& message   = edges_[i]->factor_to_variable_message_;
      tangent_Lambda_in_[i] = transportPrecision<typename EdgeType_::VariableType_>(mu_, message.mu_, message.Lambda_);
      tangent_eta_in_[i]    = message.Lambda_ * boxminus(message.mu_);
      total_eta += tangent_eta_in_[i];
      total_Lambda += tangent_Lambda_in_[i];
    }
    message_update_.prepare(id_);
    for (size_t j = 0; j < k; ++j) {
      if (message_update_.drop()) {
        continue; // dropped: the previous message stays
      }
      const MatrixLieDim Lambda = total_Lambda - tangent_Lambda_in_[j];
      // Regularized only to solve; the message keeps the true precision
      const VectorLieDim tau = SpdSolver<Dim_>(MatrixLieDim(Lambda + MatrixLieDim::Identity() * kInformationRegularization))
                                 .solve(VectorLieDim(total_eta - tangent_eta_in_[j]));

      const EstimateType_ mu_out                        = boxplus(tau);
      edges_[j]->variable_to_factor_message_.Lambda_    = transportPrecision<typename EdgeType_::VariableType_>(mu_out, mu_, tau, Lambda);
      edges_[j]->variable_to_factor_message_.mu_        = mu_out;
      edges_[j]->variable_to_factor_message_.processed_ = false;
      edges_[j]->variable_to_factor_message_.belief_mu_  = mu_;
      edges_[j]->variable_to_factor_message_.has_belief_ = true;
    }
  }

  template <int Dim_, typename EstimateType_, typename EdgeType_>
  void VariableNode<Dim_, EstimateType_, EdgeType_>::updateMuAndSigma() {
    VectorLieDim tangent_eta         = VectorLieDim::Zero();
    MatrixLieDim tangent_Lambdaprime = MatrixLieDim::Zero();
    VectorLieDim tangent_delta_mu    = VectorLieDim::Zero();

    // If there are no factors connected to the variable node, skip the update
    if (edges_.size() == 0) {
      return;
    }

    // Multiply inward messages from other factors
    for (size_t i = 0; i < edges_.size(); ++i) {
      // Access messages from edges (replace with appropriate getter methods based on your edge structure)
      const MatrixLieDim& Lambdaprime_inward = edges_[i]->factor_to_variable_message_.Lambda_;
      const EstimateType_ mu_inward          = edges_[i]->factor_to_variable_message_.mu_;

      // Handle the pose
      VectorLieDim tangent_delta_mu_inward = boxminus(mu_inward);

      // Compute the Lambda and eta (precision moved to the tangent space at mu_: LieOptions)
      MatrixLieDim tangent_Lambda_inward = transportPrecision<typename EdgeType_::VariableType_>(mu_, mu_inward, Lambdaprime_inward);
      VectorLieDim tangent_eta_inward    = tangent_Lambda_inward * tangent_delta_mu_inward;

      tangent_eta += tangent_eta_inward;
      tangent_Lambdaprime += tangent_Lambda_inward;
    }
    // Regularize to solve; the belief's covariance is that of the regularized precision
    tangent_Lambdaprime += MatrixLieDim::Identity() * kInformationRegularization;
    const SpdSolver<Dim_> solver(tangent_Lambdaprime);
    tangent_delta_mu = solver.solve(tangent_eta);

    // compute and update the estimate in the manifold
    mu_    = boxplus(tangent_delta_mu);
    Sigma_ = solver.solve(MatrixLieDim(MatrixLieDim::Identity()));
    symmetrize(Sigma_);
  }

  template <int Dim_, typename EstimateType_, typename EdgeType_>
  bool VariableNode<Dim_, EstimateType_, EdgeType_>::hasUnprocessedMessages() const {
    for (const auto& edge : edges_) {
      if (!edge->factor_to_variable_message_.processed_) {
        return true;
      }
    }
    return false;
  }

  template <int Dim_, typename EstimateType_, typename EdgeType_>
  void VariableNode<Dim_, EstimateType_, EdgeType_>::markMessagesProcessed() {
    for (const auto& edge : edges_) {
      edge->factor_to_variable_message_.processed_ = true;
    }
  }

  template <int Dim_, typename EstimateType_, typename EdgeType_>
  void VariableNode<Dim_, EstimateType_, EdgeType_>::beliefSums(MatrixLieDim& Lambda, VectorLieDim& eta) {
    Lambda.setZero();
    eta.setZero();
    for (const auto& edge : edges_) {
      const auto& m = edge->factor_to_variable_message_;
      const MatrixLieDim L = transportPrecision<typename EdgeType_::VariableType_>(mu_, m.mu_, m.Lambda_);
      Lambda += L;
      eta += L * boxminus(m.mu_);
    }
  }

  template <int Dim_, typename EstimateType_, typename EdgeType_>
  void VariableNode<Dim_, EstimateType_, EdgeType_>::sendMessages(const std::vector<int>& mask) {
    if (!send_belief_) {
      for (size_t i = 0; i < mask.size() && i < edges_.size(); ++i) {
        if (mask[i]) {
          sendMessage(i);
        }
      }
      return;
    }
    if (edges_.empty() || (edges_.size() == 1 && initial_guess_is_prior_)) {
      return;
    }
    // One belief for every selected edge (dropout still per edge, as in sendMessage)
    MatrixLieDim Lambda;
    VectorLieDim eta;
    beliefSums(Lambda, eta);
    message_update_.prepare(id_);
    for (size_t i = 0; i < mask.size() && i < edges_.size(); ++i) {
      if (!mask[i]) {
        continue;
      }
      if (edges_.size() > 1 && message_update_.drop()) {
        continue;
      }
      writeBelief(edges_[i]->variable_to_factor_message_, Lambda, eta);
    }
  }

  // ---- Generic rule inputs (core/generic_scheduler.h): every input is invariant to the units and
  // to linear reparametrisations of the tangent space, and normalised per dimension, so variables
  // of any type and size give comparable values.
  template <int Dim_, typename EstimateType_, typename EdgeType_>
  void VariableNode<Dim_, EstimateType_, EdgeType_>::ruleReset() {
    rule_has_logdet_ = false;
    for (auto& e : edges_) {
      e->last_sent_mu_    = mu_;
      e->last_sent_tick_  = 0;
      e->incoming_gap_    = 0;
      e->has_last_sent_   = true;
      e->v_osc_has_prev_  = false;
      e->v_osc_has_delta_ = false;
      e->v_osc_cos_       = 0.0;
      e->v_osc_ratio_     = 0.0;
      e->v_osc_component_.setZero();
      e->component_inputs_.clear();
      e->last_sent_sigma_ = Sigma_;
      e->kl_ref_valid_    = false;
      e->kl_cav_valid_    = false;
      e->prev_energy_     = -1.0;
      e->hist_moved_ema_     = -1.0;
      e->hist_trend_ema_     = 0.0;
      e->hist_trend_abs_ema_ = 0.0;
    }
  }

  // Cavity of edge j (the belief without its incoming message), as the solver computes the message
  // (sendAllMessages): precision and information in the tangent space at mu, from the belief sums.
  // Returns the cavity mean, its precision (regularised) and log det of its covariance.
  template <typename V, typename EstimateT, typename MatrixT, typename VectorT, typename MessageT>
  static void cavityOf(const EstimateT& mu, const MatrixT& Lambda_total, const VectorT& eta_total, const MessageT& m,
                       double reg, EstimateT& cav_mu, MatrixT& cav_lambda, double& cav_logdet) {
    const MatrixT L = transportPrecision<V>(mu, m.mu_, m.Lambda_);
    const VectorT eta = L * V::boxminus(mu, m.mu_);
    cav_lambda        = Lambda_total - L + MatrixT::Identity() * reg;
    const Eigen::LDLT<MatrixT> ldlt(cav_lambda);
    const VectorT tau = ldlt.solve(VectorT(eta_total - eta));
    cav_mu            = V::boxplus(mu, tau);
    cav_logdet        = -ldlt.vectorD().array().max(1e-300).log().sum(); // log det of the covariance
  }

  template <int Dim_, typename EstimateType_, typename EdgeType_>
  void VariableNode<Dim_, EstimateType_, EdgeType_>::ruleRecordSends(const std::vector<int>& mask, long tick) {
    using V = typename EdgeType_::VariableType_;
    bool any = false;
    for (size_t j = 0; j < mask.size() && j < edges_.size(); ++j) {
      any = any || mask[j];
    }
    MatrixLieDim Lt;
    VectorLieDim et;
    if (any) {
      beliefSums(Lt, et);
    }
    for (size_t j = 0; j < mask.size() && j < edges_.size(); ++j) {
      if (mask[j]) {
        auto& e           = edges_[j];
        cavityOf<V>(mu_, Lt, et, e->factor_to_variable_message_, kInformationRegularization, e->last_sent_cav_mu_,
                    e->kl_cav_lambda_, e->kl_cav_logdet_);
        e->kl_cav_valid_ = true;
        e->incoming_gap_  = tick - e->last_sent_tick_;
        e->last_sent_tick_ = tick;
        e->last_sent_mu_  = mu_;
        e->last_sent_sigma_ = Sigma_; // the belief sent (called before the belief update)
        e->kl_ref_valid_    = false;
        e->has_last_sent_ = true;
      }
    }
  }

  template <int Dim_, typename EstimateType_, typename EdgeType_>
  void VariableNode<Dim_, EstimateType_, EdgeType_>::ruleInputs(long tick, const double* edge_energy, double* out,
                                                               int stride, bool update, const char* active) {
    using V            = typename EdgeType_::VariableType_;
    constexpr double e = 1e-9;
    // Changes below a thousandth of a standard deviation (per dimension) count as no change: below
    // that the inputs would only carry numerical noise
    constexpr double kSigmaFloor = 1e-3;
    const size_t k     = edges_.size();
    const double d     = static_cast<double>(Dim_);
    // belief precision and log det of the covariance from one factorisation (LDLT if not positive definite)
    MatrixLieDim Lb;
    double logdet;
    {
      const Eigen::LLT<MatrixLieDim> llt(Sigma_);
      if (llt.info() == Eigen::Success) {
        Lb     = llt.solve(MatrixLieDim(MatrixLieDim::Identity()));
        logdet = 2.0 * llt.matrixL().toDenseMatrix().diagonal().array().max(1e-150).log().sum();
      } else {
        const Eigen::LDLT<MatrixLieDim> ldlt(Sigma_);
        Lb     = ldlt.solve(MatrixLieDim(MatrixLieDim::Identity()));
        logdet = ldlt.vectorD().array().max(1e-300).log().sum();
      }
    }
    const double trend  = rule_has_logdet_ ? std::clamp((logdet - rule_prev_logdet_) / d, -5.0, 5.0) : 0.0;
    MatrixLieDim cav_Lt;
    VectorLieDim cav_et;
    beliefSums(cav_Lt, cav_et);
    std::vector<double> moved(k), share(k);
    double mean_moved = 0.0, mean_energy = 0.0;
    size_t k_active = 0;
    for (size_t j = 0; j < k; ++j) {
      // an edge added after ruleReset (graphs that grow): it starts now, as if it had just sent
      if (!edges_[j]->has_last_sent_) {
        edges_[j]->last_sent_mu_   = mu_;
        edges_[j]->last_sent_sigma_ = Sigma_;
        edges_[j]->kl_ref_valid_    = false;
        edges_[j]->kl_cav_valid_    = false;
        edges_[j]->last_sent_tick_ = tick;
        edges_[j]->has_last_sent_  = true;
      }
    }
    for (size_t j = 0; j < k; ++j) {
      const VectorLieDim dm = V::boxminus(edges_[j]->last_sent_mu_, mu_);
      moved[j]              = std::sqrt(std::max(dm.dot(Lb * dm), 0.0) / d);
      if (active == nullptr || active[j]) {
        mean_moved += moved[j];
        mean_energy += edge_energy[j];
        ++k_active;
      }
    }
    mean_moved  = k_active ? mean_moved / k_active : 0.0;
    mean_energy = std::max(k_active ? mean_energy / k_active : 0.0, e);
    for (size_t j = 0; j < k; ++j) {
      auto& ed              = *edges_[j];
      const auto& msg       = ed.factor_to_variable_message_;
      const double E        = edge_energy[j];
      const double E_prev   = ed.prev_energy_ < 0 ? E : ed.prev_energy_;
      // the share of the belief's precision this message provides, per dimension
      // tr(Lm Sigma) = sum of the elementwise product (both symmetric)
      share[j] = msg.Lambda_.cwiseProduct(Sigma_).sum() / d;
      // disagreement in the combined metric (Sigma_b + Sigma_m)^-1 = Lm (Lb + Lm)^-1 Lb:
      // r^T Lm (Lb + Lm)^-1 Lb r, one factorisation and one vector solve
      const VectorLieDim r   = V::boxminus(mu_, msg.mu_);
      const VectorLieDim Lbr = Lb * r;
      const VectorLieDim Lmr = msg.Lambda_ * r;
      const SpdSolver<Dim_> combined(MatrixLieDim(Lb + msg.Lambda_ + MatrixLieDim::Identity() * kInformationRegularization));
      const double disagree = std::sqrt(std::max(Lmr.dot(combined.solve(Lbr)), 0.0) / d);
      // oscillation of the incoming message, in the belief's metric
      double cos = ed.v_osc_cos_, ratio = ed.v_osc_ratio_;
      VectorLieDim osc_c = ed.v_osc_component_;
      // an unchanged message (bitwise) has a zero change: nothing to update, skip the logarithm
      const bool msg_changed = !ed.v_osc_has_prev_ ||
                               std::memcmp(&ed.v_osc_prev_, &msg.mu_, sizeof(msg.mu_)) != 0;
      if (ed.v_osc_has_prev_ && msg_changed) {
        const VectorLieDim delta = V::boxminus(ed.v_osc_prev_, msg.mu_);
        const double sq          = delta.dot(Lb * delta);
        if (std::sqrt(std::max(sq, 0.0) / d) > kSigmaFloor) {
          if (ed.v_osc_has_delta_) {
            const double a = std::sqrt(std::max(ed.v_osc_delta_.dot(Lb * ed.v_osc_delta_), 0.0));
            if (a > 1e-12) {
              const double inner = delta.dot(Lb * ed.v_osc_delta_);
              cos                = inner / (std::sqrt(sq) * a);
              ratio              = std::clamp(inner / (a * a), -5.0, 5.0);
            }
            // per component, in its own standard deviations: changes below kSigmaFloor count as none
            for (long c = 0; c < delta.size(); ++c) {
              const double sc = std::sqrt(std::max(Sigma_(c, c), 1e-300));
              const double x = delta[c] / sc, y = ed.v_osc_delta_[c] / sc;
              osc_c[c] = (std::abs(x) > kSigmaFloor || std::abs(y) > kSigmaFloor) ? 2.0 * x * y / (x * x + y * y) : 0.0;
            }
          }
          if (update) {
            ed.v_osc_delta_     = delta;
            ed.v_osc_has_delta_ = true;
            ed.v_osc_cos_       = cos;
            ed.v_osc_ratio_     = ratio;
            ed.v_osc_component_ = osc_c;
          }
        }
      }
      double* row = out + j * stride;
      row[0] = std::log1p(static_cast<double>(std::max(tick - ed.last_sent_tick_, 0L)));
      row[1] = std::log(moved[j] + kSigmaFloor);
      row[2] = std::log((moved[j] + kSigmaFloor) / (mean_moved + kSigmaFloor));
      row[3] = trend;
      row[4] = std::log((E + e) / mean_energy);
      row[5] = std::clamp(std::log((E + e) / (E_prev + e)), -5.0, 5.0);
      row[6] = std::log(share[j] + e);
      row[7] = std::log(disagree + kSigmaFloor);
      row[8] = cos;
      // memory over many ticks (exponential averages, ~10 ticks for the change, ~20 for the energy)
      constexpr double kMovedRate = 0.1, kTrendRate = 0.05;
      const double moved_ema = ed.hist_moved_ema_ < 0 ? moved[j] : ed.hist_moved_ema_;
      row[9]                 = std::log((moved[j] + kSigmaFloor) / (moved_ema + kSigmaFloor));
      const double tr        = row[5];
      const double tr_ema    = (1 - kTrendRate) * ed.hist_trend_ema_ + kTrendRate * tr;
      const double tr_abs    = (1 - kTrendRate) * ed.hist_trend_abs_ema_ + kTrendRate * std::abs(tr);
      row[10]                = tr_abs > 1e-12 ? tr_ema / tr_abs : 0.0;
      row[11]                = std::log(tr_abs + 1e-6);
      // per tangent component c (kVariableComponentInputNames): movement since the last send in c's
      // standard deviations relative to the edge's movement (per dim); oscillation of c in the incoming
      // message; the share of c's precision this message provides (Lambda_m,cc Sigma_cc)
      {
        const VectorLieDim dmj = V::boxminus(ed.last_sent_mu_, mu_);
        auto& ci = ed.component_inputs_;
        ci.resize(static_cast<size_t>(kComponentInputs) * dmj.size());
        for (long c = 0; c < dmj.size(); ++c) {
          const double var = std::max(Sigma_(c, c), 1e-300);
          ci[kComponentInputs * c + 0] = std::log((std::abs(dmj[c]) / std::sqrt(var) + kSigmaFloor) / (moved[j] + kSigmaFloor));
          ci[kComponentInputs * c + 1] = osc_c[c];
          ci[kComponentInputs * c + 2] = std::log(std::max(msg.Lambda_(c, c), 0.0) * var + e);
        }
      }
      // KL(N(mu, Sigma) || N(mu_s, Sigma_s)) per dimension, (mu_s, Sigma_s) the belief at the last send:
      // 0.5 (tr(Lambda_s Sigma) - d + log det Sigma_s - log det Sigma + dm^T Lambda_s dm)
      if (!ed.kl_ref_valid_) {
        const Eigen::LLT<MatrixLieDim> llt_s(ed.last_sent_sigma_);
        if (llt_s.info() == Eigen::Success) {
          ed.kl_ref_lambda_ = llt_s.solve(MatrixLieDim(MatrixLieDim::Identity()));
          ed.kl_ref_logdet_ = 2.0 * llt_s.matrixL().toDenseMatrix().diagonal().array().max(1e-150).log().sum();
        } else {
          const Eigen::LDLT<MatrixLieDim> ldlt_s(ed.last_sent_sigma_);
          ed.kl_ref_lambda_ = ldlt_s.solve(MatrixLieDim(MatrixLieDim::Identity()));
          ed.kl_ref_logdet_ = ldlt_s.vectorD().array().max(1e-300).log().sum();
        }
        ed.kl_ref_valid_ = true;
      }
      const VectorLieDim dm_s = V::boxminus(ed.last_sent_mu_, mu_);
      const double kl = 0.5 * (ed.kl_ref_lambda_.cwiseProduct(Sigma_).sum() - d + ed.kl_ref_logdet_ - logdet +
                               dm_s.dot(ed.kl_ref_lambda_ * dm_s));
      row[12] = std::log(std::max(kl, 0.0) / d + 1e-6);
      row[13] = ratio;
      {
        EstimateType_ cmu;
        MatrixLieDim clam;
        double clogdet;
        cavityOf<V>(mu_, cav_Lt, cav_et, msg, kInformationRegularization, cmu, clam, clogdet);
        if (!ed.kl_cav_valid_) { // no send recorded yet: the reference is the cavity now
          ed.last_sent_cav_mu_ = cmu;
          ed.kl_cav_lambda_    = clam;
          ed.kl_cav_logdet_    = clogdet;
          ed.kl_cav_valid_     = true;
        }
        const MatrixLieDim csig  = clam.ldlt().solve(MatrixLieDim(MatrixLieDim::Identity()));
        const VectorLieDim cd    = V::boxminus(ed.last_sent_cav_mu_, cmu);
        const double kl_cav = 0.5 * (ed.kl_cav_lambda_.cwiseProduct(csig).sum() - d + ed.kl_cav_logdet_ - clogdet +
                                     cd.dot(ed.kl_cav_lambda_ * cd));
        row[14] = std::log(std::max(kl_cav, 0.0) / d + 1e-6);
      }
      if (update) {
        ed.hist_moved_ema_     = (1 - kMovedRate) * moved_ema + kMovedRate * moved[j];
        ed.hist_trend_ema_     = tr_ema;
        ed.hist_trend_abs_ema_ = tr_abs;
      }
      if (update) {
        ed.prev_energy_    = E;
        if (msg_changed) {
          ed.v_osc_prev_ = msg.mu_;
        }
        ed.v_osc_has_prev_ = true;
      }
    }
    if (update) {
      rule_prev_logdet_ = logdet;
      rule_has_logdet_  = true;
    }
  }

} // namespace gsolver
