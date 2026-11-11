#pragma once

// A learned local send rule (rls_gbp ES rules) run inside the solver.
//
// Every variable decides, for each of its edges, whether to send its variable-to-factor message
// this tick, from inputs it has locally: its belief, what it sent on that edge and when, the
// factor-to-variable message on that edge, and the scalar energy each factor attaches to its
// replies (0.5 e^T Sigma^-1 e at the factor's incoming messages). The decision is a logistic or
// small MLP function of those inputs, sampled. One tick is then: every variable sends on its
// selected edges, every factor that received something new replies, every belief absorbs the
// replies (FactorNodeBase incremental API).
//
// Inputs (kFeatureNames, the order of rls_gbp.marl.rule.FEATURES), for edge (v, f):
//   residual   = ||log(T_last_sent^-1 T_now)||, T_last_sent = v's belief when (v, f) last sent
//   staleness  = ticks since (v, f) last sent
//   energy e_f; mean over v's factors; e_f at the previous tick
//   std        = sqrt(trace Sigma_v); std at the first tick and at the previous tick
//   disagreement = ||log(T_v^-1 mu_msg)||, msg_std = sqrt(trace Lambda_msg^-1)
// The rule sees the subset given by `features` (indices into kFeatureNames).

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <memory>
#include <random>
#include <string>
#include <unordered_map>
#include <vector>

#include <Eigen/Dense>

#include <gsolver/graph/factor_graph.h>

namespace gsolver {

  inline const std::vector<std::string> kFeatureNames = {
    "log_residual",          "log1p_staleness",       "log_energy",          "log_relative_energy",
    "energy_trend",          "log_uncertainty",       "log_residual_over_std", "degree",
    "log_disagreement",      "log_message_std",       "log_relative_residual", "log_std_progress",
    "log_std_trend",         "log_relative_message_std", "log_normalised_disagreement"};

  struct SendRule {
    std::vector<int> features; // indices into kFeatureNames
    // logistic: p = sigmoid(intercept + coef . x)
    Eigen::VectorXd coef;
    double intercept = 0.0;
    // mlp (when W1 is non-empty): z = (x - mu) / sd, p = sigmoid(w2 . tanh(W1 z + b1) + b2)
    Eigen::VectorXd mu, sd, b1, w2;
    Eigen::MatrixXd W1;
    double b2 = 0.0;

    double probability(const Eigen::VectorXd& x) const {
      double logit;
      if (W1.size() == 0) {
        logit = intercept + coef.dot(x);
      } else {
        const Eigen::VectorXd z = (x - mu).cwiseQuotient(sd);
        logit = w2.dot((W1 * z + b1).array().tanh().matrix()) + b2;
      }
      return 1.0 / (1.0 + std::exp(-logit));
    }
  };

  struct TickCounts {
    size_t variable_to_factor = 0;
    size_t factor_to_variable = 0;
  };

  template <typename VariableType>
  class RuleScheduler {
  public:
    using EstimateType = typename VariableType::EstimateType;

    RuleScheduler(FactorGraph& graph, SendRule rule, unsigned seed)
      : graph_(graph), rule_(std::move(rule)), rng_(seed) {
      for (size_t k = 0; k < graph_.factor_nodes_.size(); ++k) {
        factor_index_[graph_.factor_nodes_[k]->id_] = k;
      }
      for (auto& node : graph_.variable_nodes_) {
        auto v = std::dynamic_pointer_cast<VariableType>(node);
        if (!v) {
          throw std::runtime_error("RuleScheduler: variable " + node->id_ + " is not of the scheduler's type");
        }
        variables_.push_back(v);
        std::vector<size_t> fs;
        for (auto& e : v->edges_) {
          fs.push_back(factor_index_.at(e->factor_id_));
        }
        edge_factor_.push_back(fs);
      }
      reset();
    }

    // Episode start: nothing sent yet (staleness 0, last-sent pose = current belief)
    void reset() {
      ticks_ = 0;
      last_sent_tick_.assign(variables_.size(), {});
      last_sent_pose_.assign(variables_.size(), {});
      for (size_t i = 0; i < variables_.size(); ++i) {
        last_sent_tick_[i].assign(variables_[i]->edges_.size(), 0);
        last_sent_pose_[i].assign(variables_[i]->edges_.size(), variables_[i]->mu_);
      }
      prev_energy_.clear();
      start_std_.clear();
      prev_std_.clear();
    }

    size_t numberOfEdges() const {
      size_t n = 0;
      for (auto& v : variables_) {
        n += v->edges_.size();
      }
      return n;
    }

    // All kFeatureNames for every edge (variables in graph order, edges in variable order).
    // Stateful like rls_gbp's RuleFeatures: the previous energies and stds become the current ones.
    Eigen::MatrixXd features(bool update_state = true) {
      constexpr double kEps = 1e-9;
      const size_t n_features = kFeatureNames.size();
      Eigen::MatrixXd X(numberOfEdges(), n_features);
      std::vector<double> energy(graph_.factor_nodes_.size());
      for (size_t k = 0; k < energy.size(); ++k) {
        energy[k] = graph_.factor_nodes_[k]->energy();
      }
      std::vector<double> std_now(variables_.size());
      size_t row = 0;
      for (size_t i = 0; i < variables_.size(); ++i) {
        const auto& v   = variables_[i];
        const size_t k  = v->edges_.size();
        const double sd = std::sqrt(std::max(v->Sigma_.trace(), kEps));
        std_now[i]      = sd;
        const double prev_sd  = prev_std_.empty() ? sd : prev_std_[i];
        const double start_sd = start_std_.empty() ? sd : start_std_[i];
        double mean_energy = 0.0, mean_residual = 0.0, mean_log_msg_std = 0.0;
        std::vector<double> residual(k), msg_std(k);
        for (size_t j = 0; j < k; ++j) {
          mean_energy += energy[edge_factor_[i][j]];
          residual[j] = std::max(VariableType::boxminus(last_sent_pose_[i][j], v->mu_).norm(), 0.0);
          mean_residual += residual[j];
          const auto& Lambda = v->edges_[j]->factor_to_variable_message_.Lambda_;
          msg_std[j]         = std::sqrt(std::max(Lambda.inverse().trace(), kEps));
          mean_log_msg_std += std::log(msg_std[j]);
        }
        mean_energy   = std::max(k ? mean_energy / k : 0.0, kEps);
        mean_residual = k ? mean_residual / k : 0.0;
        const double geo_msg_std = k ? std::exp(mean_log_msg_std / k) : sd;
        for (size_t j = 0; j < k; ++j, ++row) {
          const size_t f    = edge_factor_[i][j];
          const double e    = energy[f];
          const double prev = prev_energy_.empty() ? e : prev_energy_[f];
          const double disagreement = VariableType::boxminus(v->mu_, v->edges_[j]->factor_to_variable_message_.mu_).norm();
          const double staleness    = std::max(static_cast<double>(ticks_ - last_sent_tick_[i][j]), 0.0);
          X(row, 0)  = std::log(residual[j] + 1e-6);
          X(row, 1)  = std::log1p(staleness);
          X(row, 2)  = std::log(e + kEps);
          X(row, 3)  = std::log((e + kEps) / mean_energy);
          X(row, 4)  = std::clamp(std::log((e + kEps) / (prev + kEps)), -5.0, 5.0);
          X(row, 5)  = std::log(sd);
          X(row, 6)  = std::log(residual[j] / sd + 1e-6);
          X(row, 7)  = static_cast<double>(k);
          X(row, 8)  = std::log(disagreement + 1e-6);
          X(row, 9)  = std::log(msg_std[j]);
          X(row, 10) = std::log((residual[j] + 1e-9) / (mean_residual + 1e-9));
          X(row, 11) = std::log(sd / start_sd);
          X(row, 12) = std::clamp(std::log(sd / prev_sd), -5.0, 5.0);
          X(row, 13) = std::log(msg_std[j] / geo_msg_std);
          X(row, 14) = std::log(disagreement / std::sqrt(sd * sd + msg_std[j] * msg_std[j]) + 1e-6);
        }
      }
      if (update_state) {
        prev_energy_ = energy;
        prev_std_    = std_now;
        if (start_std_.empty()) {
          start_std_ = std_now;
        }
      }
      return X;
    }

    // Send probability of every edge, from all features (rows of features())
    Eigen::VectorXd probabilities(const Eigen::MatrixXd& X) const {
      Eigen::VectorXd p(X.rows());
      Eigen::VectorXd x(rule_.features.size());
      for (Eigen::Index r = 0; r < X.rows(); ++r) {
        for (size_t c = 0; c < rule_.features.size(); ++c) {
          x[c] = X(r, rule_.features[c]);
        }
        p[r] = rule_.probability(x);
      }
      return p;
    }

    // One tick with the rule. uniforms: one U(0,1) draw per edge (edge sends iff u < p); empty to
    // draw them here. Returns the messages sent.
    TickCounts tick(const Eigen::VectorXd& uniforms = Eigen::VectorXd()) {
      const Eigen::VectorXd p = probabilities(features(true));
      std::vector<std::vector<int>> masks(variables_.size());
      std::uniform_real_distribution<double> unit(0.0, 1.0);
      size_t row = 0;
      for (size_t i = 0; i < variables_.size(); ++i) {
        masks[i].resize(variables_[i]->edges_.size());
        for (auto& m : masks[i]) {
          const double u = uniforms.size() ? uniforms[row] : unit(rng_);
          m              = u < p[row] ? 1 : 0;
          ++row;
        }
      }
      return tickMasks(masks);
    }

    // One tick with given send decisions (masks[i][j]: variable i, edge j), as rls_gbp's gbp_tick,
    // with the scheduler's bookkeeping (staleness, last-sent poses)
    TickCounts tickMasks(const std::vector<std::vector<int>>& masks) {
      TickCounts counts;
      ++ticks_;
      for (size_t i = 0; i < variables_.size(); ++i) {
        auto& v = variables_[i];
        const EstimateType before = v->mu_;
        for (size_t j = 0; j < masks[i].size(); ++j) {
          if (masks[i][j]) {
            v->sendMessage(j);
            last_sent_tick_[i][j] = ticks_;
            last_sent_pose_[i][j] = before;
            ++counts.variable_to_factor;
          }
        }
        v->markMessagesProcessed();
      }
      for (auto& f : graph_.factor_nodes_) {
        f->updateFactorIfNew();
        counts.factor_to_variable += f->sendAllMessagesIfNew();
      }
      for (auto& v : variables_) {
        v->updateMuAndSigma();
      }
      return counts;
    }

    long ticks() const {
      return ticks_;
    }

  private:
    FactorGraph& graph_;
    SendRule rule_;
    std::mt19937_64 rng_;
    std::vector<std::shared_ptr<VariableType>> variables_;
    std::unordered_map<std::string, size_t> factor_index_;
    std::vector<std::vector<size_t>> edge_factor_;
    long ticks_ = 0;
    std::vector<std::vector<long>> last_sent_tick_;
    std::vector<std::vector<EstimateType>> last_sent_pose_;
    std::vector<double> prev_energy_, start_std_, prev_std_;
  };

} // namespace gsolver
