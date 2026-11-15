#pragma once

// Learned local rules for any factor graph ("v4" inputs): variables decide whether to send and how
// much to damp what they receive; factors decide whether to reply and how much to damp what they
// receive. Unlike RuleScheduler (one variable type per graph, the "v3" inputs), it works through
// the node and edge base classes, so variables of different types and sizes (an SE(3) pose and a
// 2D point on the same factor) are handled the same way.
//
// Every input is invariant to the units and to linear reparametrisations of each variable's
// tangent space, and normalised per dimension (Mahalanobis distances per dimension, ratios,
// cosines in a precision metric):
//   variable, per edge (v, f)                 kVariableInputNamesV4
//     ticks since v last sent on the edge; how far v's belief moved since then (per dim, belief
//     metric) and relative to v's other edges; v's uncertainty trend (mean log-eigenvalue change);
//     f's energy per error dim relative to v's other factors, and its trend; the share of v's
//     precision f's message provides (per dim); f's message vs v's belief in the combined metric
//     (per dim); oscillation of f's message (cosine of its last two changes, belief metric)
//   factor, per edge (f, v)                   kFactorInputNamesV4
//     ticks since f last replied to v; news for v: mean over f's other variables of how far their
//     messages moved since that reply (each per dim in its own precision); whether v sent this
//     tick; f's Mahalanobis distance per error dim; f's energy trend and change since the reply to
//     v; oscillation and change (per dim) of v's incoming message; ticks between v's last messages

#include <memory>
#include <stdexcept>
#include <random>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include <Eigen/Dense>

#include <gsolver/core/rule_scheduler.h> // LogisticHead, FactorRule, TickCounts
#include <gsolver/graph/factor_graph.h>

namespace gsolver {

  inline const std::vector<std::string> kVariableInputNamesV4 = {
    "v_log1p_staleness",     "v_log_moved",   "v_log_relative_moved", "v_uncertainty_trend", "v_log_relative_energy",
    "v_energy_trend",        "v_log_message_share", "v_log_disagreement", "v_oscillation",
    // memory over many ticks: the change against the edge's usual change; energy settling (-1:
    // steadily down, +1: steadily up, 0: up and down); log of the usual size of the energy changes
    "v_log_moved_vs_history", "v_energy_settling", "v_log_energy_fluctuation",
    // the edge's message cost (setEdgeWeights; 1 without weights): e.g. 1 between robots, 0 inside one
    "v_edge_cost"};

  inline const std::vector<std::string> kFactorInputNamesV4 = {
    "f_log1p_reply_staleness", "f_log_news",          "f_target_sent",       "f_log_mahalanobis", "f_energy_trend",
    "f_log_energy_since_reply", "f_incoming_oscillation", "f_log_incoming_change", "f_log1p_incoming_gap"};

  // Variable rule: send head (probability; inactive: send with fixed_send) and incoming damping head
  struct VariableRule {
    LogisticHead send;
    LogisticHead damping;
    double fixed_send = 1.0;
    // Staleness cap: an edge silent for max_silence ticks sends regardless of the rule (0: off).
    // Every edge then sends at least once every max_silence ticks: bounded-delay asynchronous GBP
    long max_silence = 0;
  };

  class GenericScheduler {
  public:
    GenericScheduler(FactorGraph& graph, VariableRule variable_rule, FactorRule factor_rule, unsigned seed)
      : graph_(graph), vrule_(std::move(variable_rule)), frule_(std::move(factor_rule)), rng_(seed) {
      std::unordered_map<std::string, size_t> factor_index;
      for (size_t k = 0; k < graph_.factor_nodes_.size(); ++k) {
        factor_index[graph_.factor_nodes_[k]->id_] = k;
      }
      edge_factor_.resize(graph_.variable_nodes_.size());
      factor_edges_.assign(graph_.factor_nodes_.size(), {});
      for (size_t i = 0; i < graph_.variable_nodes_.size(); ++i) {
        auto& v = graph_.variable_nodes_[i];
        for (size_t j = 0; j < v->numberOfEdges(); ++j) {
          const size_t f = factor_index.at(v->getEdgeAt(j)->factor_id_);
          edge_factor_[i].push_back(f);
          factor_edges_[f].push_back(v->getEdgeAt(j));
          factor_edge_rows_.resize(graph_.factor_nodes_.size());
          factor_edge_rows_[f].push_back(edge_slot_.size());
          edge_slot_.push_back({});
        }
      }
      reset();
    }

    void reset() {
      ticks_ = 0;
      f2v_seen_.assign(numberOfEdges(), 0);
      for (size_t f = 0; f < factor_edges_.size(); ++f) {
        for (size_t p = 0; p < factor_edges_[f].size(); ++p) {
          f2v_seen_[factor_edge_rows_[f][p]] = factor_edges_[f][p]->f2v_sent_;
        }
      }
      for (auto& v : graph_.variable_nodes_) {
        v->ruleReset();
      }
      const size_t nf = graph_.factor_nodes_.size();
      f_energy_prev_.assign(nf, -1.0);
      f_reply_tick_.assign(nf, {});
      f_energy_reply_.assign(nf, {});
      for (size_t f = 0; f < nf; ++f) {
        f_reply_tick_[f].assign(factor_edges_[f].size(), 0);
        f_energy_reply_[f].assign(factor_edges_[f].size(), -1.0);
        for (auto& e : factor_edges_[f]) {
          e->ruleResetFactorSide(factor_edges_[f].size());
        }
      }
    }

    size_t numberOfEdges() const {
      return edge_slot_.size();
    }

    long ticks() const {
      return ticks_;
    }

    // Per-edge message costs (one per (variable, edge) row, as in variableFeatures; empty: 1 each).
    // weightedCounts(): the cost of the last tick's messages, v2f and f2v, e.g. weight 1 on edges
    // between robots (or devices) and 0 inside them
    void setEdgeWeights(const std::vector<double>& w) {
      if (!w.empty() && w.size() != numberOfEdges()) {
        throw std::invalid_argument("GenericScheduler::setEdgeWeights: one weight per edge");
      }
      edge_weight_ = w;
    }
    std::pair<double, double> weightedCounts() const {
      return {last_weighted_v2f_, last_weighted_f2v_};
    }

    // Variable rule inputs for every edge (variables in graph order, edges in variable order)
    Eigen::MatrixXd variableFeatures(bool update) {
      const int n = VariableNodeBase::kRuleInputs + 1; // the variable's inputs, then the edge cost
      Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> X(numberOfEdges(), n);
      std::vector<double> energy(graph_.factor_nodes_.size());
      for (size_t f = 0; f < energy.size(); ++f) {
        energy[f] = graph_.factor_nodes_[f]->energy() / graph_.factor_nodes_[f]->errorDim();
      }
      size_t row = 0;
      std::vector<double> edge_energy;
      for (size_t i = 0; i < graph_.variable_nodes_.size(); ++i) {
        auto& v = graph_.variable_nodes_[i];
        const size_t k = v->numberOfEdges();
        edge_energy.resize(k);
        for (size_t j = 0; j < k; ++j) {
          edge_energy[j] = energy[edge_factor_[i][j]];
        }
        if (k) {
          v->ruleInputs(ticks_, edge_energy.data(), X.data() + row * n, n, update);
        }
        for (size_t j = 0; j < k; ++j) {
          X(row + j, n - 1) = weight(row + j);
        }
        row += k;
      }
      return X;
    }

    // One tick with the rules (uniforms: one U(0,1) per edge for the send decisions; empty: drawn here)
    TickCounts tick(const Eigen::VectorXd& uniforms = Eigen::VectorXd()) {
      const Eigen::MatrixXd X = variableFeatures(true);
      std::uniform_real_distribution<double> unit(0.0, 1.0);
      std::vector<std::vector<int>> masks(graph_.variable_nodes_.size());
      size_t row = 0;
      for (size_t i = 0; i < graph_.variable_nodes_.size(); ++i) {
        auto& v = graph_.variable_nodes_[i];
        masks[i].resize(v->numberOfEdges());
        for (size_t j = 0; j < masks[i].size(); ++j, ++row) {
          const Eigen::VectorXd x = X.row(row).transpose();
          if (vrule_.damping.active) {
            v->setIncomingDamping(j, vrule_.damping(x));
          }
          const double p = vrule_.send.active ? vrule_.send(x) : vrule_.fixed_send;
          const double u = uniforms.size() ? uniforms[row] : unit(rng_);
          masks[i][j]    = u < p ? 1 : 0;
          // input 0 is log1p(ticks since the last send on this edge)
          if (vrule_.max_silence > 0 && std::expm1(x[0]) + 0.5 >= static_cast<double>(vrule_.max_silence)) {
            masks[i][j] = 1;
          }
        }
      }
      return tickMasks(masks);
    }

    // One tick with given variable decisions; the factor rule (if any) decides the replies
    TickCounts tickMasks(const std::vector<std::vector<int>>& masks) {
      TickCounts counts;
      ++ticks_;
      last_weighted_v2f_ = 0.0;
      size_t row         = 0;
      for (size_t i = 0; i < graph_.variable_nodes_.size(); ++i) {
        auto& v = graph_.variable_nodes_[i];
        for (int m : masks[i]) {
          counts.variable_to_factor += m ? 1 : 0;
          last_weighted_v2f_ += m ? weight(row) : 0.0;
          ++row;
        }
        v->ruleRecordSends(masks[i], ticks_);
        v->sendMessages(masks[i]);
        v->markMessagesProcessed();
      }
      if (frule_.active() || record_factor_features_) {
        counts.factor_to_variable += factorStage();
      } else {
        for (auto& f : graph_.factor_nodes_) {
          f->updateFactorIfNew();
          counts.factor_to_variable += f->sendAllMessagesIfNew();
        }
      }
      for (auto& v : graph_.variable_nodes_) {
        v->updateMuAndSigma();
      }
      // f2v costs: the messages each edge received this tick (EdgeBase::f2v_sent_ counts them)
      last_weighted_f2v_ = 0.0;
      for (size_t f = 0; f < factor_edges_.size(); ++f) {
        for (size_t p = 0; p < factor_edges_[f].size(); ++p) {
          auto& e = *factor_edges_[f][p];
          if (e.f2v_sent_ != f2v_seen_[factor_edge_rows_[f][p]]) {
            last_weighted_f2v_ += static_cast<double>(e.f2v_sent_ - f2v_seen_[factor_edge_rows_[f][p]]) *
                                  weight(factor_edge_rows_[f][p]);
            f2v_seen_[factor_edge_rows_[f][p]] = e.f2v_sent_;
          }
        }
      }
      return counts;
    }

    const Eigen::MatrixXd& lastFactorFeatures() const {
      return last_factor_features_;
    }

    bool record_factor_features_ = false;

  private:
    size_t factorStage() {
      constexpr double kEps = 1e-9;
      constexpr double kSigmaFloor = 1e-3; // a thousandth of a standard deviation: below it, no change
      size_t rows = 0;
      for (auto& fe : factor_edges_) {
        rows += fe.size();
      }
      last_factor_features_.resize(rows, kFactorInputNamesV4.size());
      std::uniform_real_distribution<double> unit(0.0, 1.0);
      size_t replies = 0, row0 = 0;
      for (size_t f = 0; f < graph_.factor_nodes_.size(); ++f) {
        auto& factor   = graph_.factor_nodes_[f];
        auto& fe       = factor_edges_[f];
        const size_t k = fe.size();
        auto& X        = last_factor_features_;
        const double d_err = factor->errorDim();
        // New incoming messages: change and oscillation (before any damping)
        std::vector<char> fresh(k);
        for (size_t p = 0; p < k; ++p) {
          fresh[p] = fe[p]->incomingFresh();
          double change = 0.0, osc = 0.0;
          fe[p]->ruleIncoming(fresh[p], change, osc);
          X(row0 + p, 2) = fresh[p] ? 1.0 : 0.0;
          X(row0 + p, 6) = osc;
          X(row0 + p, 7) = std::log(change + kSigmaFloor);
          X(row0 + p, 8) = std::log1p(static_cast<double>(fresh[p] ? std::max(gapOf(f, p), 0L) : 0L));
        }
        const double E_before = factor->energy();
        const double E_prev   = f_energy_prev_[f] < 0 ? E_before : f_energy_prev_[f];
        if (frule_.damping.active) {
          for (size_t p = 0; p < k; ++p) {
            if (fresh[p]) {
              // energy inputs at the decision: the factor's current energy
              X(row0 + p, 3) = std::log(std::sqrt(std::max(2.0 * E_before / d_err, 0.0)) + kSigmaFloor);
              X(row0 + p, 4) = std::clamp(std::log((E_before + kEps) / (E_prev + kEps)), -5.0, 5.0);
              fe[p]->v2f_damping_ = frule_.damping(X.row(row0 + p).transpose());
            }
          }
        }
        factor->prepareIncoming();
        factor->updateFactorIfNew();
        const double E     = factor->energy();
        const bool had_new = factor->hasUnprocessedMessages();
        std::vector<char> pending(k);
        for (size_t p = 0; p < k; ++p) {
          pending[p] = fe[p]->reply_pending_ || had_new;
          double news = 0.0;
          for (size_t q = 0; q < k; ++q) {
            if (q != p) {
              news += fe[q]->ruleNewsSince(p);
            }
          }
          news                 = k > 1 ? news / (k - 1) : 0.0;
          const double E_reply = f_energy_reply_[f][p] < 0 ? E : f_energy_reply_[f][p];
          X(row0 + p, 0) = std::log1p(static_cast<double>(ticks_ - f_reply_tick_[f][p]));
          X(row0 + p, 1) = std::log(news + kSigmaFloor);
          X(row0 + p, 3) = std::log(std::sqrt(std::max(2.0 * E / d_err, 0.0)) + kSigmaFloor);
          X(row0 + p, 4) = std::clamp(std::log((E + kEps) / (E_prev + kEps)), -5.0, 5.0);
          X(row0 + p, 5) = std::log((E + kEps) / (E_reply + kEps));
          fe[p]->reply_selected_ =
            frule_.reply.active ? (unit(rng_) < frule_.reply(X.row(row0 + p).transpose()) ? 1 : 0) : 1;
        }
        replies += factor->sendRepliesIfPending();
        for (size_t p = 0; p < k; ++p) {
          if (pending[p] && fe[p]->reply_selected_) {
            f_reply_tick_[f][p]   = ticks_;
            f_energy_reply_[f][p] = E;
            for (size_t q = 0; q < k; ++q) {
              fe[q]->ruleSnapNews(p);
            }
          }
        }
        f_energy_prev_[f] = E;
        row0 += k;
      }
      return replies;
    }

    // Ticks between the variable's last two messages on factor f's edge p (from the variable side)
    long gapOf(size_t f, size_t p) const {
      // The edge object is shared with the variable, which records the gap when it sends
      return incomingGap(*factor_edges_[f][p]);
    }
    static long incomingGap(const EdgeBase& e) {
      return e.incoming_gap_;
    }

    double weight(size_t row) const {
      return edge_weight_.empty() ? 1.0 : edge_weight_[row];
    }

    FactorGraph& graph_;
    VariableRule vrule_;
    FactorRule frule_;
    std::mt19937_64 rng_;
    long ticks_ = 0;
    std::vector<std::vector<size_t>> edge_factor_;                  // per variable, per edge: factor index
    std::vector<std::vector<std::shared_ptr<EdgeBase>>> factor_edges_; // per factor: its edges
    std::vector<char> edge_slot_;                                   // one per (variable, edge)
    std::vector<std::vector<size_t>> factor_edge_rows_;             // per factor, per edge: its row
    std::vector<double> edge_weight_;                               // per row; empty: 1
    std::vector<size_t> f2v_seen_;                                  // per row: f2v_sent_ at the last tick
    double last_weighted_v2f_ = 0.0, last_weighted_f2v_ = 0.0;
    Eigen::MatrixXd last_factor_features_;
    std::vector<double> f_energy_prev_;
    std::vector<std::vector<long>> f_reply_tick_;
    std::vector<std::vector<double>> f_energy_reply_;
  };

} // namespace gsolver
