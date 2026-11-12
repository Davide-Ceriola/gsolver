#pragma once

#include <memory>
#include <string>
#include <vector>

#include <gsolver/graph/edge_base.h>
#include <gsolver/graph/message_update.h>

class VariableNodeBase {
public:
  std::string id_; // Variable node ID

  // What a variable with a single factor sends to that factor.
  //  true  (legacy): nothing is recomputed, so the edge keeps its initial message (the initial
  //                  guess with precision Sigma^-1) forever: there the guess acts as a prior.
  //  false:          zero precision with the current mean, as GBP prescribes: with no other
  //                  factors the message carries no information.
  // Initial messages on new edges are the guess with Sigma^-1 in both modes; they are overwritten
  // after the first iteration, so they do not change the fixed point. Starting from zero precision
  // instead made gsolver's (mu, Lambda) messages numerically unstable on the stereo example.
  // FactorGraph::addVariableNode applies the graph's setting.
  bool initial_guess_is_prior_ = true;

  // Dropout of its variable-to-factor messages (damping applies to factor messages only)
  gsolver::MessageUpdate message_update_;

  virtual ~VariableNodeBase() = default;

  virtual void sendMessage(size_t outedge_id) = 0;
  virtual void sendMessage(const std::string& factor_id) = 0;
  virtual void sendAllMessages()              = 0;

  // Send on the edges with mask[i] != 0 (a scheduler's decision); with send_belief_ the belief is
  // computed once and sent on all of them
  virtual void sendMessages(const std::vector<int>& mask) = 0;

  // ---- Generic rule inputs (core/generic_scheduler.h), computed by the typed variable ----
  static constexpr int kRuleInputs = 9;
  virtual int lieDim() const = 0;
  virtual void ruleReset()   = 0;
  // Rows (one per edge, kRuleInputs columns, row stride `stride`) of the variable's rule inputs;
  // edge_energy[j]: energy per error dimension of the factor on edge j. update: advance the state
  // (previous energies, uncertainty, oscillation)
  virtual void ruleInputs(long tick, const double* edge_energy, double* out, int stride, bool update) = 0;
  // Record the sends of this tick (before the belief update): last-sent belief, tick and gap
  virtual void ruleRecordSends(const std::vector<int>& mask, long tick) = 0;
  // Set the damping of the incoming factor-to-variable message on edge j
  virtual void setIncomingDamping(size_t j, double d) = 0;

  // Send the belief instead of per-edge messages (one computation, same for every edge); factors
  // recover the per-edge messages exactly. Off by default.
  bool send_belief_ = false;
  virtual void updateMuAndSigma()             = 0;

  // Incremental message passing (see FactorNodeBase): whether an incoming factor-to-variable
  // message has not been processed yet, and marking them all processed. Every message a
  // variable writes on an edge is marked unprocessed for its factor.
  virtual bool hasUnprocessedMessages() const = 0;
  virtual void markMessagesProcessed()        = 0;

  // Add an edge to a factor that lives in another graph (marked VARIABLE_SIDE). Its incoming
  // message starts with zero precision at the current mean, until a payload is received.
  virtual std::shared_ptr<gsolver::EdgeBase> addDanglingEdge(const std::string& factor_id, int remote_id) = 0;

  virtual size_t numberOfEdges() const                          = 0;
  virtual std::shared_ptr<gsolver::EdgeBase> getEdgeAt(size_t i) = 0;
  // Remove the edge to factor_id; throws if there is none
  virtual void removeEdge(const std::string& factor_id) = 0;
};
