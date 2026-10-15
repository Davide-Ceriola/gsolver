#pragma once

#include <memory>
#include <string>

#include <gsolver/graph/edge_base.h>

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

  virtual ~VariableNodeBase() = default;

  virtual void sendMessage(size_t outedge_id) = 0;
  virtual void sendMessage(const std::string& factor_id) = 0;
  virtual void sendAllMessages()              = 0;
  virtual void updateMuAndSigma()             = 0;

  // Add an edge to a factor that lives in another graph (marked VARIABLE_SIDE). Its incoming
  // message starts with zero precision at the current mean, until a payload is received.
  virtual std::shared_ptr<gsolver::EdgeBase> addDanglingEdge(const std::string& factor_id, int remote_id) = 0;
};
