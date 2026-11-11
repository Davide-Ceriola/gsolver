#pragma once

#include <memory>
#include <string>

#include <gsolver/core/robust_kernels.h>
#include <gsolver/graph/edge_base.h>
#include <gsolver/graph/message_update.h>

namespace gsolver {

  class FactorNodeBase {
  public:
    std::string id_; // Factor node ID

    RobustKernelConfig robust_kernel_; // NONE by default
    double robust_weight_ = 1.0;       // weight applied at the last updateFactor (1 without a kernel)

    // A disabled factor stays connected but contributes no information: its messages have zero
    // precision. E.g. an observation waiting for a linearization point of a remote variable.
    bool enabled_ = true;

    // Damping and dropout of its factor-to-variable messages (none by default)
    MessageUpdate message_update_;

    // Where the factor linearises. false (default): at the means of its incoming
    // variable-to-factor messages (each variable's cavity mean), re-evaluated at every update.
    // true: at the variables' belief means, which the messages carry; the factor keeps that
    // linearisation point until a belief moves by more than relinearise_tolerance_ (0: any move)
    // and adds each incoming message as an information vector at that point.
    bool linearise_at_belief_     = false;
    double relinearise_tolerance_ = 0.0;

    virtual ~FactorNodeBase() = default;

    virtual void setEdgeAt(size_t i, std::shared_ptr<EdgeBase> edge) = 0; // Set edge at index i
    virtual std::shared_ptr<EdgeBase> getEdgeAt(size_t i)            = 0; // Edge at index i
    virtual size_t numberOfEdges() const                             = 0;
    virtual void updateFactor()                                      = 0; // Update the factor based on the connected edges
    virtual void sendMessage(const std::string& variable_id)         = 0;
    virtual void sendAllMessages()                                   = 0;

    // Incremental message passing, for schedulers that decide which variable-to-factor messages
    // are sent (the solver's own loop does not use these). Every stored message starts
    // unprocessed; a factor with no unprocessed incoming message has nothing new to compute.
    virtual bool hasUnprocessedMessages() const = 0; // any incoming message not processed yet
    virtual void markMessagesProcessed()        = 0; // mark every incoming message processed
    // updateFactor() only if an incoming message is new; returns whether it ran
    virtual bool updateFactorIfNew() = 0;
    // If an incoming message is new: send to every variable, mark the incoming messages
    // processed, and return how many messages were stored (dropout can skip some). Else 0.
    virtual size_t sendAllMessagesIfNew() = 0;

    // 0.5 e^T Sigma^-1 e at the last updateFactor, i.e. at the factor's incoming messages: the
    // scalar a factor can attach to its replies (no robust kernel applied)
    virtual double energy() const = 0;
  };

} // namespace gsolver
