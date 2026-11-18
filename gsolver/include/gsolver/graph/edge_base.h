#pragma once

#include <string>
#include <vector>

namespace gsolver {

  // Which half of an edge this object is. A LOCAL edge connects a factor and a variable of the
  // same graph. An edge whose two ends live in different graphs (e.g. on different robots) is
  // split into two objects, one per graph, that a caller keeps in sync by exchanging payloads:
  //  FACTOR_SIDE:   held by the local factor; the variable lives in another graph.
  //  VARIABLE_SIDE: held by the local variable; the factor lives in another graph.
  enum class EdgeSide { LOCAL, FACTOR_SIDE, VARIABLE_SIDE };

  class EdgeBase {
  public:
    virtual ~EdgeBase() = default;

    std::string variable_id_;
    std::string factor_id_;

    EdgeSide side_  = EdgeSide::LOCAL;
    int remote_id_  = -1; // id of the graph holding the other half (e.g. a robot id); -1 if LOCAL

    // Payload of a boundary edge: the message this half sends, as doubles, and the message it
    // receives. Layout: encoded mean (VariableType::EncodedSize values), then the upper triangle
    // of Lambda row by row. Throw for LOCAL edges.
    // ---- Scheduling state set by schedulers (plain data) ----
    // Damping of the factor-to-variable message on this edge; negative: the factor's own
    double damping_ = -1.0;
    // Per-component damping of that message, one value per tangent dimension
    // (VariableNodeBase::setIncomingDampingComponents; empty: damping_ for every component). When set,
    // damping_ holds their mean, which mixes the precisions; the mean moves per component.
    std::vector<double> damping_components_;
    // Per-component rule inputs of the variable on this edge (kComponentInputs per tangent dimension,
    // dimension-major; kVariableComponentInputNames), written by the variable's ruleInputs
    std::vector<double> component_inputs_;
    // Factor-side scheduling (FactorNodeBase::sendRepliesIfPending): the factor has new
    // information for this variable it has not sent yet / may reply on this edge now
    bool reply_pending_ = false;
    int reply_selected_ = 1;
    // Factor-to-variable messages stored on this edge so far (counted by the factor; read by
    // GenericScheduler for per-edge message costs)
    size_t f2v_sent_ = 0;
    // Damping of the incoming variable-to-factor message, applied by the factor (negative: none)
    double v2f_damping_ = -1.0;
    // Ticks between the variable's last two messages on this edge (recorded by the variable)
    long incoming_gap_ = 0;

    // ---- Generic rule inputs (core/generic_scheduler.h), computed by the typed edge ----
    virtual bool incomingFresh() const = 0; // a new variable-to-factor message not processed yet
    // Dimension of the variable's tangent space
    virtual int lieDim() const = 0;
    // Factor side: a new incoming variable-to-factor message (fresh) -> its change since the previous
    // one, per dimension, in its own precision metric, and the cosine of its last two changes (same
    // metric; kept between changes)
    virtual void ruleIncoming(bool fresh, double& change_per_dim, double& oscillation) = 0;
    // Factor side, "news": how far this edge's incoming message moved since snapshot p (per dimension,
    // own precision metric); snapshots are taken when the factor replies on its edge p
    virtual double ruleNewsSince(size_t p) const          = 0;
    virtual void ruleSnapNews(size_t p)                   = 0;
    virtual void ruleResetFactorSide(size_t factor_arity) = 0;

    virtual int payloadSize() const                          = 0;
    virtual void writeOutgoingPayload(double* payload) const = 0;
    virtual void readIncomingPayload(const double* payload)  = 0;
  };

} // namespace gsolver
