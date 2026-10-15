#pragma once

#include <string>

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
    virtual int payloadSize() const                          = 0;
    virtual void writeOutgoingPayload(double* payload) const = 0;
    virtual void readIncomingPayload(const double* payload)  = 0;
  };

} // namespace gsolver
