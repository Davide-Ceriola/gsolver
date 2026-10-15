#pragma once

#include <map>
#include <utility>

#include <gsolver/graph/edge.h>
#include <gsolver/graph/factor_node.h>
#include <gsolver/graph/variable_node.h>
#include <gsolver/graph/helpers/factor_graph_helper.h>

namespace gsolver {

  class FactorGraph {
  public:
    std::vector<std::shared_ptr<VariableNodeBase>> variable_nodes_;
    std::vector<std::shared_ptr<FactorNodeBase>> factor_nodes_;

    // Applied to every variable added afterwards; see VariableNodeBase::initial_guess_is_prior_
    bool initial_guess_is_prior_ = true;

    template <typename VariableType>
    void addVariableNode(const std::shared_ptr<VariableType>& variable_node);

    template <typename FactorType>
    void addFactorNode(const std::shared_ptr<FactorType>& factor_node, std::vector<std::string> variable_ids);

    // Overload that accepts variable nodes directly — needed when variables live
    // in different factor graphs (e.g. inter-robot landmark equality factors).
    template <typename FactorType>
    void addFactorNode(const std::shared_ptr<FactorType>& factor_node,
                       std::vector<std::shared_ptr<VariableNodeBase>> variable_nodes);

    // Typed overload: number and types of the variables are checked at compile time.
    // Defined in this header, so it works for any factor type without an explicit instantiation.
    template <typename FactorType, typename... GivenVariableTypes>
    void addFactorNode(const std::shared_ptr<FactorType>& factor_node,
                       const std::shared_ptr<GivenVariableTypes>&... variable_nodes);

    std::shared_ptr<VariableNodeBase> getVariableNode(const std::string& id);

    std::shared_ptr<FactorNodeBase> getFactorNode(const std::string& id);

    // returns the pose variable nodes sorted by timestamp
    std::vector<std::shared_ptr<VariableNodeBase>> getPoseVariableNodes(); // TODO: revisit the method

    // -------------------------------------------------------------------------
    // Boundary edges: the local half of an edge whose other end lives in another graph.
    // The caller moves payloads between matching halves (see EdgeBase); the solver
    // itself treats boundary edges like any other edge.
    // -------------------------------------------------------------------------

    // All boundary edges of this graph, in the order they were created
    std::vector<std::shared_ptr<EdgeBase>> boundary_edges_;

    // Mark the edge between a local factor and a variable of another graph as FACTOR_SIDE.
    // Build that factor first with a stand-in for the remote variable: a typed variable node
    // that is not added to this graph, e.g. addFactorNode(factor, local_variable, remote_stub).
    // The stub's mean and Sigma seed the edge's incoming message until a payload arrives.
    std::shared_ptr<EdgeBase> markFactorSideBoundary(const std::string& factor_id,
                                                     const std::string& variable_id,
                                                     int remote_id);

    // Give a local variable a VARIABLE_SIDE edge to a factor of another graph
    std::shared_ptr<EdgeBase> addVariableSideBoundary(const std::string& variable_id,
                                                      const std::string& factor_id,
                                                      int remote_id);

    // The boundary edge between factor_id and variable_id, or nullptr
    std::shared_ptr<EdgeBase> getBoundaryEdge(const std::string& factor_id, const std::string& variable_id) const;

    // -------------------------------------------------------------------------
    // Removal. Call between iterations, never during one.
    // -------------------------------------------------------------------------

    // Remove a factor and its edges, including the local half of any boundary edge it holds
    void removeFactorNode(const std::string& factor_id);

    // Remove the local half of a boundary edge: a VARIABLE_SIDE edge is dropped from its
    // variable; a FACTOR_SIDE edge takes its factor with it (a factor cannot lose a variable)
    void removeBoundaryEdge(const std::string& factor_id, const std::string& variable_id);

    // Remove a variable together with every factor connected to it and its boundary halves
    void removeVariableNode(const std::string& variable_id);

  private:
    void registerBoundaryEdge(const std::shared_ptr<EdgeBase>& edge);
    void unregisterBoundaryEdge(const std::string& factor_id, const std::string& variable_id);

    std::map<std::pair<std::string, std::string>, std::shared_ptr<EdgeBase>> boundary_index_; // (factor, variable)
  };

  template <typename FactorType, typename... GivenVariableTypes>
  void FactorGraph::addFactorNode(const std::shared_ptr<FactorType>& factor_node,
                                  const std::shared_ptr<GivenVariableTypes>&... variable_nodes) {
    using Indices                = std::index_sequence_for<GivenVariableTypes...>;
    constexpr bool count_matches = sizeof...(GivenVariableTypes) == static_cast<std::size_t>(FactorType::NumberOfEdges_);
    static_assert(count_matches, "addFactorNode: wrong number of variables for this factor");

    // Each check guards the next, so a mistake gives a single clear error
    if constexpr (count_matches) {
      constexpr bool types_match =
        variableTypesMatch<typename FactorType::VariableTypes, GivenVariableTypes...>(Indices{});
      static_assert(types_match, "addFactorNode: variable types don't match the factor's variable types");

      if constexpr (types_match) {
        factor_nodes_.push_back(factor_node);
        connectAllEdges(factor_node, Indices{}, variable_nodes...);
      }
    }
  }

} // namespace gsolver
