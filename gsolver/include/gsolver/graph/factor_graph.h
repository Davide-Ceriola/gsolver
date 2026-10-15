#pragma once

#include <gsolver/graph/edge.h>
#include <gsolver/graph/factor_node.h>
#include <gsolver/graph/variable_node.h>
#include <gsolver/graph/helpers/factor_graph_helper.h>

namespace gsolver {

  class FactorGraph {
  public:
    std::vector<std::shared_ptr<VariableNodeBase>> variable_nodes_;
    std::vector<std::shared_ptr<FactorNodeBase>> factor_nodes_;

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
