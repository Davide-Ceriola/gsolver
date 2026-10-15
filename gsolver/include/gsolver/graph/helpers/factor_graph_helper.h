#pragma once

#include <tuple>
#include <type_traits>
#include <utility>

#include <gsolver/graph/edge.h>
#include <gsolver/graph/factor_node.h>
#include <gsolver/graph/variable_node.h>
#include <gsolver/graph/variable_node_base.h>

namespace gsolver {

  // Create the edge between slot i of a factor node and an already typed variable node
  template <typename VariableType, typename FactorType>
  void connectEdge(int i, const std::shared_ptr<FactorType>& factor_node, const std::shared_ptr<VariableType>& variable_node) {
    // Create a new edge between the factor node and the variable node
    std::shared_ptr<Edge<VariableType>> edge = std::make_shared<Edge<VariableType>>();
    edge->variable_id_                       = variable_node->id_;
    edge->factor_id_                         = factor_node->id_;

    // Initialize the variable-to-factor message
    edge->variable_to_factor_message_.mu_     = variable_node->mu_;
    edge->variable_to_factor_message_.Lambda_ = variable_node->Sigma_.inverse();

    // Add the edge to the factor node and variable node
    factor_node->setEdgeAt(i, edge);
    variable_node->edges_.push_back(edge);
  }

  // Helper class for variadic template recursion
  template <typename FactorType_, typename CurrentVariableType_, typename... RestVariableTypes_>
  class FactorGraphHelper_ : public FactorGraphHelper_<FactorType_, RestVariableTypes_...> {
  public:
    using FactorType            = FactorType_;
    using CurrentVariableType   = CurrentVariableType_;
    using RestFactorGraphHelper = FactorGraphHelper_<FactorType_, RestVariableTypes_...>;

    // Recursively create edges between the factor node and variable nodes
    static void createEdge(int i,
                           std::shared_ptr<FactorType> factor_node,
                           std::shared_ptr<VariableNodeBase> variable_node_base,
                           int recoursion_depth = 0) {
      if (recoursion_depth == i) {
        // Check if the variable node is of the correct type and cast it
        std::shared_ptr<CurrentVariableType> variable_node =
          std::dynamic_pointer_cast<CurrentVariableType>(variable_node_base);
        if (!variable_node) {
          throw std::runtime_error("Variable node not found or of the wrong type for this factor");
        }
        connectEdge(i, factor_node, variable_node);
        return;
      }
      RestFactorGraphHelper::createEdge(i, factor_node, variable_node_base, recoursion_depth + 1);
    }
  };

  // Base case for variadic template recursion
  template <typename FactorType_, typename CurrentVariableType_>
  class FactorGraphHelper_<FactorType_, CurrentVariableType_> {
  public:
    using FactorType          = FactorType_;
    using CurrentVariableType = CurrentVariableType_;

    // Recursively create edges between the factor node and variable nodes
    static void createEdge(int i,
                           std::shared_ptr<FactorType> factor_node,
                           std::shared_ptr<VariableNodeBase> variable_node_base,
                           int recoursion_depth = 0) {
      if (recoursion_depth == i) {
        // Check if the variable node is of the correct type and cast it
        std::shared_ptr<CurrentVariableType> variable_node =
          std::dynamic_pointer_cast<CurrentVariableType>(variable_node_base);
        if (!variable_node) {
          throw std::runtime_error("Variable node not found or of the wrong type for this factor");
        }
        connectEdge(i, factor_node, variable_node);
        return;
      }
      throw std::runtime_error("index out of bounds");
    }
  };

  // Helper to unpack tuple
  template <typename FactorNode_, typename Tuple>
  struct FactorGraphHelperBuilder;

  template <typename FactorNode_, typename... VariableNodes_>
  struct FactorGraphHelperBuilder<FactorNode_, std::tuple<VariableNodes_...>> {
    using type = FactorGraphHelper_<FactorNode_, VariableNodes_...>;
  };

  // True if each given variable type is (or derives from) the type the factor expects in that slot
  template <typename ExpectedTuple, typename... GivenVariableTypes, std::size_t... I>
  constexpr bool variableTypesMatch(std::index_sequence<I...>) {
    return (std::is_base_of_v<std::tuple_element_t<I, ExpectedTuple>, GivenVariableTypes> && ...);
  }

  // Connect every slot of a factor node to the given typed variable nodes, in order
  template <typename FactorType, typename... GivenVariableTypes, std::size_t... I>
  void connectAllEdges(const std::shared_ptr<FactorType>& factor_node,
                       std::index_sequence<I...>,
                       const std::shared_ptr<GivenVariableTypes>&... variable_nodes) {
    using ExpectedTuple = typename FactorType::VariableTypes;
    (connectEdge<std::tuple_element_t<I, ExpectedTuple>>(static_cast<int>(I), factor_node, variable_nodes), ...);
  }

} // namespace gsolver
