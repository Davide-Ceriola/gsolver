#include <gsolver/graph/factor_graph.h>
#include <gsolver/graph/helpers/factor_graph_helper.h>

namespace gsolver {

  template <typename VariableType>
  void FactorGraph::addVariableNode(const std::shared_ptr<VariableType>& variable_node) {
    // Add the variable node to the factor graph
    variable_node->initial_guess_is_prior_ = initial_guess_is_prior_;
    variable_nodes_.push_back(variable_node);
  }

  template <typename FactorType>
  void FactorGraph::addFactorNode(const std::shared_ptr<FactorType>& factor_node, std::vector<std::string> variable_ids) {
    // Add the factor node to the factor graph
    factor_nodes_.push_back(factor_node);

    // assert that the number of variable ids matches the number of edges in the factor node
    if (variable_ids.size() != factor_node->NumberOfEdges_) {
      throw std::runtime_error("Number of variable ids does not match the number of edges in the factor node: " +
                               std::to_string(variable_ids.size()) + " != " + std::to_string(factor_node->NumberOfEdges_));
    }

    using FactorGraphHelper = typename FactorGraphHelperBuilder<FactorType, typename FactorType::VariableTypes>::type;
    // Create edges between the factor node and variable nodes
    for (int i = 0; i < factor_node->NumberOfEdges_; ++i) {
      // Get the variable node from the factor graph
      std::shared_ptr<VariableNodeBase> variable_node = getVariableNode(variable_ids[i]);
      // Create edge
      FactorGraphHelper::createEdge(i, factor_node, variable_node);
    }
  }

  template <typename FactorType>
  void FactorGraph::addFactorNode(const std::shared_ptr<FactorType>& factor_node,
                                  std::vector<std::shared_ptr<VariableNodeBase>> variable_nodes) {
    factor_nodes_.push_back(factor_node);

    if (variable_nodes.size() != factor_node->NumberOfEdges_) {
      throw std::runtime_error("Number of variable nodes does not match the number of edges in the factor node: " +
                               std::to_string(variable_nodes.size()) + " != " + std::to_string(factor_node->NumberOfEdges_));
    }

    using FactorGraphHelper = typename FactorGraphHelperBuilder<FactorType, typename FactorType::VariableTypes>::type;
    for (int i = 0; i < factor_node->NumberOfEdges_; ++i) {
      FactorGraphHelper::createEdge(i, factor_node, variable_nodes[i]);
    }
  }

  std::shared_ptr<VariableNodeBase> FactorGraph::getVariableNode(const std::string& id) {
    for (const auto& variable_node : variable_nodes_) {
      if (variable_node->id_ == id) {
        return variable_node;
      }
    }
    return nullptr;
  }

  std::shared_ptr<FactorNodeBase> FactorGraph::getFactorNode(const std::string& id) {
    for (const auto& factor_node : factor_nodes_) {
      if (factor_node->id_ == id) {
        return factor_node;
      }
    }
    return nullptr;
  }

  std::shared_ptr<EdgeBase> FactorGraph::markFactorSideBoundary(const std::string& factor_id,
                                                                const std::string& variable_id,
                                                                int remote_id) {
    if (getVariableNode(variable_id)) {
      throw std::runtime_error("markFactorSideBoundary: variable " + variable_id +
                               " is in this graph, so its edge to " + factor_id + " is not a boundary");
    }
    std::shared_ptr<FactorNodeBase> factor = getFactorNode(factor_id);
    if (!factor) {
      throw std::runtime_error("markFactorSideBoundary: factor " + factor_id + " not found");
    }
    for (size_t i = 0; i < factor->numberOfEdges(); ++i) {
      std::shared_ptr<EdgeBase> edge = factor->getEdgeAt(i);
      if (edge->variable_id_ == variable_id) {
        edge->side_      = EdgeSide::FACTOR_SIDE;
        edge->remote_id_ = remote_id;
        registerBoundaryEdge(edge);
        return edge;
      }
    }
    throw std::runtime_error("markFactorSideBoundary: factor " + factor_id + " has no edge to " + variable_id);
  }

  std::shared_ptr<EdgeBase> FactorGraph::addVariableSideBoundary(const std::string& variable_id,
                                                                 const std::string& factor_id,
                                                                 int remote_id) {
    if (getFactorNode(factor_id)) {
      throw std::runtime_error("addVariableSideBoundary: factor " + factor_id +
                               " is in this graph, so its edge to " + variable_id + " is not a boundary");
    }
    std::shared_ptr<VariableNodeBase> variable = getVariableNode(variable_id);
    if (!variable) {
      throw std::runtime_error("addVariableSideBoundary: variable " + variable_id + " not found");
    }
    if (getBoundaryEdge(factor_id, variable_id)) {
      throw std::runtime_error("addVariableSideBoundary: edge " + factor_id + " - " + variable_id + " already exists");
    }
    std::shared_ptr<EdgeBase> edge = variable->addDanglingEdge(factor_id, remote_id);
    registerBoundaryEdge(edge);
    return edge;
  }

  std::shared_ptr<EdgeBase> FactorGraph::getBoundaryEdge(const std::string& factor_id, const std::string& variable_id) const {
    const auto it = boundary_index_.find({factor_id, variable_id});
    return it == boundary_index_.end() ? nullptr : it->second;
  }

  void FactorGraph::registerBoundaryEdge(const std::shared_ptr<EdgeBase>& edge) {
    const auto inserted = boundary_index_.emplace(std::make_pair(edge->factor_id_, edge->variable_id_), edge);
    if (!inserted.second) {
      throw std::runtime_error("Boundary edge " + edge->factor_id_ + " - " + edge->variable_id_ + " already registered");
    }
    boundary_edges_.push_back(edge);
  }

  // returns the pose variable nodes sorted by timestamp
  std::vector<std::shared_ptr<VariableNodeBase>> FactorGraph::getPoseVariableNodes() {
    std::vector<std::shared_ptr<VariableNodeBase>> pose_variable_nodes;
    for (const auto& variable_node : variable_nodes_) {
      if (variable_node->id_.find("POSE_") != std::string::npos) {
        pose_variable_nodes.push_back(variable_node);
      }
    }
    // variable_node->id_ = "POSE_" + timestamp, sort by timestamp
    std::sort(pose_variable_nodes.begin(),
              pose_variable_nodes.end(),
              [](const std::shared_ptr<VariableNodeBase>& a, const std::shared_ptr<VariableNodeBase>& b) {
                return std::stod(a->id_.substr(5)) < std::stod(b->id_.substr(5));
              });
    return pose_variable_nodes;
  }

} // namespace gsolver