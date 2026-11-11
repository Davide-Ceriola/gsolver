#pragma once

#include <Eigen/Dense>
#include <memory>
#include <stdexcept>
#include <utility>

#include <gsolver/graph/edge.h>
#include <gsolver/graph/edge_base.h>
#include <gsolver/graph/helpers/factor_node_helper.h>

namespace gsolver {

  /*! @brief Recursive class to access type of variable i in the tuple */
  template <typename FactorEdgeTuple_, int i>
  struct VariableTypeAt_ {
    using VariableType = typename VariableTypeAt_<typename FactorEdgeTuple_::RestFactorEdgeTuple, i - 1>::VariableType;
  };
  /*! @brief Base case of the recursion for VariableTypeAt_ */
  template <typename FactorEdgeTuple_>
  struct VariableTypeAt_<FactorEdgeTuple_, 0> {
    using VariableType = typename FactorEdgeTuple_::CurrentVariableType;
  };

  // Helper class for variadic template recursion
  template <typename CurrentVariableType_, typename... RestVariableTypes_>
  class FactorEdgeTuple_ : public FactorEdgeTuple_<RestVariableTypes_...> {
  public:
    using ThisType            = FactorEdgeTuple_<CurrentVariableType_, RestVariableTypes_...>;
    using CurrentVariableType = CurrentVariableType_;
    using RestFactorEdgeTuple = FactorEdgeTuple_<RestVariableTypes_...>;

    std::shared_ptr<Edge<CurrentVariableType>> current_edge_;

    template <int i>
    using VariableTypeAt = typename VariableTypeAt_<ThisType, i>::VariableType;

    // Access the variable at index i in the tuple
    template <int i>
    std::shared_ptr<Edge<VariableTypeAt<i>>>& at() {
      if constexpr (i == 0) {
        return current_edge_;
      } else {
        return RestFactorEdgeTuple::template at<i - 1>();
      }
    }

    // Access the edge at index i in the tuple
    std::shared_ptr<EdgeBase> getEdgeAt(size_t i) {
      if (i == 0) {
        return current_edge_;
      }
      return RestFactorEdgeTuple::getEdgeAt(i - 1);
    }

    // Set the edge at index i in the tuple
    void setEdgeAt(size_t i, std::shared_ptr<EdgeBase> edge) {
      if (i == 0) {
        current_edge_ = std::dynamic_pointer_cast<Edge<CurrentVariableType>>(edge);
        if (!current_edge_) {
          throw std::runtime_error("Edge is of the wrong type for this factor slot");
        }
        return;
      }
      RestFactorEdgeTuple::setEdgeAt(i - 1, edge);
    }

    // Compute the joint distribution conditioned on the current messages
    template <int VariablesDim_>
    Eigen::Matrix<double, VariablesDim_, VariablesDim_>
    computeJointDistribution(Eigen::Matrix<double, VariablesDim_, VariablesDim_>& Lambdaprime, int offset = 0) {
      static const int edge_size = CurrentVariableType::LieDim;
      Lambdaprime.block(offset, offset, edge_size, edge_size) += current_edge_->variable_to_factor_message_.Lambda_;
      return RestFactorEdgeTuple::computeJointDistribution(Lambdaprime, offset + edge_size);
    }

    // Send message from factor node to variable node
    template <int VariablesDim_, int out_offset_ = 0>
    void sendMessage(const std::string& variable_id,
                     const Eigen::Vector<double, VariablesDim_>& factor_eta,
                     const Eigen::Matrix<double, VariablesDim_, VariablesDim_>& factor_Lambdaprime,
                     const Eigen::Matrix<double, VariablesDim_, VariablesDim_>& conditioned_Lambdaprime,
                     MessageUpdate& update,
                     const Eigen::Vector<double, VariablesDim_>& incoming_eta = Eigen::Vector<double, VariablesDim_>::Zero(),
                     bool at_lin_point = false) {
      if (current_edge_->variable_id_ == variable_id) {
        static const int total_size = VariablesDim_;               // Total size of the joint distribution
        static const int a_size     = CurrentVariableType::LieDim; // Size of the outgoing edge
        static const int b_size     = total_size - a_size;         // Size of the remaining edges
        // Send the message to the variable node
        storeMessage(current_edge_->factor_to_variable_message_,
                    computeMessage<CurrentVariableType, out_offset_, a_size, b_size>(
                      etaExcluding<VariablesDim_, out_offset_, a_size>(factor_eta, incoming_eta), factor_Lambdaprime,
                      conditioned_Lambdaprime, outgoingPoint(at_lin_point)),
                    update);
        return;
      }
      static const int next_out_offset = out_offset_ + CurrentVariableType::LieDim;
      RestFactorEdgeTuple::template sendMessage<VariablesDim_, next_out_offset>(
        variable_id, factor_eta, factor_Lambdaprime, conditioned_Lambdaprime, update, incoming_eta, at_lin_point);
    }

    // Send messages from factor node to all the connected variable node
    // Returns how many messages were stored (not dropped)
    template <int VariablesDim_, int out_offset_ = 0>
    size_t sendAllMessages(const Eigen::Vector<double, VariablesDim_>& factor_eta,
                           const Eigen::Matrix<double, VariablesDim_, VariablesDim_>& factor_Lambdaprime,
                           const Eigen::Matrix<double, VariablesDim_, VariablesDim_>& conditioned_Lambdaprime,
                           MessageUpdate& update,
                           const Eigen::Vector<double, VariablesDim_>& incoming_eta = Eigen::Vector<double, VariablesDim_>::Zero(),
                           bool at_lin_point = false) {
      static const int total_size      = VariablesDim_;               // Total size of the joint distribution
      static const int a_size          = CurrentVariableType::LieDim; // Size of the outgoing edge
      static const int b_size          = total_size - a_size;         // Size of the remaining edges
      static const int next_out_offset = out_offset_ + CurrentVariableType::LieDim;
      // Send the message to the variable node
      const size_t stored = storeMessage(current_edge_->factor_to_variable_message_,
                                         computeMessage<CurrentVariableType, out_offset_, a_size, b_size>(
                                           etaExcluding<VariablesDim_, out_offset_, a_size>(factor_eta, incoming_eta),
                                           factor_Lambdaprime, conditioned_Lambdaprime, outgoingPoint(at_lin_point)),
                                         update);
      // Send messages to the rest of the variable nodes
      return stored + RestFactorEdgeTuple::template sendAllMessages<VariablesDim_, next_out_offset>(
                        factor_eta, factor_Lambdaprime, conditioned_Lambdaprime, update, incoming_eta, at_lin_point);
    }

    // Whether any incoming (variable-to-factor) message has not been processed yet
    bool hasUnprocessedMessage() const {
      return !current_edge_->variable_to_factor_message_.processed_ || RestFactorEdgeTuple::hasUnprocessedMessage();
    }

    // Mark every incoming (variable-to-factor) message as processed
    void setAllMessagesProcessed() {
      current_edge_->variable_to_factor_message_.processed_ = true;
      RestFactorEdgeTuple::setAllMessagesProcessed();
    }
    // ---- Linearisation at the beliefs (FactorNodeBase::linearise_at_belief_) ----
    // The point this edge's variable is linearised at: its last belief, or the message mean
    const typename CurrentVariableType::EstimateType& beliefPoint() const {
      const auto& m = current_edge_->variable_to_factor_message_;
      return m.has_belief_ ? m.belief_mu_ : m.mu_;
    }
    // Whether a belief moved away from the linearisation point by more than tolerance (> 0 if 0)
    bool needsRelinearisationHere(double tolerance) const {
      if (!current_edge_->has_lin_point_) {
        return true;
      }
      const double moved = CurrentVariableType::boxminus(current_edge_->lin_point_, beliefPoint()).norm();
      return tolerance > 0.0 ? moved > tolerance : moved > 0.0;
    }
    void setLinearisationPointHere() {
      current_edge_->lin_point_     = beliefPoint();
      current_edge_->has_lin_point_ = true;
    }
    void swapWithLinearisationPointHere() {
      std::swap(current_edge_->variable_to_factor_message_.mu_, current_edge_->lin_point_);
    }
    // Incoming message as an information vector in the tangent space at the linearisation point
    template <int VariablesDim_>
    void incomingEtaHere(Eigen::Vector<double, VariablesDim_>& eta, int offset) const {
      const auto& m = current_edge_->variable_to_factor_message_;
      eta.segment(offset, CurrentVariableType::LieDim) = m.Lambda_ * CurrentVariableType::boxminus(current_edge_->lin_point_, m.mu_);
    }
    // Factor information vector plus every incoming one except this edge's (the outgoing message
    // excludes what this variable sent)
    template <int VariablesDim_, int offset, int size>
    static Eigen::Vector<double, VariablesDim_> etaExcluding(const Eigen::Vector<double, VariablesDim_>& factor_eta,
                                                            const Eigen::Vector<double, VariablesDim_>& incoming_eta) {
      Eigen::Vector<double, VariablesDim_> eta = factor_eta + incoming_eta;
      eta.template segment<size>(offset) -= incoming_eta.template segment<size>(offset);
      return eta;
    }
    const typename CurrentVariableType::EstimateType& outgoingPoint(bool at_lin_point) const {
      return at_lin_point ? current_edge_->lin_point_ : current_edge_->variable_to_factor_message_.mu_;
    }

    bool needsRelinearisation(double tolerance) const {
      return needsRelinearisationHere(tolerance) || RestFactorEdgeTuple::needsRelinearisation(tolerance);
    }
    void setLinearisationPoints() {
      setLinearisationPointHere();
      RestFactorEdgeTuple::setLinearisationPoints();
    }
    void swapWithLinearisationPoints() {
      swapWithLinearisationPointHere();
      RestFactorEdgeTuple::swapWithLinearisationPoints();
    }
    template <int VariablesDim_>
    void incomingEta(Eigen::Vector<double, VariablesDim_>& eta, int offset = 0) const {
      incomingEtaHere(eta, offset);
      RestFactorEdgeTuple::incomingEta(eta, offset + CurrentVariableType::LieDim);
    }
  };

  // Base case for variadic template recursion
  template <typename CurrentVariableType_>
  class FactorEdgeTuple_<CurrentVariableType_> {
  public:
    using ThisType            = FactorEdgeTuple_<CurrentVariableType_>;
    using CurrentVariableType = CurrentVariableType_;

    std::shared_ptr<Edge<CurrentVariableType>> current_edge_;

    template <int i>
    using VariableTypeAt = typename VariableTypeAt_<ThisType, i>::VariableType;

    // Access the variable at index i in the tuple
    template <int i>
    std::shared_ptr<Edge<VariableTypeAt<i>>>& at() {
      if constexpr (i == 0) {
        return current_edge_;
      } else {
        throw std::runtime_error("index out of bounds");
      }
    }

    // Access the edge at index i in the tuple
    std::shared_ptr<EdgeBase> getEdgeAt(size_t i) {
      if (i == 0) {
        return current_edge_;
      }
      throw std::runtime_error("index out of bounds");
    }

    // Set the edge at index i in the tuple
    void setEdgeAt(size_t i, std::shared_ptr<EdgeBase> edge) {
      if (i == 0) {
        current_edge_ = std::dynamic_pointer_cast<Edge<CurrentVariableType>>(edge);
        if (!current_edge_) {
          throw std::runtime_error("Edge is of the wrong type for this factor slot");
        }
        return;
      }
      throw std::runtime_error("index out of bounds");
    }

    // Compute the joint distribution conditioned on the current messages
    template <int VariablesDim_>
    Eigen::Matrix<double, VariablesDim_, VariablesDim_>
    computeJointDistribution(Eigen::Matrix<double, VariablesDim_, VariablesDim_>& Lambdaprime, int offset = 0) {
      static const int edge_size = CurrentVariableType::LieDim;
      Lambdaprime.block(offset, offset, edge_size, edge_size) += current_edge_->variable_to_factor_message_.Lambda_;
      return Lambdaprime;
    }

    // Send message from factor node to variable node
    template <int VariablesDim_, int out_offset_ = 0>
    void sendMessage(const std::string& variable_id,
                     const Eigen::Vector<double, VariablesDim_>& factor_eta,
                     const Eigen::Matrix<double, VariablesDim_, VariablesDim_>& factor_Lambdaprime,
                     const Eigen::Matrix<double, VariablesDim_, VariablesDim_>& conditioned_Lambdaprime,
                     MessageUpdate& update,
                     const Eigen::Vector<double, VariablesDim_>& incoming_eta = Eigen::Vector<double, VariablesDim_>::Zero(),
                     bool at_lin_point = false) {
      if (current_edge_->variable_id_ == variable_id) {
        static const int total_size = VariablesDim_;               // Total size of the joint distribution
        static const int a_size     = CurrentVariableType::LieDim; // Size of the outgoing edge
        static const int b_size     = total_size - a_size;         // Size of the remaining edges
        // Send the message to the variable node
        storeMessage(current_edge_->factor_to_variable_message_,
                    computeMessage<CurrentVariableType, out_offset_, a_size, b_size>(
                      etaExcluding<VariablesDim_, out_offset_, a_size>(factor_eta, incoming_eta), factor_Lambdaprime,
                      conditioned_Lambdaprime, outgoingPoint(at_lin_point)),
                    update);
        return;
      }
      throw std::runtime_error("variable id not found");
    }

    // Send messages from factor node to all the connected variable node
    template <int VariablesDim_, int out_offset_ = 0>
    size_t sendAllMessages(const Eigen::Vector<double, VariablesDim_>& factor_eta,
                           const Eigen::Matrix<double, VariablesDim_, VariablesDim_>& factor_Lambdaprime,
                           const Eigen::Matrix<double, VariablesDim_, VariablesDim_>& conditioned_Lambdaprime,
                           MessageUpdate& update,
                           const Eigen::Vector<double, VariablesDim_>& incoming_eta = Eigen::Vector<double, VariablesDim_>::Zero(),
                           bool at_lin_point = false) {
      static const int total_size = VariablesDim_;               // Total size of the joint distribution
      static const int a_size     = CurrentVariableType::LieDim; // Size of the outgoing edge
      static const int b_size     = total_size - a_size;         // Size of the remaining edges
      // Send the message to the variable node
      return storeMessage(current_edge_->factor_to_variable_message_,
                          computeMessage<CurrentVariableType, out_offset_, a_size, b_size>(
                            etaExcluding<VariablesDim_, out_offset_, a_size>(factor_eta, incoming_eta), factor_Lambdaprime,
                            conditioned_Lambdaprime, outgoingPoint(at_lin_point)),
                          update);
    }

    bool hasUnprocessedMessage() const {
      return !current_edge_->variable_to_factor_message_.processed_;
    }

    void setAllMessagesProcessed() {
      current_edge_->variable_to_factor_message_.processed_ = true;
    }
    // ---- Linearisation at the beliefs (FactorNodeBase::linearise_at_belief_) ----
    // The point this edge's variable is linearised at: its last belief, or the message mean
    const typename CurrentVariableType::EstimateType& beliefPoint() const {
      const auto& m = current_edge_->variable_to_factor_message_;
      return m.has_belief_ ? m.belief_mu_ : m.mu_;
    }
    // Whether a belief moved away from the linearisation point by more than tolerance (> 0 if 0)
    bool needsRelinearisationHere(double tolerance) const {
      if (!current_edge_->has_lin_point_) {
        return true;
      }
      const double moved = CurrentVariableType::boxminus(current_edge_->lin_point_, beliefPoint()).norm();
      return tolerance > 0.0 ? moved > tolerance : moved > 0.0;
    }
    void setLinearisationPointHere() {
      current_edge_->lin_point_     = beliefPoint();
      current_edge_->has_lin_point_ = true;
    }
    void swapWithLinearisationPointHere() {
      std::swap(current_edge_->variable_to_factor_message_.mu_, current_edge_->lin_point_);
    }
    // Incoming message as an information vector in the tangent space at the linearisation point
    template <int VariablesDim_>
    void incomingEtaHere(Eigen::Vector<double, VariablesDim_>& eta, int offset) const {
      const auto& m = current_edge_->variable_to_factor_message_;
      eta.segment(offset, CurrentVariableType::LieDim) = m.Lambda_ * CurrentVariableType::boxminus(current_edge_->lin_point_, m.mu_);
    }
    // Factor information vector plus every incoming one except this edge's (the outgoing message
    // excludes what this variable sent)
    template <int VariablesDim_, int offset, int size>
    static Eigen::Vector<double, VariablesDim_> etaExcluding(const Eigen::Vector<double, VariablesDim_>& factor_eta,
                                                            const Eigen::Vector<double, VariablesDim_>& incoming_eta) {
      Eigen::Vector<double, VariablesDim_> eta = factor_eta + incoming_eta;
      eta.template segment<size>(offset) -= incoming_eta.template segment<size>(offset);
      return eta;
    }
    const typename CurrentVariableType::EstimateType& outgoingPoint(bool at_lin_point) const {
      return at_lin_point ? current_edge_->lin_point_ : current_edge_->variable_to_factor_message_.mu_;
    }

    bool needsRelinearisation(double tolerance) const {
      return needsRelinearisationHere(tolerance);
    }
    void setLinearisationPoints() {
      setLinearisationPointHere();
    }
    void swapWithLinearisationPoints() {
      swapWithLinearisationPointHere();
    }
    template <int VariablesDim_>
    void incomingEta(Eigen::Vector<double, VariablesDim_>& eta, int offset = 0) const {
      incomingEtaHere(eta, offset);
    }
  };

} // namespace gsolver
