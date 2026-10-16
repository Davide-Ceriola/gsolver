#include <gsolver/graph/factor_node.h>

namespace gsolver {

  template <int Dim_, typename... VariableTypes_>
  FactorNode<Dim_, VariableTypes_...>::FactorNode(const std::string& id, const MatrixErrorDim& Sigma) {
    id_          = id;
    Sigma_       = Sigma;
    eta_         = VectorVariablesDim::Zero();
    Lambdaprime_ = MatrixVariablesDim::Identity();
    error_       = VectorErrorDim::Zero();
    J_           = MatrixJacobianDim::Zero();
  }

  template <int Dim_, typename... VariableTypes_>
  void FactorNode<Dim_, VariableTypes_...>::updateFactor() {
    // Update the error and Jacobian
    updateErrorAndJacobian();
    const MatrixErrorDim Omega = Sigma_.inverse();

    // Robust kernel: scale the factor's information by a weight from its Mahalanobis distance
    robust_weight_ = 1.0;
    if (robust_kernel_.kernel != RobustKernel::NONE) {
      const double squared = error_.dot(Omega * error_);
      robust_weight_       = robustWeight(robust_kernel_, std::sqrt(std::max(squared, 0.0)));
    }
    const double k = robust_weight_;

    // Update factor estimate (canonical form)
    eta_                       = J_.transpose() * Omega * (-error_) * k;
    Lambdaprime_               = J_.transpose() * Omega * J_ * k;
    symmetrize(Lambdaprime_);

    // Update Lambda by conditioning on the connected edges
    Lambdaprime_conditioned_ = Lambdaprime_;
    Lambdaprime_conditioned_ = edges_.template computeJointDistribution<VariablesDim_>(Lambdaprime_conditioned_);
  }

  template <int Dim_, typename... VariableTypes_>
  void FactorNode<Dim_, VariableTypes_...>::sendMessage(const std::string& variable_id) {
    // Send message from factor node to variable node
    edges_.template sendMessage<VariablesDim_>(variable_id, eta_, Lambdaprime_, Lambdaprime_conditioned_);
  }

  template <int Dim_, typename... VariableTypes_>
  void FactorNode<Dim_, VariableTypes_...>::sendAllMessages() {
    // iterate over all edges and send messages to all connected variable nodes
    edges_.template sendAllMessages<VariablesDim_>(eta_, Lambdaprime_, Lambdaprime_conditioned_);
  }

} // namespace gsolver