#pragma once
// Accessors shared by every variable / factor binding: a variable's edges, a factor's error,
// covariance and Jacobian. Used by external schedulers and objectives (rls_gbp) that read the
// GBP state from Python.

#include <memory>
#include <vector>

#include <nanobind/nanobind.h>
#include <nanobind/eigen/dense.h>
#include <nanobind/stl/shared_ptr.h>
#include <nanobind/stl/vector.h>

namespace gsolver::python {

  namespace nb = nanobind;

  template <typename VariableType, typename Class>
  Class& add_variable_accessors(Class& cls) {
    using EdgePtr = std::shared_ptr<typename VariableType::EdgeType>;
    return cls.def_prop_ro(
      "edges",
      [](VariableType& v) -> std::vector<EdgePtr>& { return v.edges_; },
      "Edges connected to this variable, in message order",
      nb::rv_policy::reference_internal);
  }

  template <typename FactorType, typename Class>
  Class& add_factor_accessors(Class& cls) {
    return cls
      .def_prop_ro(
        "error", [](const FactorType& f) { return f.error_; }, "Residual at the last updateFactor")
      .def_prop_ro(
        "covariance", [](const FactorType& f) { return f.Sigma_; }, "Measurement covariance")
      .def_prop_ro(
        "jacobian", [](const FactorType& f) { return f.J_; }, "Jacobian of the residual at the last updateFactor");
  }

} // namespace gsolver::python
