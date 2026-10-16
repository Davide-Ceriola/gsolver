// =============================================================================
// gsolver Python bindings - FactorNodeBase
// =============================================================================
#include <nanobind/nanobind.h>
#include <nanobind/stl/shared_ptr.h>
#include <nanobind/stl/string.h>

#include <gsolver/graph/factor_node_base.h>

namespace nb = nanobind;

namespace gsolver {
  namespace python {

    void bind_factor_node_base(nb::module_& m) {
      nb::enum_<RobustKernel>(m, "RobustKernel", "Robust kernel applied to a factor")
        .value("NONE", RobustKernel::NONE)
        .value("HUBER", RobustKernel::HUBER)
        .value("CAUCHY", RobustKernel::CAUCHY)
        .value("GEMAN_MCCLURE", RobustKernel::GEMAN_MCCLURE)
        .value("TUKEY", RobustKernel::TUKEY);

      nb::class_<RobustKernelConfig>(m, "RobustKernelConfig", "Kernel and threshold (Mahalanobis distance)")
        .def(nb::init<>())
        .def_rw("kernel", &RobustKernelConfig::kernel)
        .def_rw("threshold", &RobustKernelConfig::threshold);

      nb::class_<FactorNodeBase>(m, "FactorNodeBase", "Base class for factor nodes")
        .def_ro("id", &FactorNodeBase::id_, "Factor node ID")
        .def_rw("robust_kernel", &FactorNodeBase::robust_kernel_, "Robust kernel (NONE by default)")
        .def_ro("robust_weight", &FactorNodeBase::robust_weight_, "Weight applied at the last update")
        .def_rw("enabled", &FactorNodeBase::enabled_, "A disabled factor contributes no information")
        .def(
          "set_edge_at",
          &FactorNodeBase::setEdgeAt,
          nb::arg("index"),
          nb::arg("edge"),
          "Associate an edge with the factor at the given slot")
        .def(
          "update_factor",
          &FactorNodeBase::updateFactor,
          "Recompute the factor's canonical parameters from the connected edges")
        .def(
          "send_message",
          &FactorNodeBase::sendMessage,
          nb::arg("variable_id"),
          "Send a message toward the specified variable node")
        .def(
          "send_all_messages",
          &FactorNodeBase::sendAllMessages,
          "Broadcast messages to every connected variable node");
    }

  } // namespace python
} // namespace gsolver
