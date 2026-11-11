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
        .def_prop_rw(
          "damping",
          [](const FactorNodeBase& f) { return f.message_update_.damping; },
          [](FactorNodeBase& f, double d) { f.message_update_.damping = d; },
          "Share of the previous factor-to-variable message each new one keeps (0: none)")
        .def_prop_rw(
          "dropout",
          [](const FactorNodeBase& f) { return f.message_update_.dropout; },
          [](FactorNodeBase& f, double d) { f.message_update_.dropout = d; },
          "Chance that a new factor-to-variable message is not sent (0: none)")
        .def(
          "set_edge_at",
          &FactorNodeBase::setEdgeAt,
          nb::arg("index"),
          nb::arg("edge"),
          "Associate an edge with the factor at the given slot")
        .def(
          "update_factor",
          &FactorNodeBase::updateFactor,
          nb::call_guard<nb::gil_scoped_release>(),
          "Recompute the factor's canonical parameters from the connected edges")
        .def(
          "send_message",
          &FactorNodeBase::sendMessage,
          nb::arg("variable_id"),
          nb::call_guard<nb::gil_scoped_release>(),
          "Send a message toward the specified variable node")
        .def(
          "send_all_messages",
          &FactorNodeBase::sendAllMessages,
          nb::call_guard<nb::gil_scoped_release>(),
          "Broadcast messages to every connected variable node")
        // Incremental message passing for external schedulers (see FactorNodeBase)
        .def("has_unprocessed_messages",
             &FactorNodeBase::hasUnprocessedMessages,
             "Whether an incoming variable-to-factor message has not been processed yet")
        .def("mark_messages_processed",
             &FactorNodeBase::markMessagesProcessed,
             "Mark every incoming variable-to-factor message processed")
        .def("update_factor_if_new",
             &FactorNodeBase::updateFactorIfNew,
             nb::call_guard<nb::gil_scoped_release>(),
             "update_factor() only if an incoming message is new; returns whether it ran")
        .def("send_all_messages_if_new",
             &FactorNodeBase::sendAllMessagesIfNew,
             nb::call_guard<nb::gil_scoped_release>(),
             "If an incoming message is new: reply to every variable, mark the incoming messages processed, "
             "return the number of messages stored. Else 0.")
        .def_prop_ro("num_edges", &FactorNodeBase::numberOfEdges, "Number of connected variables");
    }

  } // namespace python
} // namespace gsolver
