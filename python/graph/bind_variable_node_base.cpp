// =============================================================================
// gsolver Python bindings - VariableNodeBase
// =============================================================================
#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include <gsolver/graph/variable_node_base.h>

namespace nb = nanobind;

namespace gsolver {
  namespace python {

    void bind_variable_node_base(nb::module_& m) {
      nb::class_<VariableNodeBase>(m, "VariableNodeBase", "Base class for variable nodes")
        .def_ro("id", &VariableNodeBase::id_, "Variable node ID")
        .def_rw("initial_guess_is_prior",
                &VariableNodeBase::initial_guess_is_prior_,
                "Legacy (True): with a single factor, keep sending the initial guess. False: send zero precision.")
        .def(
          "send_message",
          static_cast<void (VariableNodeBase::*)(size_t)>(&VariableNodeBase::sendMessage),
          nb::arg("edge_index"),
          nb::call_guard<nb::gil_scoped_release>(),
          "Send a message along the specified outgoing edge (by index)")
        .def(
          "send_message_to_factor",
          static_cast<void (VariableNodeBase::*)(const std::string&)>(&VariableNodeBase::sendMessage),
          nb::arg("factor_id"),
          "Send a message toward the factor connected via the provided ID")
        .def(
          "send_all_messages",
          &VariableNodeBase::sendAllMessages,
          nb::call_guard<nb::gil_scoped_release>(),
          "Send messages along all connected edges")
        .def(
          "update_mu_sigma",
          &VariableNodeBase::updateMuAndSigma,
          nb::call_guard<nb::gil_scoped_release>(),
          "Recompute the variable belief by fusing incoming messages")
        .def_rw("send_belief", &VariableNodeBase::send_belief_,
                "Send the belief instead of per-edge messages; factors recover the per-edge messages exactly")
        .def("send_messages", &VariableNodeBase::sendMessages, nb::arg("mask"), nb::call_guard<nb::gil_scoped_release>(),
             "Send on the edges with mask[i] != 0 (one belief computation with send_belief)")
        .def("has_unprocessed_messages",
             &VariableNodeBase::hasUnprocessedMessages,
             "Whether an incoming factor-to-variable message has not been processed yet")
        .def("mark_messages_processed",
             &VariableNodeBase::markMessagesProcessed,
             "Mark every incoming factor-to-variable message processed")
        .def_prop_ro("num_edges", &VariableNodeBase::numberOfEdges, "Number of connected factors");
    }

  } // namespace python
} // namespace gsolver
