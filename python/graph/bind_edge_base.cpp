// =============================================================================
// gsolver Python bindings - EdgeBase
// =============================================================================
#include <nanobind/nanobind.h>
#include <nanobind/stl/shared_ptr.h>
#include <nanobind/stl/string.h>

#include <sstream>

#include <gsolver/graph/edge_base.h>

namespace nb = nanobind;

namespace gsolver {
  namespace python {

    void bind_edge_base(nb::module_& m) {
      nb::enum_<EdgeSide>(m, "EdgeSide", "Which half of an edge this object is")
        .value("LOCAL", EdgeSide::LOCAL, "Factor and variable in the same graph")
        .value("FACTOR_SIDE", EdgeSide::FACTOR_SIDE, "Held by the local factor; the variable is in another graph")
        .value("VARIABLE_SIDE", EdgeSide::VARIABLE_SIDE, "Held by the local variable; the factor is in another graph");

      nb::class_<EdgeBase>(m, "EdgeBase", "Base class for graph edges")
        .def_prop_ro("side", [](const EdgeBase& edge) { return edge.side_; }, "Which half of an edge this is")
        .def_prop_ro("remote_id", [](const EdgeBase& edge) { return edge.remote_id_; }, "Graph holding the other half, or -1")

        // Properties
        .def_prop_ro(
          "variable_id",
          [](const EdgeBase& edge) { return edge.variable_id_; },
          "Variable ID associated with the edge")
        .def_prop_ro(
          "factor_id",
          [](const EdgeBase& edge) { return edge.factor_id_; },
          "Factor ID associated with the edge")
        .def("__repr__", [](const EdgeBase& edge) {
          std::ostringstream oss;
          oss << "EdgeBase(variable_id='" << edge.variable_id_ << "', factor_id='" << edge.factor_id_ << "')";
          return oss.str();
        });
    }

  } // namespace python
} // namespace gsolver
