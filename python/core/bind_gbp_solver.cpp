// =============================================================================
#include <nanobind/nanobind.h>
// gsolver Python bindings - GbpSolver
// =============================================================================


#include <gsolver/core/gbp_solver.h>

namespace nb = nanobind;

namespace gsolver {
  namespace python {

    void bind_gbp_solver(nb::module_& m) {
      // SolverScheduleType enum
      nb::enum_<SolverScheduleType>(m, "SolverScheduleType", "Solver scheduling type")
        .value("SYNCHRONOUS", SolverScheduleType::SYNCHRONOUS, "Synchronous message passing")
        .value("ASYNCHRONOUS", SolverScheduleType::ASYNCHRONOUS, "Asynchronous message passing")
        .export_values();

      // GbpSolver class
      nb::class_<GbpSolver>(m, "GbpSolver", "Gaussian Belief Propagation Solver")
        .def(nb::init<SolverScheduleType>(), nb::arg("schedule_type"), "Create a GBP solver with the specified schedule type")
        .def_rw("schedule_type", &GbpSolver::schedule_type_, "Solver scheduling type")
        .def_rw("parallel", &GbpSolver::parallel_, "Run the per-node loops with OpenMP")
        .def("perform_iteration",
             &GbpSolver::performIteration,
             nb::arg("factor_graph"),
             "Perform a single iteration of belief propagation")
        .def("factor_phase",
             &GbpSolver::factorPhase,
             nb::arg("factor_graph"),
             "First half of a synchronous iteration: update factors and send factor-to-variable messages")
        .def("variable_phase",
             &GbpSolver::variablePhase,
             nb::arg("factor_graph"),
             "Second half of a synchronous iteration: update beliefs and send variable-to-factor messages")
        .def("perform_attention_iteration",
             &GbpSolver::performAttentionIteration,
             nb::arg("factor_graph"),
             nb::arg("attention_variable_ids"),
             nb::arg("attention_factor_ids"),
             "Perform an attention-weighted iteration")
        .def("solve", &GbpSolver::solve, nb::arg("factor_graph"), "Solve the factor graph until convergence");
    }

  } // namespace python
} // namespace gsolver
