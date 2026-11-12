// =============================================================================
// gsolver Python bindings - RuleScheduler (learned local send rule run in C++)
// =============================================================================

#include <nanobind/nanobind.h>
#include <nanobind/eigen/dense.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/vector.h>

#include <gsolver/core/rule_scheduler.h>
#include <gsolver/graph/types/variables/se2_pose.h>
#include <gsolver/graph/types/variables/se3_pose.h>

namespace nb = nanobind;

namespace gsolver {
  namespace python {

    template <typename VariableType>
    void bind_rule_scheduler_type(nb::module_& m, const char* name) {
      using S = RuleScheduler<VariableType>;
      nb::class_<S>(m, name, "Learned local send rule run in the solver (see core/rule_scheduler.h)")
        .def(nb::init<FactorGraph&, SendRule, unsigned>(), nb::arg("graph"), nb::arg("rule"), nb::arg("seed") = 0,
             nb::keep_alive<1, 2>())
        .def("reset", &S::reset, "Episode start: nothing sent yet")
        .def("features", &S::features, nb::arg("update_state") = true, nb::call_guard<nb::gil_scoped_release>(),
             "All rule inputs (edges x len(feature_names)), stateful like rls_gbp's RuleFeatures")
        .def("probabilities", &S::probabilities, nb::arg("X"), nb::call_guard<nb::gil_scoped_release>())
        .def("apply_damping", &S::applyDamping, nb::arg("X"), nb::call_guard<nb::gil_scoped_release>(),
             "Set every edge's damping from the rule's damping head (no-op without one)")
        .def(
          "tick",
          [](S& s, const Eigen::VectorXd& uniforms) {
            nb::gil_scoped_release release;
            const TickCounts c = s.tick(uniforms);
            return std::make_pair(c.variable_to_factor, c.factor_to_variable);
          },
          nb::arg("uniforms") = Eigen::VectorXd(),
          "One tick with the rule; uniforms: one U(0,1) per edge (empty: drawn here). Returns (#v2f, #f2v)")
        .def(
          "tick_masks",
          [](S& s, const std::vector<std::vector<int>>& masks) {
            nb::gil_scoped_release release;
            const TickCounts c = s.tickMasks(masks);
            return std::make_pair(c.variable_to_factor, c.factor_to_variable);
          },
          nb::arg("masks"), "One tick with given decisions (per variable, per edge). Returns (#v2f, #f2v)")
        .def_prop_ro("ticks", &S::ticks)
        .def_prop_ro("num_edges", &S::numberOfEdges);
    }

    void bind_rule_scheduler(nb::module_& m) {
      m.attr("RULE_FEATURE_NAMES") = kFeatureNames;
      nb::class_<SendRule>(m, "SendRule", "Logistic (coef, intercept) or MLP (mu, sd, W1, b1, w2, b2) send rule")
        .def(nb::init<>())
        .def_rw("features", &SendRule::features, "Indices into RULE_FEATURE_NAMES")
        .def_rw("coef", &SendRule::coef)
        .def_rw("intercept", &SendRule::intercept)
        .def_rw("mu", &SendRule::mu)
        .def_rw("sd", &SendRule::sd)
        .def_rw("W1", &SendRule::W1)
        .def_rw("b1", &SendRule::b1)
        .def_rw("w2", &SendRule::w2)
        .def_rw("b2", &SendRule::b2)
        .def_rw("has_damping", &SendRule::has_damping)
        .def_rw("damping_features", &SendRule::damping_features)
        .def_rw("damping_coef", &SendRule::damping_coef)
        .def_rw("damping_intercept", &SendRule::damping_intercept)
        .def_rw("damping_max", &SendRule::damping_max);
      bind_rule_scheduler_type<SE2Pose>(m, "RuleSchedulerSE2");
      bind_rule_scheduler_type<SE3Pose>(m, "RuleSchedulerSE3");
    }

  } // namespace python
} // namespace gsolver
