// =============================================================================
// gsolver Python bindings - RuleScheduler (learned local send rule run in C++)
// =============================================================================

#include <nanobind/nanobind.h>
#include <nanobind/eigen/dense.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/vector.h>

#include <gsolver/core/generic_scheduler.h>
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
        .def("set_factor_rule", &S::setFactorRule, nb::arg("rule"), "Factor rule (reply / incoming damping heads)")
        .def("last_factor_features", [](const S& s) { return Eigen::MatrixXd(s.lastFactorFeatures()); },
             "Factor-rule inputs at the last tick (rows: factors in graph order, then their edges)")
        .def_rw("record_factor_features", &S::record_factor_features_,
                "Compute the factor-rule inputs every tick even without an active factor rule")
        .def_prop_ro("ticks", &S::ticks)
        .def_prop_ro("num_edges", &S::numberOfEdges);
    }

    void bind_rule_scheduler(nb::module_& m) {
      m.attr("RULE_FEATURE_NAMES") = kFeatureNames;
      m.attr("FACTOR_FEATURE_NAMES") = kFactorFeatureNames;
      nb::class_<LogisticHead>(m, "LogisticHead", "out = scale * sigmoid(intercept + coef . x[features])")
        .def(nb::init<>())
        .def_rw("active", &LogisticHead::active)
        .def_rw("features", &LogisticHead::features)
        .def_rw("coef", &LogisticHead::coef)
        .def_rw("intercept", &LogisticHead::intercept)
        .def_rw("scale", &LogisticHead::scale);
      nb::class_<FactorRule>(m, "FactorRule", "Factor rule: reply head (probability) and incoming damping head")
        .def(nb::init<>())
        .def_rw("reply", &FactorRule::reply)
        .def_rw("damping", &FactorRule::damping);
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
      // ---- v4: GenericScheduler (any variable types; unit-invariant inputs) ----
      m.attr("VARIABLE_INPUT_NAMES_V4") = kVariableInputNamesV4;
      m.attr("FACTOR_INPUT_NAMES_V4")   = kFactorInputNamesV4;
      nb::class_<VariableRule>(m, "VariableRule", "Variable rule: send head (or fixed_send) and incoming damping head")
        .def(nb::init<>())
        .def_rw("send", &VariableRule::send)
        .def_rw("damping", &VariableRule::damping)
        .def_rw("fixed_send", &VariableRule::fixed_send)
        .def_rw("max_silence", &VariableRule::max_silence);
      using G = GenericScheduler;
      nb::class_<G>(m, "GenericScheduler", "Learned local rules on any factor graph (core/generic_scheduler.h)")
        .def(nb::init<FactorGraph&, VariableRule, FactorRule, unsigned>(), nb::arg("graph"), nb::arg("variable_rule"),
             nb::arg("factor_rule"), nb::arg("seed") = 0, nb::keep_alive<1, 2>())
        .def("reset", &G::reset)
        .def("features", &G::variableFeatures, nb::arg("update_state") = true, nb::call_guard<nb::gil_scoped_release>(),
             "Variable rule inputs (edges x len(VARIABLE_INPUT_NAMES_V4))")
        .def(
          "tick",
          [](G& s, const Eigen::VectorXd& uniforms) {
            nb::gil_scoped_release release;
            const TickCounts c = s.tick(uniforms);
            return std::make_pair(c.variable_to_factor, c.factor_to_variable);
          },
          nb::arg("uniforms") = Eigen::VectorXd())
        .def(
          "tick_masks",
          [](G& s, const std::vector<std::vector<int>>& masks) {
            nb::gil_scoped_release release;
            const TickCounts c = s.tickMasks(masks);
            return std::make_pair(c.variable_to_factor, c.factor_to_variable);
          },
          nb::arg("masks"))
        .def("last_factor_features", [](const G& s) { return Eigen::MatrixXd(s.lastFactorFeatures()); })
        .def_rw("record_factor_features", &G::record_factor_features_)
        .def_prop_ro("ticks", &G::ticks)
        .def_prop_ro("num_edges", &G::numberOfEdges)
        .def("set_edge_weights", &G::setEdgeWeights, nb::arg("weights"),
             "Per-edge message costs, one per (variable, edge) row; [] = 1 each")
        .def("set_active_edges", &G::setActiveEdges, nb::arg("active"),
             "Active edges, one flag per (variable, edge) row ([] = all): inactive edges never send")
        .def("weighted_counts", &G::weightedCounts, "Cost of the last tick's messages: (v2f, f2v)");

      bind_rule_scheduler_type<SE2Pose>(m, "RuleSchedulerSE2");
      bind_rule_scheduler_type<SE3Pose>(m, "RuleSchedulerSE3");
    }

  } // namespace python
} // namespace gsolver
