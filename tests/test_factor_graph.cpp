#include <gtest/gtest.h>

#include <gsolver/core/gbp_solver.h>
#include <gsolver/graph/factor_graph.h>
#include <gsolver/graph/types/all_types.h>

using namespace gsolver;

namespace {

  struct TwoPointGraph {
    FactorGraph graph;
    std::shared_ptr<SE3Point> a, b;
  };

  // A and B joined by an equality factor, built through either addFactorNode overload
  void buildTwoPointGraph(TwoPointGraph& g, bool typed) {
    g.a = std::make_shared<SE3Point>("A", Eigen::Vector3d(0, 0, 0), Eigen::Matrix3d::Identity());
    g.b = std::make_shared<SE3Point>("B", Eigen::Vector3d(1, 2, 3), Eigen::Matrix3d::Identity() * 4);
    g.graph.addVariableNode(g.a);
    g.graph.addVariableNode(g.b);
    auto equality = std::make_shared<SE3PointEquality>("EQ", Eigen::Matrix3d::Identity() * 0.01);
    if (typed) {
      g.graph.addFactorNode(equality, g.a, g.b);
    } else {
      g.graph.addFactorNode(equality, std::vector<std::string>{"A", "B"});
    }
  }

} // namespace

TEST(FactorGraph, TypedAndStringOverloadsBuildIdenticalGraphs) {
  TwoPointGraph by_id, typed;
  buildTwoPointGraph(by_id, false);
  buildTwoPointGraph(typed, true);

  ASSERT_EQ(typed.graph.factor_nodes_.size(), 1u);
  ASSERT_EQ(typed.a->edges_.size(), 1u);
  ASSERT_EQ(typed.b->edges_.size(), 1u);

  GbpSolver solver(SolverScheduleType::SYNCHRONOUS);
  for (int i = 0; i < 20; ++i) {
    solver.performIteration(by_id.graph);
    solver.performIteration(typed.graph);
  }
  EXPECT_EQ(by_id.a->mu_, typed.a->mu_);
  EXPECT_EQ(by_id.b->mu_, typed.b->mu_);
  EXPECT_EQ(by_id.a->Sigma_, typed.a->Sigma_);
  EXPECT_EQ(by_id.b->Sigma_, typed.b->Sigma_);
}

TEST(FactorGraph, StringOverloadRejectsWrongVariableType) {
  FactorGraph graph;
  graph.addVariableNode(std::make_shared<SE3Pose>("P", Eigen::Isometry3d::Identity(), Eigen::Matrix<double, 6, 6>::Identity()));
  graph.addVariableNode(std::make_shared<SE3Point>("L", Eigen::Vector3d::Zero(), Eigen::Matrix3d::Identity()));
  auto equality = std::make_shared<SE3PointEquality>("EQ", Eigen::Matrix3d::Identity());
  EXPECT_THROW(graph.addFactorNode(equality, std::vector<std::string>{"P", "L"}), std::runtime_error);
}

TEST(FactorGraph, StringOverloadRejectsMissingVariable) {
  FactorGraph graph;
  graph.addVariableNode(std::make_shared<SE3Point>("L", Eigen::Vector3d::Zero(), Eigen::Matrix3d::Identity()));
  auto equality = std::make_shared<SE3PointEquality>("EQ", Eigen::Matrix3d::Identity());
  EXPECT_THROW(graph.addFactorNode(equality, std::vector<std::string>{"L", "MISSING"}), std::runtime_error);
}

TEST(GbpSolver, PhasesMatchPerformIteration) {
  TwoPointGraph whole, phased;
  buildTwoPointGraph(whole, true);
  buildTwoPointGraph(phased, true);

  GbpSolver solver(SolverScheduleType::SYNCHRONOUS);
  for (int i = 0; i < 20; ++i) {
    solver.performIteration(whole.graph);
    solver.factorPhase(phased.graph);
    solver.variablePhase(phased.graph);
  }
  EXPECT_EQ(whole.a->mu_, phased.a->mu_);
  EXPECT_EQ(whole.b->mu_, phased.b->mu_);
  EXPECT_EQ(whole.a->Sigma_, phased.a->Sigma_);
}
