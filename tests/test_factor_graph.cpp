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

// ---------------------------------------------------------------------------
// Removal
// ---------------------------------------------------------------------------

namespace {

  const Eigen::Matrix3d kSigma2d = Eigen::Vector3d(0.01, 0.01, 0.001).asDiagonal();

  Isometry2d pose2d(double x, double y, double theta) {
    return expMapSE2(Eigen::Vector3d(x, y, theta));
  }

  // prior - P0 - ODOM_0 - P1 - ODOM_1 - P2, plus a bearing from P2 to a remote pose (factor side)
  // and a dangling edge on P0 from a remote factor (variable side)
  struct SmallGraph {
    FactorGraph graph;
    std::vector<std::shared_ptr<SE2Pose>> poses;

    SmallGraph() {
      graph.initial_guess_is_prior_ = false;
      for (int k = 0; k < 3; ++k) {
        poses.push_back(std::make_shared<SE2Pose>("P" + std::to_string(k), pose2d(k, 0, 0), Eigen::Matrix3d::Identity()));
        graph.addVariableNode(poses.back());
      }
      graph.addFactorNode(std::make_shared<SE2Prior>("PRIOR", pose2d(0, 0, 0), kSigma2d), poses[0]);
      graph.addFactorNode(std::make_shared<SE2Odometry>("ODOM_0", pose2d(1, 0, 0), kSigma2d), poses[0], poses[1]);
      graph.addFactorNode(std::make_shared<SE2Odometry>("ODOM_1", pose2d(1, 0, 0), kSigma2d), poses[1], poses[2]);
      auto remote = std::make_shared<SE2Pose>("REMOTE", pose2d(2, 3, 0), Eigen::Matrix3d::Identity());
      graph.addFactorNode(std::make_shared<SE2Bearing>("OBS", M_PI / 2, SE2Bearing::MatrixErrorDim::Constant(1e-4)), poses[2], remote);
      graph.markFactorSideBoundary("OBS", "REMOTE", 1);
      graph.addVariableSideBoundary("P0", "REMOTE_OBS", 1);
    }
  };

} // namespace

TEST(FactorGraphRemoval, RemoveFactorDetachesItFromItsVariables) {
  SmallGraph g;
  g.graph.removeFactorNode("ODOM_0");
  EXPECT_EQ(g.graph.factor_nodes_.size(), 3u);
  EXPECT_EQ(g.graph.getFactorNode("ODOM_0"), nullptr);
  EXPECT_EQ(g.poses[0]->edges_.size(), 2u); // PRIOR + dangling
  EXPECT_EQ(g.poses[1]->edges_.size(), 1u); // ODOM_1
  EXPECT_THROW(g.graph.removeFactorNode("ODOM_0"), std::runtime_error);
}

TEST(FactorGraphRemoval, RemovingAFactorSideBoundaryRemovesItsFactor) {
  SmallGraph g;
  g.graph.removeBoundaryEdge("OBS", "REMOTE");
  EXPECT_EQ(g.graph.getFactorNode("OBS"), nullptr);
  EXPECT_EQ(g.graph.getBoundaryEdge("OBS", "REMOTE"), nullptr);
  EXPECT_EQ(g.poses[2]->edges_.size(), 1u); // ODOM_1
  EXPECT_EQ(g.graph.boundary_edges_.size(), 1u);
}

TEST(FactorGraphRemoval, RemovingAVariableSideBoundaryDropsTheEdge) {
  SmallGraph g;
  g.graph.removeBoundaryEdge("REMOTE_OBS", "P0");
  EXPECT_EQ(g.graph.getBoundaryEdge("REMOTE_OBS", "P0"), nullptr);
  EXPECT_EQ(g.poses[0]->edges_.size(), 2u); // PRIOR + ODOM_0
  EXPECT_EQ(g.graph.factor_nodes_.size(), 4u);
  EXPECT_THROW(g.graph.removeBoundaryEdge("REMOTE_OBS", "P0"), std::runtime_error);
}

TEST(FactorGraphRemoval, RemoveVariableTakesItsFactorsAndBoundaryHalves) {
  SmallGraph g;
  g.graph.removeVariableNode("P0");
  EXPECT_EQ(g.graph.getVariableNode("P0"), nullptr);
  EXPECT_EQ(g.graph.getFactorNode("PRIOR"), nullptr);
  EXPECT_EQ(g.graph.getFactorNode("ODOM_0"), nullptr);
  EXPECT_EQ(g.graph.getBoundaryEdge("REMOTE_OBS", "P0"), nullptr);
  EXPECT_EQ(g.poses[1]->edges_.size(), 1u);
  EXPECT_EQ(g.graph.boundary_edges_.size(), 1u); // OBS on P2 remains

  g.graph.removeVariableNode("P2");
  EXPECT_EQ(g.graph.getFactorNode("OBS"), nullptr);
  EXPECT_TRUE(g.graph.boundary_edges_.empty());
  EXPECT_EQ(g.graph.variable_nodes_.size(), 1u);
  EXPECT_TRUE(g.poses[1]->edges_.empty());
}

TEST(FactorGraphRemoval, SolverRunsAfterRemoval) {
  SmallGraph g;
  GbpSolver solver(SolverScheduleType::SYNCHRONOUS);
  for (int i = 0; i < 5; ++i) {
    solver.performIteration(g.graph);
  }
  g.graph.removeVariableNode("P0");
  g.graph.addFactorNode(std::make_shared<SE2Prior>("PRIOR_1", g.poses[1]->mu_, kSigma2d), g.poses[1]);
  for (int i = 0; i < 20; ++i) {
    solver.performIteration(g.graph);
  }
  EXPECT_TRUE(g.poses[2]->mu_.translation().allFinite());
  EXPECT_NEAR(g.poses[2]->mu_.translation().x(), 2.0, 1e-3);
}

TEST(GbpSolver, SerialAndParallelGiveIdenticalResults) {
  SmallGraph parallel, serial;
  GbpSolver parallel_solver(SolverScheduleType::SYNCHRONOUS), serial_solver(SolverScheduleType::SYNCHRONOUS);
  serial_solver.parallel_ = false;
  for (int i = 0; i < 20; ++i) {
    parallel_solver.performIteration(parallel.graph);
    serial_solver.performIteration(serial.graph);
  }
  for (int k = 0; k < 3; ++k) {
    EXPECT_TRUE(parallel.poses[k]->mu_.isApprox(serial.poses[k]->mu_, 0.0));
    EXPECT_EQ(parallel.poses[k]->Sigma_, serial.poses[k]->Sigma_);
  }
}

TEST(FactorNode, DisabledFactorsContributeNothing) {
  // P with a prior at 0 and a second prior at 1 m: disabling the second leaves the first alone
  const Eigen::Matrix3d Sigma = Eigen::Vector3d(0.01, 0.01, 0.001).asDiagonal();
  auto build = [&](FactorGraph& g, bool second) {
    g.initial_guess_is_prior_ = false;
    auto p                    = std::make_shared<SE2Pose>("P", pose2d(0.3, 0, 0), Eigen::Matrix3d::Identity() * 1e8);
    g.addVariableNode(p);
    g.addFactorNode(std::make_shared<SE2Prior>("A", pose2d(0, 0, 0), Sigma), p);
    if (second) {
      g.addFactorNode(std::make_shared<SE2Prior>("B", pose2d(1, 0, 0), Sigma), p);
    }
    return p;
  };
  FactorGraph with_disabled, without;
  auto p = build(with_disabled, true), q = build(without, false);
  with_disabled.getFactorNode("B")->enabled_ = false;
  GbpSolver solver(SolverScheduleType::SYNCHRONOUS);
  for (int i = 0; i < 20; ++i) {
    solver.performIteration(with_disabled);
    solver.performIteration(without);
  }
  EXPECT_LT((p->mu_.translation() - q->mu_.translation()).norm(), 1e-9);
  EXPECT_LT((p->Sigma_ - q->Sigma_).norm() / q->Sigma_.norm(), 1e-6);

  with_disabled.getFactorNode("B")->enabled_ = true; // and it counts again
  for (int i = 0; i < 20; ++i) {
    solver.performIteration(with_disabled);
  }
  EXPECT_NEAR(p->mu_.translation().x(), 0.5, 1e-6);
}
