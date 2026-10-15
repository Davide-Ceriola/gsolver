// Boundary edges: payload encoding, registry checks, and a graph split across two
// "robots" that must reproduce the single-graph solution when payloads are exchanged
// after each solver phase.
#include <gtest/gtest.h>

#include <vector>

#include <gsolver/core/gbp_solver.h>
#include <gsolver/graph/factor_graph.h>
#include <gsolver/graph/types/all_types.h>

#include "se2_test_utils.h"

using namespace gsolver;
using namespace gsolver::test;

namespace {

  // Random symmetric positive-definite matrix
  template <int N>
  Eigen::Matrix<double, N, N> randomSpd(std::mt19937& rng) {
    std::normal_distribution<double> n(0.0, 1.0);
    Eigen::Matrix<double, N, N> A;
    for (int r = 0; r < N; ++r) {
      for (int c = 0; c < N; ++c) {
        A(r, c) = n(rng);
      }
    }
    return A * A.transpose() + Eigen::Matrix<double, N, N>::Identity();
  }

  // Send a factor-side message through a payload; the variable side must receive it intact
  template <typename VariableType>
  void expectPayloadRoundTrip(const typename VariableType::EstimateType& mu,
                              int expected_size,
                              std::mt19937& rng,
                              const std::function<bool(const typename VariableType::EstimateType&,
                                                       const typename VariableType::EstimateType&)>& same_mean) {
    Edge<VariableType> sender, receiver;
    sender.side_   = EdgeSide::FACTOR_SIDE;
    receiver.side_ = EdgeSide::VARIABLE_SIDE;
    sender.factor_to_variable_message_.mu_     = mu;
    sender.factor_to_variable_message_.Lambda_ = randomSpd<VariableType::LieDim>(rng);

    ASSERT_EQ(sender.payloadSize(), expected_size);
    std::vector<double> payload(sender.payloadSize());
    sender.writeOutgoingPayload(payload.data());
    receiver.readIncomingPayload(payload.data());

    EXPECT_TRUE(same_mean(receiver.factor_to_variable_message_.mu_, mu));
    EXPECT_TRUE(receiver.factor_to_variable_message_.Lambda_.isApprox(sender.factor_to_variable_message_.Lambda_, 1e-15));
  }

} // namespace

// ---------------------------------------------------------------------------
// Payloads
// ---------------------------------------------------------------------------

TEST(BoundaryEdge, PayloadRoundTripsForEveryVariableType) {
  std::mt19937 rng(20);
  auto isometry2 = [](const Isometry2d& a, const Isometry2d& b) { return a.isApprox(b, 1e-12); };
  auto isometry3 = [](const Eigen::Isometry3d& a, const Eigen::Isometry3d& b) { return a.isApprox(b, 1e-12); };

  // mean + upper triangle of Lambda
  expectPayloadRoundTrip<SE2Pose>(se2(1.5, -2.0, 2.9), 3 + 6, rng, isometry2);
  expectPayloadRoundTrip<SE2Point>(Eigen::Vector2d(3.0, -1.0), 2 + 3, rng, [](auto& a, auto& b) { return a == b; });
  expectPayloadRoundTrip<SE3Point>(Eigen::Vector3d(1.0, 2.0, 3.0), 3 + 6, rng, [](auto& a, auto& b) { return a == b; });

  Eigen::Isometry3d pose = Eigen::Isometry3d::Identity();
  pose.translation()     = Eigen::Vector3d(1.0, -2.0, 0.5);
  pose.linear()          = Eigen::AngleAxisd(1.2, Eigen::Vector3d(1, 2, 3).normalized()).toRotationMatrix();
  expectPayloadRoundTrip<SE3Pose>(pose, 6 + 21, rng, isometry3);

  SE3PoseVelValue posevel{pose, (Eigen::Vector<double, 6>() << 1, 2, 3, 4, 5, 6).finished()};
  expectPayloadRoundTrip<SE3PoseVel>(posevel, 12 + 78, rng, [](const SE3PoseVelValue& a, const SE3PoseVelValue& b) {
    return a.pose.isApprox(b.pose, 1e-12) && a.vel == b.vel;
  });
}

TEST(BoundaryEdge, EachSideSendsAndReceivesTheRightMessage) {
  // FACTOR_SIDE sends factor-to-variable and receives variable-to-factor; VARIABLE_SIDE the reverse
  std::mt19937 rng(21);
  Edge<SE2Pose> factor_side, variable_side;
  factor_side.side_   = EdgeSide::FACTOR_SIDE;
  variable_side.side_ = EdgeSide::VARIABLE_SIDE;
  std::vector<double> payload(factor_side.payloadSize());

  variable_side.variable_to_factor_message_.mu_     = se2(1, 2, 0.3);
  variable_side.variable_to_factor_message_.Lambda_ = randomSpd<3>(rng);
  variable_side.writeOutgoingPayload(payload.data());
  factor_side.readIncomingPayload(payload.data());
  EXPECT_TRUE(factor_side.variable_to_factor_message_.mu_.isApprox(se2(1, 2, 0.3), 1e-12));
  EXPECT_TRUE(factor_side.variable_to_factor_message_.Lambda_.isApprox(variable_side.variable_to_factor_message_.Lambda_));
}

TEST(BoundaryEdge, LocalEdgesHaveNoPayload) {
  Edge<SE2Pose> local;
  std::vector<double> payload(local.payloadSize());
  EXPECT_THROW(local.writeOutgoingPayload(payload.data()), std::logic_error);
  EXPECT_THROW(local.readIncomingPayload(payload.data()), std::logic_error);
}

// ---------------------------------------------------------------------------
// Registry
// ---------------------------------------------------------------------------

TEST(BoundaryEdge, RegistryRejectsInvalidBoundaries) {
  const Eigen::Matrix3d Sigma = Eigen::Matrix3d::Identity();
  FactorGraph graph;
  auto local = std::make_shared<SE2Pose>("A", se2(0, 0, 0), Sigma);
  auto other = std::make_shared<SE2Pose>("B_LOCAL", se2(1, 0, 0), Sigma);
  auto stub  = std::make_shared<SE2Pose>("B", se2(1, 0, 0), Sigma); // stands in for a remote variable
  graph.addVariableNode(local);
  graph.addVariableNode(other);
  graph.addFactorNode(std::make_shared<SE2Bearing>("OBS", 0.0, SE2Bearing::MatrixErrorDim::Identity()), local, stub);
  graph.addFactorNode(std::make_shared<SE2Bearing>("OBS_LOCAL", 0.0, SE2Bearing::MatrixErrorDim::Identity()), local, other);

  EXPECT_THROW(graph.markFactorSideBoundary("MISSING", "B", 1), std::runtime_error);         // no such factor
  EXPECT_THROW(graph.markFactorSideBoundary("OBS", "C", 1), std::runtime_error);             // factor has no edge to C
  EXPECT_THROW(graph.markFactorSideBoundary("OBS_LOCAL", "B_LOCAL", 1), std::runtime_error); // both ends are local

  auto edge = graph.markFactorSideBoundary("OBS", "B", 1);
  EXPECT_EQ(edge->side_, EdgeSide::FACTOR_SIDE);
  EXPECT_EQ(edge->remote_id_, 1);
  EXPECT_EQ(graph.getBoundaryEdge("OBS", "B"), edge);
  EXPECT_THROW(graph.markFactorSideBoundary("OBS", "B", 1), std::runtime_error); // already registered

  EXPECT_THROW(graph.addVariableSideBoundary("MISSING", "REMOTE_OBS", 1), std::runtime_error); // no such variable
  EXPECT_THROW(graph.addVariableSideBoundary("A", "OBS", 1), std::runtime_error);              // factor is local
  auto dangling = graph.addVariableSideBoundary("A", "REMOTE_OBS", 1);
  EXPECT_EQ(dangling->side_, EdgeSide::VARIABLE_SIDE);
  EXPECT_EQ(local->edges_.size(), 3u);
  EXPECT_TRUE(local->edges_.back()->factor_to_variable_message_.Lambda_.isZero(0.0)); // nothing received yet
  EXPECT_THROW(graph.addVariableSideBoundary("A", "REMOTE_OBS", 1), std::runtime_error); // already exists

  EXPECT_EQ(graph.boundary_edges_.size(), 2u);
  EXPECT_EQ(graph.getBoundaryEdge("OBS_LOCAL", "B_LOCAL"), nullptr);
}

// ---------------------------------------------------------------------------
// A two-robot graph, solved whole and split
// ---------------------------------------------------------------------------

namespace {

  const Eigen::Matrix3d kInitialSigma  = Eigen::Matrix3d::Identity() * 1e8;
  const Eigen::Matrix3d kPriorSigma    = Eigen::Vector3d(1e-2, 1e-2, 1e-3).asDiagonal();
  const Eigen::Matrix3d kOdometrySigma = Eigen::Vector3d(1e-2, 1e-2, 1e-3).asDiagonal();
  const double kBearingSigma2          = 1e-4;

  struct TwoRobotProblem {
    int steps = 6;
    std::vector<std::vector<Isometry2d>> truth;   // [robot][step]
    std::vector<std::vector<Isometry2d>> guesses; // [robot][step], shared by both builds
  };

  TwoRobotProblem makeProblem(std::mt19937& rng) {
    TwoRobotProblem p;
    p.truth = {{se2(0, 0, 0)}, {se2(6, 3, 2.5)}};
    for (int r = 0; r < 2; ++r) {
      for (int t = 1; t < p.steps; ++t) {
        p.truth[r].push_back(p.truth[r].back() * se2(1.0, 0.0, r == 0 ? 0.2 : -0.15));
      }
      p.guesses.emplace_back();
      for (int t = 0; t < p.steps; ++t) {
        p.guesses[r].push_back(perturb(rng, p.truth[r][t], 0.2, 0.05));
      }
    }
    return p;
  }

  std::string poseId(int robot, int t) {
    return "R" + std::to_string(robot) + "_POSE_" + std::to_string(t);
  }
  std::string obsId(int observer, int t) {
    return "R" + std::to_string(observer) + "_OBS_" + std::to_string(t);
  }

  double bearing(const Isometry2d& from, const Isometry2d& to) {
    const Eigen::Vector2d p = from.linear().transpose() * (to.translation() - from.translation());
    return atan2(p.y(), p.x());
  }

  // Robot r's own variables and factors: poses, prior on the first, odometry
  std::vector<std::shared_ptr<SE2Pose>> addRobot(FactorGraph& graph, const TwoRobotProblem& p, int r) {
    std::vector<std::shared_ptr<SE2Pose>> poses;
    for (int t = 0; t < p.steps; ++t) {
      poses.push_back(std::make_shared<SE2Pose>(poseId(r, t), p.guesses[r][t], kInitialSigma));
      graph.addVariableNode(poses.back());
    }
    graph.addFactorNode(std::make_shared<SE2Prior>("R" + std::to_string(r) + "_PRIOR", p.truth[r][0], kPriorSigma), poses[0]);
    for (int t = 0; t + 1 < p.steps; ++t) {
      const Isometry2d z = p.truth[r][t].inverse() * p.truth[r][t + 1];
      graph.addFactorNode(std::make_shared<SE2Odometry>("R" + std::to_string(r) + "_ODOM_" + std::to_string(t), z, kOdometrySigma),
                          poses[t],
                          poses[t + 1]);
    }
    return poses;
  }

  std::shared_ptr<SE2Bearing> makeObservation(const TwoRobotProblem& p, int observer, int t) {
    const int target = 1 - observer;
    return std::make_shared<SE2Bearing>(obsId(observer, t),
                                        bearing(p.truth[observer][t], p.truth[target][t]),
                                        SE2Bearing::MatrixErrorDim::Constant(kBearingSigma2));
  }

  // Everything in one graph; each robot observes the other at every step
  struct Whole {
    FactorGraph graph;
    std::vector<std::vector<std::shared_ptr<SE2Pose>>> poses;
  };

  void buildWhole(Whole& w, const TwoRobotProblem& p, bool initial_guess_is_prior) {
    w.graph.initial_guess_is_prior_ = initial_guess_is_prior;
    w.poses = {addRobot(w.graph, p, 0), addRobot(w.graph, p, 1)};
    for (int t = 0; t < p.steps; ++t) {
      for (int observer = 0; observer < 2; ++observer) {
        w.graph.addFactorNode(makeObservation(p, observer, t), w.poses[observer][t], w.poses[1 - observer][t]);
      }
    }
  }

  // One graph per robot. Each observation factor lives with its observer; the observed pose is
  // a stub there, and the observed robot holds the variable side of that edge.
  struct Split {
    FactorGraph graphs[2];
    std::vector<std::vector<std::shared_ptr<SE2Pose>>> poses;
  };

  void buildSplit(Split& s, const TwoRobotProblem& p, bool initial_guess_is_prior) {
    for (int r = 0; r < 2; ++r) {
      s.graphs[r].initial_guess_is_prior_ = initial_guess_is_prior;
      s.poses.push_back(addRobot(s.graphs[r], p, r));
    }
    // Same order as buildWhole, so every variable's edges are created in the same order
    for (int t = 0; t < p.steps; ++t) {
      for (int observer = 0; observer < 2; ++observer) {
        const int target = 1 - observer;
        auto stub        = std::make_shared<SE2Pose>(poseId(target, t), p.guesses[target][t], kInitialSigma);
        stub->initial_guess_is_prior_ = initial_guess_is_prior;
        s.graphs[observer].addFactorNode(makeObservation(p, observer, t), s.poses[observer][t], stub);
        s.graphs[observer].markFactorSideBoundary(obsId(observer, t), poseId(target, t), target);
        s.graphs[target].addVariableSideBoundary(poseId(target, t), obsId(observer, t), observer);
      }
    }
  }

  // Perfect channel: copy every boundary edge's outgoing payload into its counterpart
  void exchange(FactorGraph& from, FactorGraph& to) {
    std::vector<double> payload;
    for (const auto& edge : from.boundary_edges_) {
      std::shared_ptr<EdgeBase> counterpart = to.getBoundaryEdge(edge->factor_id_, edge->variable_id_);
      ASSERT_NE(counterpart, nullptr) << edge->factor_id_ << " - " << edge->variable_id_;
      payload.resize(edge->payloadSize());
      edge->writeOutgoingPayload(payload.data());
      counterpart->readIncomingPayload(payload.data());
    }
  }

  void exchangeBothWays(Split& s) {
    exchange(s.graphs[0], s.graphs[1]);
    exchange(s.graphs[1], s.graphs[0]);
  }

  void expectSplitMatchesWhole(bool initial_guess_is_prior) {
    std::mt19937 rng(22);
    const TwoRobotProblem problem = makeProblem(rng);
    Whole whole;
    Split split;
    buildWhole(whole, problem, initial_guess_is_prior);
    buildSplit(split, problem, initial_guess_is_prior);

    // Each graph has a boundary half for every observation: 2 per step (one of each side)
    ASSERT_EQ(split.graphs[0].boundary_edges_.size(), 2u * problem.steps);
    ASSERT_EQ(split.graphs[1].boundary_edges_.size(), 2u * problem.steps);

    // Compared at convergence: payloads round angles through (x, y, theta) and symmetrize
    // Lambda, so mid-run the two differ by rounding that loopy GBP can amplify (about 1e-5
    // after 100 iterations); both reach the same fixed point. Exchanging raw Message copies
    // instead matches at every iteration.
    GbpSolver solver(SolverScheduleType::SYNCHRONOUS);
    for (int iteration = 0; iteration < 300; ++iteration) {
      solver.performIteration(whole.graph);

      for (auto& graph : split.graphs) {
        solver.factorPhase(graph);
      }
      exchangeBothWays(split);
      for (auto& graph : split.graphs) {
        solver.variablePhase(graph);
      }
      exchangeBothWays(split);
    }

    for (int r = 0; r < 2; ++r) {
      for (int t = 0; t < problem.steps; ++t) {
        const auto& w = whole.poses[r][t];
        const auto& s = split.poses[r][t];
        EXPECT_LT(se2Error(w->mu_, s->mu_).norm(), 1e-9) << poseId(r, t);
        EXPECT_LT((w->Sigma_ - s->Sigma_).norm(), 1e-9 * w->Sigma_.norm()) << poseId(r, t);
        // And the solution itself is right: noise-free measurements, so ground truth
        EXPECT_LT(se2Error(problem.truth[r][t], s->mu_).norm(), 1e-4) << poseId(r, t);
      }
    }
  }

} // namespace

TEST(BoundaryEdge, SplitGraphMatchesWholeGraph) {
  expectSplitMatchesWhole(false); // the Robot Web setting
}

TEST(BoundaryEdge, SplitGraphMatchesWholeGraphInLegacyMode) {
  expectSplitMatchesWhole(true);
}
