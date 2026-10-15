// GBP on small SE(2) graphs: convergence to ground truth with noise-free
// measurements, and marginal covariances against a dense solve on a tree.
#include <gtest/gtest.h>

#include <map>

#include <gsolver/core/gbp_solver.h>
#include <gsolver/graph/factor_graph.h>
#include <gsolver/graph/types/all_types.h>

#include "se2_test_utils.h"

using namespace gsolver;
using namespace gsolver::test;

namespace {

  // Variables start with a very weak covariance, so their initial message carries
  // (almost) no information: everything the solver knows comes from the factors.
  const Eigen::Matrix3d kWeakInitialSigma = Eigen::Matrix3d::Identity() * 1e8;

  const Eigen::Matrix3d kPriorSigma    = Eigen::Vector3d(1e-2, 1e-2, 1e-3).asDiagonal();
  const Eigen::Matrix3d kOdometrySigma = Eigen::Vector3d(1e-2, 1e-2, 1e-3).asDiagonal();

  double maxPoseError(const std::vector<std::shared_ptr<SE2Pose>>& poses, const std::vector<Isometry2d>& truth) {
    double worst = 0.0;
    for (size_t k = 0; k < poses.size(); ++k) {
      worst = std::max(worst, se2Error(truth[k], poses[k]->mu_).norm());
    }
    return worst;
  }

  // ---------------------------------------------------------------------------
  // Single robot: prior + odometry chain (a tree)
  // ---------------------------------------------------------------------------
  struct Chain {
    FactorGraph graph;
    std::vector<std::shared_ptr<SE2Pose>> poses;
    std::vector<Isometry2d> truth;
    std::vector<std::shared_ptr<SE2Odometry>> odometry;
    std::vector<std::shared_ptr<SE2Prior>> priors; // priors_[k] is on pose k
  };

  // Prior on the first pose only, or on every pose
  void buildChain(Chain& c, int length, std::mt19937& rng, bool prior_on_every_pose, bool initial_guess_is_prior = true) {
    c.graph.initial_guess_is_prior_ = initial_guess_is_prior;
    c.truth.push_back(se2(1.0, -2.0, 0.3));
    for (int k = 1; k < length; ++k) {
      c.truth.push_back(c.truth.back() * se2(1.0, 0.1, 0.2));
    }
    for (int k = 0; k < length; ++k) {
      auto pose = std::make_shared<SE2Pose>("POSE_" + std::to_string(k), perturb(rng, c.truth[k], 0.3, 0.1), kWeakInitialSigma);
      c.graph.addVariableNode(pose);
      c.poses.push_back(pose);
    }
    // Built through the string-id overload, so that path is exercised for the SE(2) types too
    for (int k = 0; k < (prior_on_every_pose ? length : 1); ++k) {
      auto prior = std::make_shared<SE2Prior>("PRIOR_" + std::to_string(k), c.truth[k], kPriorSigma);
      c.graph.addFactorNode(prior, std::vector<std::string>{"POSE_" + std::to_string(k)});
      c.priors.push_back(prior);
    }
    for (int k = 0; k + 1 < length; ++k) {
      auto odom = std::make_shared<SE2Odometry>(
        "ODOM_" + std::to_string(k), c.truth[k].inverse() * c.truth[k + 1], kOdometrySigma);
      c.graph.addFactorNode(odom, std::vector<std::string>{"POSE_" + std::to_string(k), "POSE_" + std::to_string(k + 1)});
      c.odometry.push_back(odom);
    }
  }

  // ---------------------------------------------------------------------------
  // Several robots: prior + odometry per robot, inter-robot bearing or range-bearing (loopy)
  // ---------------------------------------------------------------------------
  enum class InterRobotModel { BEARING, RANGE_BEARING };

  struct Team {
    FactorGraph graph;
    std::vector<std::shared_ptr<SE2Pose>> poses;
    std::vector<Isometry2d> truth;
  };

  void buildTeam(Team& team, InterRobotModel model, int robots, int steps, std::mt19937& rng, bool initial_guess_is_prior = true) {
    team.graph.initial_guess_is_prior_ = initial_guess_is_prior;
    std::vector<std::vector<std::shared_ptr<SE2Pose>>> by_robot(robots);
    std::vector<std::vector<Isometry2d>> truth(robots);

    for (int r = 0; r < robots; ++r) {
      const double start_angle = 2.0 * M_PI * r / robots;
      truth[r].push_back(se2(8.0 * cos(start_angle), 8.0 * sin(start_angle), start_angle + M_PI / 2));
      for (int t = 1; t < steps; ++t) {
        truth[r].push_back(truth[r].back() * se2(1.0, 0.0, 0.15));
      }
      for (int t = 0; t < steps; ++t) {
        auto pose = std::make_shared<SE2Pose>(
          "R" + std::to_string(r) + "_POSE_" + std::to_string(t), perturb(rng, truth[r][t], 0.2, 0.05), kWeakInitialSigma);
        team.graph.addVariableNode(pose);
        by_robot[r].push_back(pose);
        team.poses.push_back(pose);
        team.truth.push_back(truth[r][t]);
      }
      team.graph.addFactorNode(std::make_shared<SE2Prior>("R" + std::to_string(r) + "_PRIOR", truth[r][0], kPriorSigma),
                               by_robot[r][0]);
      for (int t = 0; t + 1 < steps; ++t) {
        team.graph.addFactorNode(
          std::make_shared<SE2Odometry>(
            "R" + std::to_string(r) + "_ODOM_" + std::to_string(t), truth[r][t].inverse() * truth[r][t + 1], kOdometrySigma),
          by_robot[r][t],
          by_robot[r][t + 1]);
      }
    }

    // Every robot observes every other robot at every step
    for (int t = 0; t < steps; ++t) {
      for (int i = 0; i < robots; ++i) {
        for (int j = 0; j < robots; ++j) {
          if (i == j) {
            continue;
          }
          const Isometry2d& xi    = truth[i][t];
          const Isometry2d& xj    = truth[j][t];
          const Eigen::Vector2d p = xi.linear().transpose() * (xj.translation() - xi.translation());
          const double bearing    = atan2(p.y(), p.x());
          const std::string id    = "OBS_" + std::to_string(i) + "_" + std::to_string(j) + "_" + std::to_string(t);
          if (model == InterRobotModel::BEARING) {
            auto factor = std::make_shared<SE2Bearing>(id, bearing, SE2Bearing::MatrixErrorDim::Identity() * 1e-4);
            team.graph.addFactorNode(factor, by_robot[i][t], by_robot[j][t]);
          } else {
            const Eigen::Matrix2d Sigma = Eigen::Vector2d(1e-2, 1e-4).asDiagonal();
            auto factor = std::make_shared<SE2RangeBearing>(id, Eigen::Vector2d(p.norm(), bearing), Sigma);
            team.graph.addFactorNode(factor, by_robot[i][t], by_robot[j][t]);
          }
        }
      }
    }
  }

} // namespace

TEST(SE2Gbp, ChainConvergesToGroundTruth) {
  std::mt19937 rng(10);
  Chain chain;
  buildChain(chain, 10, rng, false);
  ASSERT_GT(maxPoseError(chain.poses, chain.truth), 0.1);

  GbpSolver solver(SolverScheduleType::SYNCHRONOUS);
  for (int i = 0; i < 100; ++i) {
    solver.performIteration(chain.graph);
  }
  EXPECT_LT(maxPoseError(chain.poses, chain.truth), 1e-6);
}

namespace {

  // Run GBP on the chain, then compare its marginal covariances with a dense solve built
  // from the same factor Jacobians at the converged estimate. GBP is exact on trees.
  void expectCovariancesMatchDenseSolve(Chain& chain) {
    const int n = static_cast<int>(chain.poses.size());
    GbpSolver solver(SolverScheduleType::SYNCHRONOUS);
    for (int i = 0; i < 100; ++i) {
      solver.performIteration(chain.graph);
    }

    Eigen::MatrixXd H = Eigen::MatrixXd::Zero(3 * n, 3 * n);
    for (size_t k = 0; k < chain.priors.size(); ++k) {
      SE2Prior::VectorErrorDim e;
      SE2Prior::MatrixJacobianDim J;
      SE2Prior::evaluate(chain.poses[k]->mu_, chain.priors[k]->z_, e, J);
      H.block<3, 3>(3 * k, 3 * k) += J.transpose() * kPriorSigma.inverse() * J;
    }
    for (int k = 0; k + 1 < n; ++k) {
      SE2Odometry::VectorErrorDim e;
      SE2Odometry::MatrixJacobianDim J;
      SE2Odometry::evaluate(chain.poses[k]->mu_, chain.poses[k + 1]->mu_, chain.odometry[k]->z_, e, J);
      H.block<6, 6>(3 * k, 3 * k) += J.transpose() * kOdometrySigma.inverse() * J;
    }
    const Eigen::MatrixXd Sigma_dense = H.inverse();

    for (int k = 0; k < n; ++k) {
      const Eigen::Matrix3d expected = Sigma_dense.block<3, 3>(3 * k, 3 * k);
      const Eigen::Matrix3d actual   = chain.poses[k]->Sigma_;
      EXPECT_LT((actual - expected).norm() / expected.norm(), 1e-4) << "pose " << k << "\nGBP\n"
                                                                    << actual << "\ndense\n"
                                                                    << expected;
    }
  }

} // namespace

TEST(SE2Gbp, ChainCovariancesMatchDenseSolve) {
  std::mt19937 rng(11);
  Chain chain;
  // Legacy mode needs every variable to have at least two factors (see the leaf tests below),
  // so this chain has a prior on every pose
  buildChain(chain, 10, rng, true);
  expectCovariancesMatchDenseSolve(chain);
}

// ---------------------------------------------------------------------------
// Single-factor variables send zero precision (initial_guess_is_prior_ = false)
// ---------------------------------------------------------------------------

TEST(SE2Gbp, InitialMessagesAreTheGuessInBothModes) {
  for (bool initial_guess_is_prior : {true, false}) {
    std::mt19937 rng(14);
    Chain chain;
    buildChain(chain, 3, rng, false, initial_guess_is_prior);
    for (const auto& pose : chain.poses) {
      EXPECT_EQ(pose->initial_guess_is_prior_, initial_guess_is_prior);
      for (const auto& edge : pose->edges_) {
        EXPECT_TRUE(edge->variable_to_factor_message_.Lambda_.isApprox(kWeakInitialSigma.inverse()));
        EXPECT_TRUE(edge->variable_to_factor_message_.mu_.isApprox(pose->mu_));
      }
    }
  }
}

TEST(SE2Gbp, LeafSendsZeroPrecisionAtItsBeliefWhenGuessIsNotPrior) {
  std::mt19937 rng(15);
  Chain chain;
  buildChain(chain, 5, rng, false, false);
  GbpSolver solver(SolverScheduleType::SYNCHRONOUS);
  for (int i = 0; i < 20; ++i) {
    solver.performIteration(chain.graph);
  }
  // The last pose is a leaf: its only factor is the last odometry
  const auto& leaf = chain.poses.back();
  ASSERT_EQ(leaf->edges_.size(), 1u);
  const auto& message = leaf->edges_[0]->variable_to_factor_message_;
  EXPECT_TRUE(message.Lambda_.isZero(0.0));
  EXPECT_TRUE(message.mu_.isApprox(leaf->mu_));
}

TEST(SE2Gbp, LeafKeepsInitialMessageInLegacyMode) {
  std::mt19937 rng(15);
  Chain chain;
  buildChain(chain, 5, rng, false, true);
  const auto& leaf               = chain.poses.back();
  const Isometry2d initial_guess = leaf->mu_;
  GbpSolver solver(SolverScheduleType::SYNCHRONOUS);
  for (int i = 0; i < 20; ++i) {
    solver.performIteration(chain.graph);
  }
  const auto& message = leaf->edges_[0]->variable_to_factor_message_;
  EXPECT_TRUE(message.Lambda_.isApprox(kWeakInitialSigma.inverse()));
  EXPECT_TRUE(message.mu_.isApprox(initial_guess));
}

TEST(SE2Gbp, ChainWithLeafConvergesWhenGuessIsNotPrior) {
  std::mt19937 rng(16);
  Chain chain;
  buildChain(chain, 10, rng, false, false);
  GbpSolver solver(SolverScheduleType::SYNCHRONOUS);
  for (int i = 0; i < 100; ++i) {
    solver.performIteration(chain.graph);
  }
  EXPECT_LT(maxPoseError(chain.poses, chain.truth), 1e-6);
}

TEST(SE2Gbp, ChainWithLeafCovariancesMatchDenseSolveWhenGuessIsNotPrior) {
  // The case that fails in legacy mode: prior on the first pose only, so the last pose is a leaf
  std::mt19937 rng(11);
  Chain chain;
  buildChain(chain, 10, rng, false, false);
  expectCovariancesMatchDenseSolve(chain);
}

TEST(SE2Gbp, BearingOnlyTeamConvergesWhenGuessIsNotPrior) {
  std::mt19937 rng(12);
  Team team;
  buildTeam(team, InterRobotModel::BEARING, 4, 5, rng, false);
  GbpSolver solver(SolverScheduleType::SYNCHRONOUS);
  for (int i = 0; i < 300; ++i) {
    solver.performIteration(team.graph);
  }
  EXPECT_LT(maxPoseError(team.poses, team.truth), 1e-4);
}

TEST(SE2Gbp, BearingOnlyTeamConvergesToGroundTruth) {
  std::mt19937 rng(12);
  Team team;
  buildTeam(team, InterRobotModel::BEARING, 4, 5, rng);
  ASSERT_GT(maxPoseError(team.poses, team.truth), 0.1);

  GbpSolver solver(SolverScheduleType::SYNCHRONOUS);
  for (int i = 0; i < 300; ++i) {
    solver.performIteration(team.graph);
  }
  EXPECT_LT(maxPoseError(team.poses, team.truth), 1e-4);
}

TEST(SE2Gbp, RangeBearingTeamConvergesToGroundTruth) {
  std::mt19937 rng(13);
  Team team;
  buildTeam(team, InterRobotModel::RANGE_BEARING, 4, 5, rng);
  ASSERT_GT(maxPoseError(team.poses, team.truth), 0.1);

  GbpSolver solver(SolverScheduleType::SYNCHRONOUS);
  for (int i = 0; i < 300; ++i) {
    solver.performIteration(team.graph);
  }
  EXPECT_LT(maxPoseError(team.poses, team.truth), 1e-4);
}
