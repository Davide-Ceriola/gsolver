// LieOptions: the group exponential of SE(2)/SE(3) as the chart, and exact transport of
// precisions between tangent spaces. Both off by default; every combination must still solve a
// noise-free pose graph exactly.
#include <gtest/gtest.h>

#include <random>

#include <gsolver/core/gbp_solver.h>
#include <gsolver/graph/factor_graph.h>
#include <gsolver/graph/types/all_types.h>
#include <gsolver/maths/lie_options.h>

#include "se2_test_utils.h"

using namespace gsolver;
using namespace gsolver::test;

namespace {

  // Restores the global options after each test
  struct LieOptionsTest : ::testing::Test {
    void TearDown() override {
      LieOptions::true_exponential = false;
      LieOptions::exact_transport  = false;
    }
  };

  Vector6d randomTwist(std::mt19937& rng, double scale) {
    std::normal_distribution<double> n(0.0, scale);
    Vector6d x;
    for (int i = 0; i < 6; ++i) {
      x[i] = n(rng);
    }
    return x;
  }

} // namespace

TEST_F(LieOptionsTest, TrueExpAndLogAreInverse) {
  std::mt19937 rng(1);
  for (int trial = 0; trial < 200; ++trial) {
    std::normal_distribution<double> n(0.0, 1.0);
    const Eigen::Vector3d x2(n(rng), n(rng), 0.9 * M_PI * std::tanh(n(rng)));
    EXPECT_LT((logSE2True(expSE2True(x2)) - x2).norm(), 1e-10);
    Vector6d x3 = randomTwist(rng, 1.0);
    if (x3.tail<3>().norm() > 3.0) {
      x3.tail<3>() *= 3.0 / x3.tail<3>().norm();
    }
    EXPECT_LT((logSE3True(expSE3True(x3)) - x3).norm(), 1e-8);
  }
  // small angles (series branches)
  const Eigen::Vector3d s2(0.3, -0.2, 1e-10);
  EXPECT_LT((logSE2True(expSE2True(s2)) - s2).norm(), 1e-12);
  Vector6d s3;
  s3 << 0.3, -0.2, 0.1, 1e-7, -2e-7, 1e-7;
  EXPECT_LT((logSE3True(expSE3True(s3)) - s3).norm(), 1e-10);
}

TEST_F(LieOptionsTest, TrueExpDiffersFromSeparateChartOnlyWithRotation) {
  const Eigen::Vector3d pure_translation(1.0, 2.0, 0.0);
  EXPECT_LT((expSE2True(pure_translation).matrix() - expMapSE2(pure_translation).matrix()).norm(), 1e-12);
  const Eigen::Vector3d with_rotation(1.0, 2.0, 0.5);
  EXPECT_GT((expSE2True(with_rotation).matrix() - expMapSE2(with_rotation).matrix()).norm(), 1e-2);
}

TEST_F(LieOptionsTest, ChartChangeJacobianAndTransport) {
  std::mt19937 rng(2);
  for (bool true_exp : {false, true}) {
    LieOptions::true_exponential = true_exp;
    LieOptions::exact_transport  = true;
    const Eigen::Isometry3d a = SE3Pose::boxplus(Eigen::Isometry3d::Identity(), randomTwist(rng, 1.0));
    const Eigen::Isometry3d b = SE3Pose::boxplus(a, randomTwist(rng, 0.3));
    // Same point, no offset: identity
    EXPECT_LT((chartChangeJacobian<SE3Pose>(a, a, Vector6d::Zero()) - Eigen::Matrix<double, 6, 6>::Identity()).norm(), 1e-6);
    // There and back gives the same precision. A Gaussian around a, moved to b's chart, sits at
    // log(b^-1 a) there, which is where the way back starts.
    Eigen::Matrix<double, 6, 6> M = Eigen::Matrix<double, 6, 6>::Random();
    const Eigen::Matrix<double, 6, 6> Lambda = M * M.transpose() + Eigen::Matrix<double, 6, 6>::Identity();
    const auto there = transportPrecision<SE3Pose>(b, a, Lambda);
    const auto back  = transportPrecision<SE3Pose>(a, b, SE3Pose::boxminus(b, a), there);
    EXPECT_LT((back - Lambda).norm() / Lambda.norm(), 1e-6) << "true_exponential " << true_exp;
    EXPECT_GT((there - Lambda).norm() / Lambda.norm(), 1e-3); // it does change between distinct points
    // Off: unchanged
    LieOptions::exact_transport = false;
    EXPECT_EQ((transportPrecision<SE3Pose>(b, a, Lambda) - Lambda).norm(), 0.0);
  }
}

TEST_F(LieOptionsTest, NoiseFreeLoopsSolvedInEveryMode) {
  for (bool true_exp : {false, true}) {
    for (bool transport : {false, true}) {
      for (bool at_belief : {false, true}) {
        LieOptions::true_exponential = true_exp;
        LieOptions::exact_transport  = transport;
        // SE(2): 10 poses around a circle, prior on the first, loop closure
        {
          std::mt19937 rng(3);
          const int n = 10;
          std::vector<Isometry2d> truth{se2(0, 0, 0)};
          for (int k = 1; k < n; ++k) {
            truth.push_back(truth.back() * se2(1.0, 0.0, 2 * M_PI / n));
          }
          FactorGraph g;
          g.initial_guess_is_prior_ = false;
          std::vector<std::shared_ptr<SE2Pose>> poses;
          for (int k = 0; k < n; ++k) {
            auto p = std::make_shared<SE2Pose>("P" + std::to_string(k), perturb(rng, truth[k], 0.2, 0.1), Eigen::Matrix3d::Identity() * 1e4);
            g.addVariableNode(p);
            poses.push_back(p);
          }
          const Eigen::Matrix3d S = Eigen::Matrix3d::Identity() * 1e-2;
          g.addFactorNode(std::make_shared<SE2Prior>("PR", truth[0], S), std::vector<std::string>{"P0"});
          for (int k = 0; k < n; ++k) {
            const int j = (k + 1) % n;
            g.addFactorNode(std::make_shared<SE2Odometry>("O" + std::to_string(k), truth[k].inverse() * truth[j], S),
                            std::vector<std::string>{"P" + std::to_string(k), "P" + std::to_string(j)});
          }
          for (auto& f : g.factor_nodes_) {
            f->linearise_at_belief_ = at_belief;
          }
          GbpSolver solver(SolverScheduleType::SYNCHRONOUS);
          solver.parallel_ = false;
          for (int it = 0; it < 400; ++it) {
            solver.performIteration(g);
          }
          double worst = 0;
          for (int k = 0; k < n; ++k) {
            worst = std::max(worst, se2Error(truth[k], poses[k]->mu_).norm());
          }
          EXPECT_LT(worst, 1e-6) << "SE2 true_exp " << true_exp << " transport " << transport << " at_belief " << at_belief;
        }
        // SE(3): 10 poses on a tilted circle
        {
          std::mt19937 rng(4);
          const int n = 10;
          Vector6d step;
          step << 1.0, 0.0, 0.1, 0.05, 0.1, 2 * M_PI / n;
          std::vector<Eigen::Isometry3d> truth{Eigen::Isometry3d::Identity()};
          for (int k = 1; k < n; ++k) {
            truth.push_back(truth.back() * expMapSE3(step));
          }
          FactorGraph g;
          g.initial_guess_is_prior_ = false;
          std::vector<std::shared_ptr<SE3Pose>> poses;
          for (int k = 0; k < n; ++k) {
            auto p = std::make_shared<SE3Pose>("P" + std::to_string(k), SE3Pose::boxplus(truth[k], randomTwist(rng, 0.05)),
                                               Eigen::Matrix<double, 6, 6>::Identity() * 1e4);
            g.addVariableNode(p);
            poses.push_back(p);
          }
          const Eigen::Matrix<double, 6, 6> S = Eigen::Matrix<double, 6, 6>::Identity() * 1e-2;
          g.addFactorNode(std::make_shared<SE3PosePrior>("PR", truth[0], S), std::vector<std::string>{"P0"});
          for (int k = 0; k < n; ++k) {
            const int j = (k + 1) % n;
            g.addFactorNode(std::make_shared<SE3PosePose>("E" + std::to_string(k), truth[k].inverse() * truth[j], S),
                            std::vector<std::string>{"P" + std::to_string(k), "P" + std::to_string(j)});
          }
          for (auto& f : g.factor_nodes_) {
            f->linearise_at_belief_ = at_belief;
          }
          GbpSolver solver(SolverScheduleType::SYNCHRONOUS);
          solver.parallel_ = false;
          for (int it = 0; it < 400; ++it) {
            solver.performIteration(g);
          }
          double worst = 0;
          for (int k = 0; k < n; ++k) {
            worst = std::max(worst, logMapSE3(truth[k].inverse() * poses[k]->mu_).norm());
          }
          EXPECT_LT(worst, 1e-6) << "SE3 true_exp " << true_exp << " transport " << transport << " at_belief " << at_belief;
        }
      }
    }
  }
}
