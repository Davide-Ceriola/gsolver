#include <gtest/gtest.h>

#include <gsolver/core/gbp_solver.h>
#include <gsolver/core/robust_kernels.h>
#include <gsolver/graph/factor_graph.h>
#include <gsolver/graph/types/all_types.h>

#include "se2_test_utils.h"

using namespace gsolver;
using namespace gsolver::test;

TEST(HuberWeight, IsOneWithinTheThresholdAndContinuousAtIt) {
  EXPECT_EQ(huberWeight(0.0, 2.0), 1.0);
  EXPECT_EQ(huberWeight(1.99, 2.0), 1.0);
  EXPECT_EQ(huberWeight(2.0, 2.0), 1.0);
  EXPECT_NEAR(huberWeight(2.0 + 1e-9, 2.0), 1.0, 1e-8);
}

TEST(HuberWeight, WeightedEnergyIsTheHuberEnergy) {
  const double n = 2.0;
  for (double m : {2.5, 4.0, 10.0, 100.0}) {
    const double w = huberWeight(m, n);
    EXPECT_NEAR(w * m * m, 2.0 * n * m - n * n, 1e-9) << "m = " << m;
    EXPECT_LT(w, 1.0);
    EXPECT_GT(w, 0.0);
  }
  EXPECT_GT(huberWeight(3.0, n), huberWeight(6.0, n)); // farther out, less weight
}

TEST(RobustFactor, WeightScalesTheFactorInformation) {
  // A prior 1 m away from the pose, with sigma 0.1 m: Mahalanobis distance 10
  const Eigen::Matrix3d Sigma = Eigen::Vector3d(0.01, 0.01, 0.01).asDiagonal();
  FactorGraph plain, robust;
  for (FactorGraph* g : {&plain, &robust}) {
    auto pose = std::make_shared<SE2Pose>("P", se2(0, 0, 0), Eigen::Matrix3d::Identity());
    g->addVariableNode(pose);
    g->addFactorNode(std::make_shared<SE2Prior>("PRIOR", se2(1, 0, 0), Sigma), pose);
  }
  auto& robust_factor                  = *robust.factor_nodes_[0];
  robust_factor.robust_kernel_.kernel    = RobustKernel::HUBER;
  robust_factor.robust_kernel_.threshold = 2.0;

  plain.factor_nodes_[0]->updateFactor();
  robust_factor.updateFactor();

  const double expected = huberWeight(10.0, 2.0);
  EXPECT_EQ(plain.factor_nodes_[0]->robust_weight_, 1.0);
  EXPECT_NEAR(robust_factor.robust_weight_, expected, 1e-12);

  auto& p = static_cast<SE2Prior&>(*plain.factor_nodes_[0]);
  auto& r = static_cast<SE2Prior&>(robust_factor);
  EXPECT_TRUE(r.eta_.isApprox(p.eta_ * expected, 1e-12));
  EXPECT_TRUE(r.Lambdaprime_.isApprox(p.Lambdaprime_ * expected, 1e-12));
}

TEST(RobustFactor, InliersAreUntouched) {
  const Eigen::Matrix3d Sigma = Eigen::Vector3d(0.01, 0.01, 0.01).asDiagonal();
  FactorGraph g;
  auto pose = std::make_shared<SE2Pose>("P", se2(0, 0, 0), Eigen::Matrix3d::Identity());
  g.addVariableNode(pose);
  g.addFactorNode(std::make_shared<SE2Prior>("PRIOR", se2(0.1, 0, 0), Sigma), pose); // 1 sigma
  g.factor_nodes_[0]->robust_kernel_.kernel = RobustKernel::HUBER;
  g.factor_nodes_[0]->updateFactor();
  EXPECT_EQ(g.factor_nodes_[0]->robust_weight_, 1.0);
}

namespace {

  // One pose, four priors at the truth and one gross outlier 5 m away (50 sigma)
  double outlierPull(bool robust) {
    const Eigen::Matrix3d Sigma = Eigen::Vector3d(0.01, 0.01, 0.0001).asDiagonal();
    FactorGraph g;
    g.initial_guess_is_prior_ = false;
    auto pose                 = std::make_shared<SE2Pose>("P", se2(0.5, 0.5, 0.0), Eigen::Matrix3d::Identity() * 1e8);
    g.addVariableNode(pose);
    for (int k = 0; k < 4; ++k) {
      g.addFactorNode(std::make_shared<SE2Prior>("INLIER_" + std::to_string(k), se2(0, 0, 0), Sigma), pose);
    }
    g.addFactorNode(std::make_shared<SE2Prior>("OUTLIER", se2(5, 0, 0), Sigma), pose);
    if (robust) {
      for (auto& factor : g.factor_nodes_) {
        factor->robust_kernel_ = RobustKernelConfig{RobustKernel::HUBER, 2.0};
      }
    }
    GbpSolver solver(SolverScheduleType::SYNCHRONOUS);
    for (int i = 0; i < 50; ++i) {
      solver.performIteration(g);
    }
    if (robust) {
      EXPECT_LT(g.getFactorNode("OUTLIER")->robust_weight_, 0.1);
      for (int k = 0; k < 4; ++k) {
        EXPECT_GT(g.getFactorNode("INLIER_" + std::to_string(k))->robust_weight_, 0.9);
      }
    }
    return pose->mu_.translation().norm();
  }

} // namespace

TEST(RobustFactor, HuberLimitsTheOutlierPull) {
  const double plain  = outlierPull(false);
  const double robust = outlierPull(true);
  EXPECT_NEAR(plain, 1.0, 0.01); // the outlier gets a fifth of the vote: 5 m / 5
  EXPECT_LT(robust, 0.15);       // down-weighted to a few percent
}
