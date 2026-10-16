#include <gtest/gtest.h>

#include <functional>

#include <gsolver/core/gbp_solver.h>
#include <gsolver/core/robust_kernels.h>
#include <gsolver/graph/factor_graph.h>
#include <gsolver/graph/types/all_types.h>

#include "se2_test_utils.h"

using namespace gsolver;
using namespace gsolver::test;

namespace {

  const RobustKernel kAllKernels[] = {RobustKernel::HUBER, RobustKernel::CAUCHY, RobustKernel::GEMAN_MCCLURE, RobustKernel::TUKEY};

  const char* name(RobustKernel k) {
    switch (k) {
      case RobustKernel::HUBER:
        return "Huber";
      case RobustKernel::CAUCHY:
        return "Cauchy";
      case RobustKernel::GEMAN_MCCLURE:
        return "Geman-McClure";
      case RobustKernel::TUKEY:
        return "Tukey";
      default:
        return "none";
    }
  }

} // namespace

TEST(RobustKernels, WeightsAreOneAtZeroAndNeverIncrease) {
  for (RobustKernel k : kAllKernels) {
    EXPECT_EQ(robustWeight(k, 0.0, 2.0), 1.0) << name(k);
    double previous = 1.0;
    for (double m = 0.1; m < 20.0; m += 0.1) {
      const double w = robustWeight(k, m, 2.0);
      EXPECT_LE(w, previous + 1e-15) << name(k) << " m = " << m;
      EXPECT_GE(w, 0.0) << name(k);
      previous = w;
    }
  }
  EXPECT_EQ(robustWeight(RobustKernel::NONE, 100.0, 2.0), 1.0);
}

TEST(RobustKernels, KnownValues) {
  EXPECT_EQ(robustWeight(RobustKernel::HUBER, 1.5, 2.0), 1.0);
  EXPECT_DOUBLE_EQ(robustWeight(RobustKernel::HUBER, 8.0, 2.0), 0.25); // n / m
  EXPECT_DOUBLE_EQ(robustWeight(RobustKernel::CAUCHY, 2.0, 2.0), 0.5);
  EXPECT_DOUBLE_EQ(robustWeight(RobustKernel::GEMAN_MCCLURE, 2.0, 2.0), 0.25);
  EXPECT_DOUBLE_EQ(robustWeight(RobustKernel::TUKEY, 1.0, 2.0), 0.5625); // (1 - 1/4)^2
  EXPECT_EQ(robustWeight(RobustKernel::TUKEY, 2.0, 2.0), 0.0);
}

TEST(RobustKernels, HuberInfluenceIsBoundedAndRedescendersVanish) {
  // psi(m) = w(m) * m: Huber's stays at n, the others go to zero
  EXPECT_DOUBLE_EQ(robustWeight(RobustKernel::HUBER, 50.0, 2.0) * 50.0, 2.0);
  EXPECT_LT(robustWeight(RobustKernel::CAUCHY, 1e4, 2.0) * 1e4, 1e-3);
  EXPECT_LT(robustWeight(RobustKernel::GEMAN_MCCLURE, 1e3, 2.0) * 1e3, 1e-6);
  EXPECT_EQ(robustWeight(RobustKernel::TUKEY, 50.0, 2.0) * 50.0, 0.0);
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
  auto& robust_factor          = *robust.factor_nodes_[0];
  robust_factor.robust_kernel_ = RobustKernelConfig{RobustKernel::HUBER, 2.0};

  plain.factor_nodes_[0]->updateFactor();
  robust_factor.updateFactor();

  EXPECT_EQ(plain.factor_nodes_[0]->robust_weight_, 1.0);
  EXPECT_NEAR(robust_factor.robust_weight_, 0.2, 1e-12); // n / m = 2 / 10

  auto& p = static_cast<SE2Prior&>(*plain.factor_nodes_[0]);
  auto& r = static_cast<SE2Prior&>(robust_factor);
  EXPECT_TRUE(r.eta_.isApprox(p.eta_ * 0.2, 1e-12));
  EXPECT_TRUE(r.Lambdaprime_.isApprox(p.Lambdaprime_ * 0.2, 1e-12));
}

TEST(RobustFactor, InliersAreUntouchedByHuberAndTukey) {
  const Eigen::Matrix3d Sigma = Eigen::Vector3d(0.01, 0.01, 0.01).asDiagonal();
  FactorGraph g;
  auto pose = std::make_shared<SE2Pose>("P", se2(0, 0, 0), Eigen::Matrix3d::Identity());
  g.addVariableNode(pose);
  g.addFactorNode(std::make_shared<SE2Prior>("PRIOR", se2(0.1, 0, 0), Sigma), pose); // 1 sigma
  g.factor_nodes_[0]->robust_kernel_ = RobustKernelConfig{RobustKernel::HUBER, 2.0};
  g.factor_nodes_[0]->updateFactor();
  EXPECT_EQ(g.factor_nodes_[0]->robust_weight_, 1.0);
}

// ---------------------------------------------------------------------------
// GBP converges to the M-estimator
// ---------------------------------------------------------------------------

namespace {

  constexpr double kSigma      = 0.1; // position noise of every prior [m]
  constexpr double kOutlierAt  = 5.0; // 50 sigma away
  constexpr double kThreshold  = 2.0;
  constexpr int kInliers       = 4;

  // One pose; four priors at the origin and one outlier at x = 5 m, all with sigma 0.1 m
  double gbpEstimate(RobustKernel kernel) {
    const Eigen::Matrix3d Sigma = Eigen::Vector3d(kSigma * kSigma, kSigma * kSigma, 1e-4).asDiagonal();
    FactorGraph g;
    g.initial_guess_is_prior_ = false;
    auto pose                 = std::make_shared<SE2Pose>("P", se2(0.0, 0.0, 0.0), Eigen::Matrix3d::Identity() * 1e8);
    g.addVariableNode(pose);
    for (int k = 0; k < kInliers; ++k) {
      g.addFactorNode(std::make_shared<SE2Prior>("INLIER_" + std::to_string(k), se2(0, 0, 0), Sigma), pose);
    }
    g.addFactorNode(std::make_shared<SE2Prior>("OUTLIER", se2(kOutlierAt, 0, 0), Sigma), pose);
    for (auto& factor : g.factor_nodes_) {
      factor->robust_kernel_ = RobustKernelConfig{kernel, kThreshold};
    }
    GbpSolver solver(SolverScheduleType::SYNCHRONOUS);
    for (int i = 0; i < 200; ++i) {
      solver.performIteration(g);
    }
    return pose->mu_.translation().x();
  }

  // The M-estimator, solved directly: sum of psi(residual) = 0 along x, in sigma units.
  // psi(r) = w(|r|) r. Bisection on [0, x_max], where the inliers dominate.
  double mEstimate(RobustKernel kernel) {
    auto psi       = [&](double r) { return robustWeight(kernel, std::abs(r), kThreshold) * r; };
    auto stationary = [&](double x) { return kInliers * psi((x - 0.0) / kSigma) + psi((x - kOutlierAt) / kSigma); };
    double lo = 0.0, hi = 1.0; // stationary(lo) < 0 < stationary(hi) for these kernels
    for (int i = 0; i < 200; ++i) {
      const double mid = 0.5 * (lo + hi);
      (stationary(mid) < 0.0 ? lo : hi) = mid;
    }
    return 0.5 * (lo + hi);
  }

} // namespace

TEST(RobustFactor, WithoutAKernelTheOutlierGetsAFifthOfTheVote) {
  EXPECT_NEAR(gbpEstimate(RobustKernel::NONE), kOutlierAt / (kInliers + 1), 1e-6);
}

TEST(RobustFactor, GbpConvergesToTheMEstimatorOfEachKernel) {
  // Close, not exact: gsolver's factors linearize at their incoming message means, which leave
  // out the factor itself, so each weight comes from a leave-one-out residual. For Huber here
  // the outlier's weight is taken at x = 0 (what the inliers say) instead of x = 0.05, which
  // moves the solution by 1% of the pull. The redescending kernels match to ~1e-9.
  for (RobustKernel k : kAllKernels) {
    const double expected = mEstimate(k);
    EXPECT_NEAR(gbpEstimate(k), expected, 0.02 * std::abs(expected) + 1e-8) << name(k);
  }
  // Huber leaves the outlier a bounded pull, n sigma / inliers; the redescenders nearly none
  EXPECT_NEAR(mEstimate(RobustKernel::HUBER), kThreshold * kSigma / kInliers, 1e-9);
  EXPECT_LT(mEstimate(RobustKernel::CAUCHY), 2e-3);
  EXPECT_LT(mEstimate(RobustKernel::GEMAN_MCCLURE), 1e-4);
  EXPECT_NEAR(mEstimate(RobustKernel::TUKEY), 0.0, 1e-12);
}
