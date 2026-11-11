// SE(3) pose-graph factors (SE3PosePose, SE3PosePrior): Jacobians against finite differences,
// and GBP on a noise-free loop recovering the ground truth.
#include <gtest/gtest.h>

#include <functional>
#include <random>
#include <vector>

#include <gsolver/core/gbp_solver.h>
#include <gsolver/graph/factor_graph.h>
#include <gsolver/graph/types/all_types.h>

using namespace gsolver;

namespace {

  constexpr double kStep = 1e-6;

  Eigen::Isometry3d randomSE3(std::mt19937& rng, double extent = 5.0, double angle = 3.0) {
    std::uniform_real_distribution<double> t(-extent, extent), a(-angle, angle);
    Vector6d x;
    x << t(rng), t(rng), t(rng), a(rng) / std::sqrt(3.0), a(rng) / std::sqrt(3.0), a(rng) / std::sqrt(3.0);
    return expMapSE3(x);
  }

  Eigen::Isometry3d perturb(std::mt19937& rng, const Eigen::Isometry3d& x, double sigma) {
    std::normal_distribution<double> n(0.0, sigma);
    Vector6d tau;
    for (int i = 0; i < 6; ++i) {
      tau[i] = n(rng);
    }
    return SE3Pose::boxplus(x, tau);
  }

  using ErrorFunction = std::function<Eigen::VectorXd(const std::vector<Eigen::Isometry3d>&)>;

  // Central differences under right perturbations x_k <- x_k * exp(tau), as the variables apply
  Eigen::MatrixXd numericJacobian(const ErrorFunction& error_of, const std::vector<Eigen::Isometry3d>& xs) {
    const Eigen::VectorXd e0 = error_of(xs);
    Eigen::MatrixXd J(e0.size(), 6 * xs.size());
    for (size_t k = 0; k < xs.size(); ++k) {
      for (int d = 0; d < 6; ++d) {
        Vector6d tau = Vector6d::Zero();
        tau[d]       = kStep;
        auto plus = xs, minus = xs;
        plus[k]          = SE3Pose::boxplus(xs[k], tau);
        minus[k]         = SE3Pose::boxplus(xs[k], -tau);
        J.col(6 * k + d) = (error_of(plus) - error_of(minus)) / (2 * kStep);
      }
    }
    return J;
  }

} // namespace

TEST(SE3PosePose, JacobianMatchesFiniteDifferences) {
  std::mt19937 rng(1);
  for (int trial = 0; trial < 200; ++trial) {
    const Eigen::Isometry3d xi = randomSE3(rng), xj = randomSE3(rng);
    // Measurement near the true relative pose, as at a GBP linearization point
    const Eigen::Isometry3d z = perturb(rng, xi.inverse() * xj, 0.3);
    SE3PosePose::VectorErrorDim e;
    SE3PosePose::MatrixJacobianDim J;
    SE3PosePose::evaluate(xi, xj, z, e, J);
    auto error_of = [&](const std::vector<Eigen::Isometry3d>& xs) -> Eigen::VectorXd {
      SE3PosePose::VectorErrorDim ee;
      SE3PosePose::MatrixJacobianDim JJ;
      SE3PosePose::evaluate(xs[0], xs[1], z, ee, JJ);
      return ee;
    };
    const Eigen::MatrixXd Jn = numericJacobian(error_of, {xi, xj});
    ASSERT_LT((J - Jn).norm(), 1e-5) << "trial " << trial << "\nanalytic\n" << J << "\nnumeric\n" << Jn;
  }
}

TEST(SE3PosePrior, JacobianMatchesFiniteDifferences) {
  std::mt19937 rng(2);
  for (int trial = 0; trial < 200; ++trial) {
    const Eigen::Isometry3d x = randomSE3(rng), z = perturb(rng, x, 0.3);
    SE3PosePrior::VectorErrorDim e;
    SE3PosePrior::MatrixJacobianDim J;
    SE3PosePrior::evaluate(x, z, e, J);
    auto error_of = [&](const std::vector<Eigen::Isometry3d>& xs) -> Eigen::VectorXd {
      SE3PosePrior::VectorErrorDim ee;
      SE3PosePrior::MatrixJacobianDim JJ;
      SE3PosePrior::evaluate(xs[0], z, ee, JJ);
      return ee;
    };
    const Eigen::MatrixXd Jn = numericJacobian(error_of, {x});
    ASSERT_LT((J - Jn).norm(), 1e-5) << "trial " << trial;
  }
}

TEST(SE3PosePose, ErrorIsZeroAtTheMeasurement) {
  std::mt19937 rng(3);
  const Eigen::Isometry3d xi = randomSE3(rng), xj = randomSE3(rng);
  SE3PosePose::VectorErrorDim e;
  SE3PosePose::MatrixJacobianDim J;
  SE3PosePose::evaluate(xi, xj, xi.inverse() * xj, e, J);
  EXPECT_LT(e.norm(), 1e-12);
}

TEST(SE3PoseGraph, GbpRecoversNoiseFreeLoop) {
  // A loop of n poses with a prior on the first and a loop closure, noise-free measurements,
  // perturbed initial guesses: GBP must converge to the ground truth.
  std::mt19937 rng(4);
  const int n = 12;
  std::vector<Eigen::Isometry3d> truth{Eigen::Isometry3d::Identity()};
  Vector6d step;
  step << 1.0, 0.0, 0.1, 0.05, 0.1, 2.0 * M_PI / n;
  for (int k = 1; k < n; ++k) {
    truth.push_back(truth.back() * expMapSE3(step));
  }
  FactorGraph graph;
  std::vector<std::shared_ptr<SE3Pose>> poses;
  const Eigen::Matrix<double, 6, 6> weak  = Eigen::Matrix<double, 6, 6>::Identity() * 1e8;
  const Eigen::Matrix<double, 6, 6> sigma = Eigen::Matrix<double, 6, 6>::Identity() * 1e-2;
  for (int k = 0; k < n; ++k) {
    auto p = std::make_shared<SE3Pose>("P" + std::to_string(k), perturb(rng, truth[k], 0.1), weak);
    graph.addVariableNode(p);
    poses.push_back(p);
  }
  graph.addFactorNode(std::make_shared<SE3PosePrior>("PRIOR", truth[0], sigma), std::vector<std::string>{"P0"});
  for (int k = 0; k < n; ++k) {
    const int j = (k + 1) % n;
    graph.addFactorNode(std::make_shared<SE3PosePose>("E" + std::to_string(k), truth[k].inverse() * truth[j], sigma),
                        std::vector<std::string>{"P" + std::to_string(k), "P" + std::to_string(j)});
  }
  GbpSolver solver(SolverScheduleType::SYNCHRONOUS);
  solver.parallel_ = false;
  for (int it = 0; it < 300; ++it) {
    solver.performIteration(graph);
  }
  double worst = 0.0;
  for (int k = 0; k < n; ++k) {
    worst = std::max(worst, logMapSE3(truth[k].inverse() * poses[k]->mu_).norm());
  }
  EXPECT_LT(worst, 1e-6);
}
