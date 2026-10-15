#include <gtest/gtest.h>

#include <functional>
#include <vector>

#include <gsolver/graph/types/all_types.h>

#include "se2_test_utils.h"

using namespace gsolver;
using namespace gsolver::test;

namespace {

  constexpr int kTrials       = 200;
  constexpr double kStep      = 1e-6;
  constexpr double kTolerance = 1e-6;

  using ErrorFunction = std::function<Eigen::VectorXd(const std::vector<Isometry2d>&)>;

  // Central differences of the error under right perturbations x_k <- x_k * exp(tau)
  Eigen::MatrixXd numericJacobian(const ErrorFunction& error_of,
                                  const std::vector<Isometry2d>& xs,
                                  const std::vector<int>& angle_rows) {
    const Eigen::VectorXd e0 = error_of(xs);
    Eigen::MatrixXd J(e0.size(), 3 * xs.size());
    for (size_t k = 0; k < xs.size(); ++k) {
      for (int d = 0; d < 3; ++d) {
        Eigen::Vector3d tau = Eigen::Vector3d::Zero();
        tau[d]              = kStep;
        std::vector<Isometry2d> plus = xs, minus = xs;
        plus[k]               = SE2Pose::boxplus(xs[k], tau);
        minus[k]              = SE2Pose::boxplus(xs[k], -tau);
        Eigen::VectorXd delta = error_of(plus) - error_of(minus);
        for (int row : angle_rows) {
          delta[row] = wrapAngle(delta[row]);
        }
        J.col(3 * k + d) = delta / (2 * kStep);
      }
    }
    return J;
  }

  // Random pair of poses at least min_distance apart, so bearings are well defined
  std::pair<Isometry2d, Isometry2d> separatedPair(std::mt19937& rng, double min_distance = 1.0) {
    while (true) {
      Isometry2d a = randomSE2(rng), b = randomSE2(rng);
      if ((a.translation() - b.translation()).norm() > min_distance) {
        return {a, b};
      }
    }
  }

  double trueBearing(const Isometry2d& xi, const Isometry2d& xj) {
    const Eigen::Vector2d p = xi.linear().transpose() * (xj.translation() - xi.translation());
    return atan2(p.y(), p.x());
  }

} // namespace

// ---------------------------------------------------------------------------
// Jacobians against finite differences
// ---------------------------------------------------------------------------

TEST(SE2Prior, JacobianMatchesFiniteDifferences) {
  std::mt19937 rng(1);
  for (int trial = 0; trial < kTrials; ++trial) {
    const Isometry2d x = randomSE2(rng), z = randomSE2(rng);
    SE2Prior::VectorErrorDim e;
    SE2Prior::MatrixJacobianDim J;
    SE2Prior::evaluate(x, z, e, J);

    auto error_of = [&](const std::vector<Isometry2d>& xs) -> Eigen::VectorXd {
      SE2Prior::VectorErrorDim ee;
      SE2Prior::MatrixJacobianDim JJ;
      SE2Prior::evaluate(xs[0], z, ee, JJ);
      return ee;
    };
    const Eigen::MatrixXd Jn = numericJacobian(error_of, {x}, {2});
    ASSERT_LT((J - Jn).norm(), kTolerance) << "trial " << trial << "\nanalytic\n" << J << "\nnumeric\n" << Jn;
  }
}

TEST(SE2Odometry, JacobianMatchesFiniteDifferences) {
  std::mt19937 rng(2);
  for (int trial = 0; trial < kTrials; ++trial) {
    const Isometry2d xi = randomSE2(rng), xj = randomSE2(rng), z = randomSE2(rng);
    SE2Odometry::VectorErrorDim e;
    SE2Odometry::MatrixJacobianDim J;
    SE2Odometry::evaluate(xi, xj, z, e, J);

    auto error_of = [&](const std::vector<Isometry2d>& xs) -> Eigen::VectorXd {
      SE2Odometry::VectorErrorDim ee;
      SE2Odometry::MatrixJacobianDim JJ;
      SE2Odometry::evaluate(xs[0], xs[1], z, ee, JJ);
      return ee;
    };
    const Eigen::MatrixXd Jn = numericJacobian(error_of, {xi, xj}, {2});
    ASSERT_LT((J - Jn).norm(), kTolerance) << "trial " << trial << "\nanalytic\n" << J << "\nnumeric\n" << Jn;
  }
}

TEST(SE2Bearing, JacobianMatchesFiniteDifferences) {
  std::mt19937 rng(3);
  std::uniform_real_distribution<double> angle(-M_PI, M_PI);
  for (int trial = 0; trial < kTrials; ++trial) {
    const auto [xi, xj] = separatedPair(rng);
    const double z      = angle(rng);
    SE2Bearing::VectorErrorDim e;
    SE2Bearing::MatrixJacobianDim J;
    SE2Bearing::evaluate(xi, xj, z, e, J);

    auto error_of = [&](const std::vector<Isometry2d>& xs) -> Eigen::VectorXd {
      SE2Bearing::VectorErrorDim ee;
      SE2Bearing::MatrixJacobianDim JJ;
      SE2Bearing::evaluate(xs[0], xs[1], z, ee, JJ);
      return ee;
    };
    const Eigen::MatrixXd Jn = numericJacobian(error_of, {xi, xj}, {0});
    ASSERT_LT((J - Jn).norm(), kTolerance) << "trial " << trial << "\nanalytic\n" << J << "\nnumeric\n" << Jn;
  }
}

TEST(SE2RangeBearing, JacobianMatchesFiniteDifferences) {
  std::mt19937 rng(4);
  std::uniform_real_distribution<double> angle(-M_PI, M_PI), range(0.5, 20.0);
  for (int trial = 0; trial < kTrials; ++trial) {
    const auto [xi, xj]     = separatedPair(rng);
    const Eigen::Vector2d z = Eigen::Vector2d(range(rng), angle(rng));
    SE2RangeBearing::VectorErrorDim e;
    SE2RangeBearing::MatrixJacobianDim J;
    SE2RangeBearing::evaluate(xi, xj, z, e, J);

    auto error_of = [&](const std::vector<Isometry2d>& xs) -> Eigen::VectorXd {
      SE2RangeBearing::VectorErrorDim ee;
      SE2RangeBearing::MatrixJacobianDim JJ;
      SE2RangeBearing::evaluate(xs[0], xs[1], z, ee, JJ);
      return ee;
    };
    const Eigen::MatrixXd Jn = numericJacobian(error_of, {xi, xj}, {1});
    ASSERT_LT((J - Jn).norm(), kTolerance) << "trial " << trial << "\nanalytic\n" << J << "\nnumeric\n" << Jn;
  }
}

// ---------------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------------

TEST(SE2Factors, ErrorIsZeroForNoiseFreeMeasurements) {
  std::mt19937 rng(5);
  for (int trial = 0; trial < kTrials; ++trial) {
    const auto [xi, xj] = separatedPair(rng);

    SE2Prior::VectorErrorDim e_prior;
    SE2Prior::MatrixJacobianDim J_prior;
    SE2Prior::evaluate(xi, xi, e_prior, J_prior);
    EXPECT_LT(e_prior.norm(), 1e-12);

    SE2Odometry::VectorErrorDim e_odom;
    SE2Odometry::MatrixJacobianDim J_odom;
    SE2Odometry::evaluate(xi, xj, xi.inverse() * xj, e_odom, J_odom);
    EXPECT_LT(e_odom.norm(), 1e-9);

    SE2Bearing::VectorErrorDim e_bearing;
    SE2Bearing::MatrixJacobianDim J_bearing;
    SE2Bearing::evaluate(xi, xj, trueBearing(xi, xj), e_bearing, J_bearing);
    EXPECT_LT(e_bearing.norm(), 1e-12);

    const double range = (xj.translation() - xi.translation()).norm();
    SE2RangeBearing::VectorErrorDim e_rb;
    SE2RangeBearing::MatrixJacobianDim J_rb;
    SE2RangeBearing::evaluate(xi, xj, Eigen::Vector2d(range, trueBearing(xi, xj)), e_rb, J_rb);
    EXPECT_LT(e_rb.norm(), 1e-9);
  }
}

TEST(SE2Bearing, ErrorWrapsAcrossPi) {
  // Target just below the -x axis: true bearing is -pi + 0.01
  const Isometry2d xi = se2(0, 0, 0);
  const Isometry2d xj = se2(-cos(0.01), -sin(0.01), 0);
  SE2Bearing::VectorErrorDim e;
  SE2Bearing::MatrixJacobianDim J;

  // Measured just above the -x axis: the error is 0.02 rad, not 0.02 - 2 pi
  SE2Bearing::evaluate(xi, xj, M_PI - 0.01, e, J);
  EXPECT_NEAR(e(0), 0.02, 1e-9);

  SE2RangeBearing::VectorErrorDim e_rb;
  SE2RangeBearing::MatrixJacobianDim J_rb;
  SE2RangeBearing::evaluate(xi, xj, Eigen::Vector2d(1.0, M_PI - 0.01), e_rb, J_rb);
  EXPECT_NEAR(e_rb(1), 0.02, 1e-9);
}

TEST(SE2Prior, ErrorWrapsAcrossPi) {
  SE2Prior::VectorErrorDim e;
  SE2Prior::MatrixJacobianDim J;
  // x at pi - 0.01, prior at -pi + 0.01: the error is -0.02 rad, not 2 pi - 0.02
  SE2Prior::evaluate(se2(0, 0, M_PI - 0.01), se2(0, 0, -M_PI + 0.01), e, J);
  EXPECT_NEAR(e(2), -0.02, 1e-9);
}
