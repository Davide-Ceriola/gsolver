// Symmetric numerics: the SPD solver, and messages/beliefs staying exactly symmetric.
#include <gtest/gtest.h>

#include <random>

#include <gsolver/core/gbp_solver.h>
#include <gsolver/graph/factor_graph.h>
#include <gsolver/graph/types/all_types.h>
#include <gsolver/maths/symmetric.h>

#include "se2_test_utils.h"

using namespace gsolver;
using namespace gsolver::test;

TEST(SpdSolver, MatchesInverseForPositiveDefiniteMatrices) {
  std::mt19937 rng(30);
  std::normal_distribution<double> n(0.0, 1.0);
  for (int trial = 0; trial < 50; ++trial) {
    Eigen::Matrix<double, 6, 6> M;
    for (int i = 0; i < 36; ++i) {
      M(i) = n(rng);
    }
    const Eigen::Matrix<double, 6, 6> A = M * M.transpose() + Eigen::Matrix<double, 6, 6>::Identity() * 0.1;
    const Eigen::Matrix<double, 6, 2> b = Eigen::Matrix<double, 6, 2>::Random();

    const SpdSolver<6> solver(A);
    EXPECT_FALSE(solver.usedFallback());
    EXPECT_TRUE(solver.solve(b).isApprox(A.inverse() * b, 1e-10));
  }
}

TEST(SpdSolver, FallsBackWhenNotPositiveDefinite) {
  // Symmetric but indefinite, as rounding can leave a nearly singular precision
  Eigen::Matrix3d A;
  A << 2, 1, 0,  //
    1, -1e-9, 0, //
    0, 0, 3;
  const Eigen::Vector3d b(1, 2, 3);

  const SpdSolver<3> solver(A);
  EXPECT_TRUE(solver.usedFallback());
  EXPECT_TRUE((A * solver.solve(b)).isApprox(b, 1e-9));
}

TEST(SymmetricNumerics, MessagesAndBeliefsStayExactlySymmetric) {
  // A loopy graph: odometry chain plus bearings between poses three steps apart
  FactorGraph graph;
  graph.initial_guess_is_prior_ = false;
  std::vector<std::shared_ptr<SE2Pose>> poses;
  Isometry2d x = se2(0, 0, 0);
  for (int k = 0; k < 20; ++k) {
    poses.push_back(std::make_shared<SE2Pose>("P" + std::to_string(k), x, Eigen::Matrix3d::Identity() * 1e8));
    graph.addVariableNode(poses.back());
    x = x * se2(1.0, 0.1, 0.2);
  }
  graph.addFactorNode(std::make_shared<SE2Prior>("PRIOR", poses[0]->mu_, Eigen::Matrix3d::Identity() * 1e-2), poses[0]);
  for (int k = 0; k + 1 < 20; ++k) {
    graph.addFactorNode(std::make_shared<SE2Odometry>("O" + std::to_string(k),
                                                      poses[k]->mu_.inverse() * poses[k + 1]->mu_,
                                                      Eigen::Vector3d(1e-2, 1e-2, 1e-3).asDiagonal()),
                        poses[k],
                        poses[k + 1]);
    if (k + 3 < 20) {
      const Eigen::Vector2d p = poses[k]->mu_.linear().transpose() * (poses[k + 3]->mu_.translation() - poses[k]->mu_.translation());
      graph.addFactorNode(
        std::make_shared<SE2Bearing>("B" + std::to_string(k), atan2(p.y(), p.x()), SE2Bearing::MatrixErrorDim::Constant(1e-4)),
        poses[k],
        poses[k + 3]);
    }
  }

  GbpSolver solver(SolverScheduleType::SYNCHRONOUS);
  for (int i = 0; i < 50; ++i) {
    solver.performIteration(graph);
  }
  for (const auto& pose : poses) {
    EXPECT_EQ(pose->Sigma_, pose->Sigma_.transpose()) << pose->id_;
    for (const auto& edge : pose->edges_) {
      const auto& f2v = edge->factor_to_variable_message_.Lambda_;
      const auto& v2f = edge->variable_to_factor_message_.Lambda_;
      EXPECT_EQ(f2v, f2v.transpose()) << edge->factor_id_ << " -> " << pose->id_;
      EXPECT_EQ(v2f, v2f.transpose()) << pose->id_ << " -> " << edge->factor_id_;
    }
  }
}
