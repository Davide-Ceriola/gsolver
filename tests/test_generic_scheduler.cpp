// GenericScheduler ("v4" rule inputs): equals the default tick when everything is sent, its inputs
// do not depend on the units, and it runs on graphs mixing variable types.
#include <gtest/gtest.h>

#include <random>

#include <gsolver/core/generic_scheduler.h>
#include <gsolver/graph/factor_graph.h>
#include <gsolver/graph/types/all_types.h>

#include "se2_test_utils.h"

using namespace gsolver;
using namespace gsolver::test;

namespace {

  // SE(2) loop with a prior, translations (poses, measurements, guesses) scaled by `scale` and the
  // translation parts of every covariance by scale^2: the same problem in other units
  struct ScaledLoop {
    FactorGraph graph;
    std::vector<std::shared_ptr<SE2Pose>> poses;
  };

  Isometry2d scaled(const Isometry2d& T, double s) {
    Isometry2d out    = T;
    out.translation() = T.translation() * s;
    return out;
  }

  Eigen::Matrix3d scaledCov(const Eigen::Matrix3d& C, double s) {
    const Eigen::Matrix3d S = Eigen::Vector3d(s, s, 1.0).asDiagonal();
    return S * C * S;
  }

  void buildScaledLoop(ScaledLoop& l, int n, unsigned seed, double scale) {
    std::mt19937 rng(seed);
    l.graph.initial_guess_is_prior_ = false;
    std::vector<Isometry2d> truth{se2(0, 0, 0)};
    for (int k = 1; k < n; ++k) {
      truth.push_back(truth.back() * se2(1.0, 0.0, 2 * M_PI / n));
    }
    const Eigen::Matrix3d S = Eigen::Vector3d(1e-2, 2e-2, 1e-3).asDiagonal();
    for (int k = 0; k < n; ++k) {
      // Initial covariance 1e-2: every precision stays far above the solver's fixed regularization
      // (kInformationRegularization = 1e-6, an absolute constant, the one thing here that does not
      // scale with the units)
      auto p = std::make_shared<SE2Pose>("P" + std::to_string(k), scaled(perturb(rng, truth[k], 0.2, 0.1), scale),
                                         scaledCov(Eigen::Matrix3d::Identity() * 1e-2, scale));
      l.graph.addVariableNode(p);
      l.poses.push_back(p);
    }
    l.graph.addFactorNode(std::make_shared<SE2Prior>("PR", scaled(truth[0], scale), scaledCov(S, scale)),
                          std::vector<std::string>{"P0"});
    for (int k = 0; k < n; ++k) {
      const int j = (k + 1) % n;
      l.graph.addFactorNode(
        std::make_shared<SE2Odometry>("O" + std::to_string(k),
                                      scaled(perturb(rng, truth[k].inverse() * truth[j], 0.02, 0.01), scale), scaledCov(S, scale)),
        std::vector<std::string>{"P" + std::to_string(k), "P" + std::to_string(j)});
    }
    for (auto& f : l.graph.factor_nodes_) {
      f->linearise_at_belief_ = true;
      f->updateFactorIfNew();
      f->sendAllMessagesIfNew();
    }
    for (auto& v : l.graph.variable_nodes_) {
      v->updateMuAndSigma();
    }
  }

  LogisticHead constantHead(double logit) {
    LogisticHead h;
    h.active    = true;
    h.intercept = logit;
    return h;
  }

  LogisticHead someHead(int n_inputs, unsigned seed, double scale = 1.0) {
    std::mt19937 rng(seed);
    std::normal_distribution<double> nd(0.0, 0.3);
    LogisticHead h;
    h.active = true;
    for (int i = 0; i < n_inputs; ++i) {
      h.features.push_back(i);
    }
    h.coef.resize(n_inputs);
    for (int i = 0; i < n_inputs; ++i) {
      h.coef[i] = nd(rng);
    }
    h.intercept = 0.5;
    h.scale     = scale;
    return h;
  }

} // namespace

TEST(GenericScheduler, EverythingSentEqualsDefaultTick) {
  ScaledLoop a, b, c;
  buildScaledLoop(a, 8, 1, 1.0);
  buildScaledLoop(b, 8, 1, 1.0);
  buildScaledLoop(c, 8, 1, 1.0);
  GenericScheduler sb(b.graph, {}, {}, 0);
  FactorRule always;
  always.reply = constantHead(40.0);
  GenericScheduler sc(c.graph, {}, always, 0);
  for (int it = 0; it < 40; ++it) {
    std::vector<std::vector<int>> masks;
    for (auto& v : a.graph.variable_nodes_) {
      masks.emplace_back(v->numberOfEdges(), 1);
    }
    // reference: the plain incremental tick
    for (size_t i = 0; i < masks.size(); ++i) {
      a.graph.variable_nodes_[i]->sendMessages(masks[i]);
      a.graph.variable_nodes_[i]->markMessagesProcessed();
    }
    for (auto& f : a.graph.factor_nodes_) {
      f->updateFactorIfNew();
      f->sendAllMessagesIfNew();
    }
    for (auto& v : a.graph.variable_nodes_) {
      v->updateMuAndSigma();
    }
    sb.tickMasks(masks);
    sc.tickMasks(masks);
    for (size_t k = 0; k < a.poses.size(); ++k) {
      ASSERT_LT(se2Error(a.poses[k]->mu_, b.poses[k]->mu_).norm(), 1e-12);
      ASSERT_LT(se2Error(a.poses[k]->mu_, c.poses[k]->mu_).norm(), 1e-12);
    }
  }
}

TEST(GenericScheduler, InputsDoNotDependOnTheUnits) {
  // The same problem in units 1 and 10: with the same random draws, every input of both rules and
  // every decision must agree (up to the solver's fixed regularization)
  ScaledLoop a, b;
  buildScaledLoop(a, 8, 2, 1.0);
  buildScaledLoop(b, 8, 2, 10.0);
  VariableRule vr;
  vr.send    = someHead(VariableNodeBase::kRuleInputs, 3);
  vr.damping = someHead(VariableNodeBase::kRuleInputs, 4, 0.9);
  FactorRule fr;
  fr.reply   = someHead(9, 5);
  fr.damping = someHead(9, 6, 0.9);
  GenericScheduler sa(a.graph, vr, fr, 11), sb(b.graph, vr, fr, 11);
  std::mt19937 rng(7);
  std::uniform_real_distribution<double> unit(0.0, 1.0);
  for (int it = 0; it < 30; ++it) {
    const Eigen::MatrixXd Xa = sa.variableFeatures(false), Xb = sb.variableFeatures(false);
    {
      Eigen::Index r, c;
      const double worst = (Xa - Xb).cwiseAbs().maxCoeff(&r, &c);
      ASSERT_LT(worst, 1e-2) << "variable inputs, tick " << it << ", input " << kVariableInputNamesV4[c] << " row " << r
                             << ": " << Xa(r, c) << " vs " << Xb(r, c);
    }
    Eigen::VectorXd u(sa.numberOfEdges());
    for (int r = 0; r < u.size(); ++r) {
      u[r] = unit(rng);
    }
    const TickCounts ca = sa.tick(u), cb = sb.tick(u);
    ASSERT_EQ(ca.variable_to_factor, cb.variable_to_factor) << "tick " << it;
    ASSERT_EQ(ca.factor_to_variable, cb.factor_to_variable) << "tick " << it;
    {
      // 1e-2 in log near the 1e-3 sigma floor = differences below 1e-5 sigma: the solver's fixed
      // regularization moves the two GBPs apart by millionths of a standard deviation
      Eigen::Index r, c;
      const Eigen::MatrixXd D = (sa.lastFactorFeatures() - sb.lastFactorFeatures()).cwiseAbs();
      const double worst      = D.maxCoeff(&r, &c);
      ASSERT_LT(worst, 1e-2) << "factor inputs, tick " << it << ", input " << kFactorInputNamesV4[c] << " row " << r
                             << ": " << sa.lastFactorFeatures()(r, c) << " vs " << sb.lastFactorFeatures()(r, c);
    }
  }
}

TEST(GenericScheduler, MixedVariableTypes) {
  // Two SE(3) poses with velocity and three 3D points, factors between the two types
  FactorGraph g;
  g.initial_guess_is_prior_ = false;
  std::mt19937 rng(9);
  std::normal_distribution<double> nd(0.0, 0.05);
  std::vector<Eigen::Isometry3d> truth{Eigen::Isometry3d::Identity(), Eigen::Isometry3d::Identity()};
  truth[1].translation() = Eigen::Vector3d(1.0, 0.0, 0.0);
  for (int k = 0; k < 2; ++k) {
    SE3PoseVelValue mu;
    mu.pose = truth[k];
    mu.pose.translation() += Eigen::Vector3d(nd(rng), nd(rng), nd(rng));
    mu.vel.setZero();
    g.addVariableNode(std::make_shared<SE3PoseVel>("X" + std::to_string(k), mu, Eigen::Matrix<double, 12, 12>::Identity(), k));
  }
  const std::vector<Eigen::Vector3d> points{{2, 1, 0}, {2, -1, 0.5}, {3, 0, -0.5}};
  for (size_t m = 0; m < points.size(); ++m) {
    g.addVariableNode(std::make_shared<SE3Point>("L" + std::to_string(m),
                                                 points[m] + Eigen::Vector3d(nd(rng), nd(rng), nd(rng)),
                                                 Eigen::Matrix3d::Identity()));
  }
  g.addFactorNode(std::make_shared<SE3PoseVelPrior>("PR", truth[0], Eigen::Matrix<double, 6, 6>::Identity() * 1e-4),
                  std::vector<std::string>{"X0"});
  g.addFactorNode(std::make_shared<SE3PoseVelPoseVel>("OD", truth[0].inverse() * truth[1],
                                                      Eigen::Matrix<double, 6, 6>::Identity() * 1e-3),
                  std::vector<std::string>{"X0", "X1"});
  for (int k = 0; k < 2; ++k) {
    for (size_t m = 0; m < points.size(); ++m) {
      g.addFactorNode(std::make_shared<SE3PoseVelPoint>("M" + std::to_string(k) + std::to_string(m),
                                                        truth[k].inverse() * points[m], Eigen::Matrix3d::Identity() * 1e-3),
                      std::vector<std::string>{"X" + std::to_string(k), "L" + std::to_string(m)});
    }
  }
  for (auto& f : g.factor_nodes_) {
    f->updateFactorIfNew();
    f->sendAllMessagesIfNew();
  }
  for (auto& v : g.variable_nodes_) {
    v->updateMuAndSigma();
  }
  VariableRule vr;
  vr.send = someHead(VariableNodeBase::kRuleInputs, 12);
  FactorRule fr;
  fr.reply   = someHead(9, 13);
  fr.damping = someHead(9, 14, 0.9);
  GenericScheduler s(g, vr, fr, 3);
  size_t sent = 0;
  for (int it = 0; it < 60; ++it) {
    const Eigen::MatrixXd X = s.variableFeatures(false);
    ASSERT_TRUE(X.allFinite()) << "tick " << it;
    sent += s.tick().variable_to_factor;
    ASSERT_TRUE(s.lastFactorFeatures().allFinite()) << "tick " << it;
  }
  EXPECT_GT(sent, 0u);
  auto p = std::dynamic_pointer_cast<SE3Point>(g.variable_nodes_[2]);
  EXPECT_LT((p->mu_ - points[0]).norm(), 0.5);
}

TEST(GenericScheduler, WeightedCountsAndStalenessCap) {
  // Unit weights: the weighted counts are the message counts; zero weights on half the edges: the
  // cost of the others. A rule that never sends, capped at 5 silent ticks: every edge sends at
  // least once every 5 ticks.
  ScaledLoop a, b;
  buildScaledLoop(a, 8, 3, 1.0);
  buildScaledLoop(b, 8, 3, 1.0);
  FactorRule fr;
  fr.reply = someHead(9, 21);
  VariableRule vr;
  vr.send = someHead(VariableNodeBase::kRuleInputs, 22);
  // without a factor rule (every factor with news replies to all its variables) and with one
  ScaledLoop c;
  buildScaledLoop(c, 8, 3, 1.0);
  GenericScheduler sc(c.graph, vr, {}, 5);
  sc.setEdgeWeights(std::vector<double>(sc.numberOfEdges(), 1.0));
  for (int it = 0; it < 20; ++it) {
    const TickCounts cc = sc.tick();
    ASSERT_DOUBLE_EQ(sc.weightedCounts().second, static_cast<double>(cc.factor_to_variable)) << "tick " << it;
  }
  GenericScheduler sa(a.graph, vr, fr, 5);
  sa.setEdgeWeights(std::vector<double>(sa.numberOfEdges(), 1.0));
  ASSERT_EQ(sa.variableFeatures(false).cols(), static_cast<long>(kVariableInputNamesV4.size()));
  for (int it = 0; it < 20; ++it) {
    const TickCounts c = sa.tick();
    const auto w       = sa.weightedCounts();
    ASSERT_DOUBLE_EQ(w.first, static_cast<double>(c.variable_to_factor)) << "tick " << it;
    ASSERT_DOUBLE_EQ(w.second, static_cast<double>(c.factor_to_variable)) << "tick " << it;
  }
  EXPECT_THROW(sa.setEdgeWeights(std::vector<double>(3, 1.0)), std::invalid_argument);

  VariableRule silent;
  silent.send        = constantHead(-40.0);
  silent.max_silence = 5;
  GenericScheduler sb(b.graph, silent, {}, 6);
  std::vector<double> w(sb.numberOfEdges(), 0.0);
  for (size_t r = 0; r < w.size(); r += 2) {
    w[r] = 1.0;
  }
  sb.setEdgeWeights(w);
  {
    const Eigen::MatrixXd X = sb.variableFeatures(false);
    for (size_t r = 0; r < w.size(); ++r) {
      ASSERT_EQ(X(r, X.cols() - 1), w[r]) << "edge cost input, row " << r;
    }
  }
  for (int it = 0; it < 30; ++it) {
    const TickCounts c = sb.tick();
    // the never-sending rule sends only through the cap: every edge after 5 silent ticks
    // (ticks 0-4 silent, send at 5; 6-10 silent, send at 11; ...)
    const size_t expected = it >= 5 && (it - 5) % 6 == 0 ? sb.numberOfEdges() : 0;
    ASSERT_EQ(c.variable_to_factor, expected) << "tick " << it;
    ASSERT_DOUBLE_EQ(sb.weightedCounts().first, expected ? std::ceil(w.size() / 2.0) : 0.0) << "tick " << it;
  }
}
