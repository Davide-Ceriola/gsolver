// Incremental message passing (FactorNodeBase/VariableNodeBase *IfNew, processed flags), the
// API an external scheduler uses to send only some variable-to-factor messages per iteration.
// The solver's own loop does not use it, so it must leave that loop unchanged, and with every
// message sent it must reproduce that loop exactly.
#include <gtest/gtest.h>

#include <gsolver/core/gbp_solver.h>
#include <gsolver/graph/factor_graph.h>
#include <gsolver/graph/types/all_types.h>

#include "se2_test_utils.h"

using namespace gsolver;
using namespace gsolver::test;

namespace {

  const Eigen::Matrix3d kWeakInitialSigma = Eigen::Matrix3d::Identity() * 1e8;
  const Eigen::Matrix3d kSigma            = Eigen::Vector3d(1e-2, 1e-2, 1e-3).asDiagonal();

  // A loop: prior on pose 0, odometry 0-1-...-(n-1), and a closure (n-1) -> 0
  struct Loop {
    FactorGraph graph;
    std::vector<std::shared_ptr<SE2Pose>> poses;
  };

  void buildLoop(Loop& l, int n, unsigned seed) {
    std::mt19937 rng(seed);
    std::vector<Isometry2d> truth{se2(0.0, 0.0, 0.0)};
    const Isometry2d step = se2(1.0, 0.0, 2.0 * M_PI / n);
    for (int k = 1; k < n; ++k) {
      truth.push_back(truth.back() * step);
    }
    for (int k = 0; k < n; ++k) {
      auto pose = std::make_shared<SE2Pose>("POSE_" + std::to_string(k), perturb(rng, truth[k], 0.3, 0.1), kWeakInitialSigma);
      l.graph.addVariableNode(pose);
      l.poses.push_back(pose);
    }
    l.graph.addFactorNode(std::make_shared<SE2Prior>("PRIOR_0", truth[0], kSigma), std::vector<std::string>{"POSE_0"});
    for (int k = 0; k < n; ++k) {
      const int j   = (k + 1) % n;
      const auto id = "ODOM_" + std::to_string(k);
      l.graph.addFactorNode(std::make_shared<SE2Odometry>(id, perturb(rng, truth[k].inverse() * truth[j], 0.01, 0.005), kSigma),
                            std::vector<std::string>{"POSE_" + std::to_string(k), "POSE_" + std::to_string(j)});
    }
  }

  // One tick as an external scheduler runs it with every edge selected: variables send on each
  // edge, factors with something new reply, beliefs absorb the replies
  size_t incrementalTick(FactorGraph& g) {
    for (auto& v : g.variable_nodes_) {
      for (size_t i = 0; i < v->numberOfEdges(); ++i) {
        v->sendMessage(i);
      }
      v->markMessagesProcessed();
    }
    size_t replies = 0;
    for (auto& f : g.factor_nodes_) {
      f->updateFactorIfNew();
      replies += f->sendAllMessagesIfNew();
    }
    for (auto& v : g.variable_nodes_) {
      v->updateMuAndSigma();
    }
    return replies;
  }

  double maxDifference(const Loop& a, const Loop& b) {
    double worst = 0.0;
    for (size_t k = 0; k < a.poses.size(); ++k) {
      worst = std::max(worst, se2Error(a.poses[k]->mu_, b.poses[k]->mu_).norm());
      worst = std::max(worst, (a.poses[k]->Sigma_ - b.poses[k]->Sigma_).norm());
    }
    return worst;
  }

} // namespace

TEST(Incremental, NothingNewMeansNothingSent) {
  Loop l;
  buildLoop(l, 6, 1);
  auto& f = l.graph.factor_nodes_[1]; // an odometry factor
  // Initial variable-to-factor messages are unprocessed
  EXPECT_TRUE(f->hasUnprocessedMessages());
  EXPECT_TRUE(f->updateFactorIfNew());
  EXPECT_EQ(f->sendAllMessagesIfNew(), 2u);
  EXPECT_FALSE(f->hasUnprocessedMessages());
  // Nothing new: no update, nothing sent, the replies stay as they were
  auto v          = std::dynamic_pointer_cast<SE2Pose>(l.graph.variable_nodes_[1]);
  const auto before = v->edges_[0]->factor_to_variable_message_;
  EXPECT_FALSE(f->updateFactorIfNew());
  EXPECT_EQ(f->sendAllMessagesIfNew(), 0u);
  EXPECT_TRUE(v->edges_[0]->factor_to_variable_message_.Lambda_.isApprox(before.Lambda_));
}

TEST(Incremental, ASentMessageIsNewOnlyForItsFactor) {
  Loop l;
  buildLoop(l, 6, 2);
  for (auto& f : l.graph.factor_nodes_) {
    f->markMessagesProcessed();
  }
  auto v = std::dynamic_pointer_cast<SE2Pose>(l.graph.variable_nodes_[2]);
  ASSERT_EQ(v->edges_.size(), 2u);
  v->sendMessage(size_t{0});
  const std::string target = v->edges_[0]->factor_id_;
  for (auto& f : l.graph.factor_nodes_) {
    EXPECT_EQ(f->hasUnprocessedMessages(), f->id_ == target) << f->id_;
  }
}

TEST(Incremental, FactorRepliesAreNewForTheVariable) {
  Loop l;
  buildLoop(l, 6, 3);
  auto v = l.graph.variable_nodes_[3];
  v->markMessagesProcessed();
  EXPECT_FALSE(v->hasUnprocessedMessages());
  for (auto& f : l.graph.factor_nodes_) {
    f->updateFactorIfNew();
    f->sendAllMessagesIfNew();
  }
  EXPECT_TRUE(v->hasUnprocessedMessages());
}

TEST(Incremental, EverythingSentMatchesTheSolverLoop) {
  // Solver loop: factor phase, then variable phase. The incremental tick starts with the
  // variables, so the reference does one factor phase and the first variable phase first.
  Loop a, b;
  buildLoop(a, 8, 4);
  buildLoop(b, 8, 4);
  GbpSolver solver(SolverScheduleType::SYNCHRONOUS);
  solver.parallel_ = false;
  // Same start: both graphs get one factor phase and one belief update
  solver.factorPhase(a.graph);
  for (auto& v : a.graph.variable_nodes_) {
    v->updateMuAndSigma();
  }
  for (auto& f : b.graph.factor_nodes_) {
    f->updateFactorIfNew();
    f->sendAllMessagesIfNew();
  }
  for (auto& v : b.graph.variable_nodes_) {
    v->updateMuAndSigma();
  }
  ASSERT_LT(maxDifference(a, b), 1e-12);
  for (int it = 0; it < 30; ++it) {
    // Reference: variable messages, then factor phase, then beliefs. Per edge, as the
    // incremental tick sends: sendAllMessages computes the same messages as sum-minus-own-edge,
    // which differs by rounding (~1e-8 after one iteration), and loopy GBP amplifies that.
    for (auto& v : a.graph.variable_nodes_) {
      for (size_t i = 0; i < v->numberOfEdges(); ++i) {
        v->sendMessage(i);
      }
    }
    solver.factorPhase(a.graph);
    for (auto& v : a.graph.variable_nodes_) {
      v->updateMuAndSigma();
    }
    const size_t replies = incrementalTick(b.graph);
    EXPECT_EQ(replies, 1u + 2u * 8u); // every factor replies on every edge
    ASSERT_LT(maxDifference(a, b), 1e-12) << "iteration " << it;
  }
}

TEST(Incremental, DroppedRepliesAreNotCounted) {
  Loop l;
  buildLoop(l, 6, 5);
  auto& f = l.graph.factor_nodes_[1];
  f->updateFactorIfNew();
  EXPECT_EQ(f->sendAllMessagesIfNew(), 2u); // first messages are never dropped
  f->message_update_.dropout = 1.0;
  l.graph.variable_nodes_[1]->sendMessage(size_t{0});
  l.graph.variable_nodes_[1]->sendMessage(size_t{1});
  for (auto& g : l.graph.factor_nodes_) {
    if (g->hasUnprocessedMessages() && g != f) {
      g->markMessagesProcessed();
    }
  }
  if (f->hasUnprocessedMessages()) {
    f->updateFactorIfNew();
    EXPECT_EQ(f->sendAllMessagesIfNew(), 0u); // all dropped
    EXPECT_FALSE(f->hasUnprocessedMessages());
  }
}

TEST(BeliefMessages, EqualPerEdgeMessages) {
  // A variable sending its belief, and each factor removing its own last message, must give the
  // same GBP as per-edge variable messages, in both linearisation modes
  for (bool at_belief : {false, true}) {
    Loop a, b;
    buildLoop(a, 8, 6);
    buildLoop(b, 8, 6);
    for (auto* l : {&a, &b}) {
      for (auto& f : l->graph.factor_nodes_) {
        f->linearise_at_belief_ = at_belief;
      }
    }
    for (auto& v : b.graph.variable_nodes_) {
      v->send_belief_ = true;
    }
    std::mt19937 rng(3);
    std::bernoulli_distribution send(0.7);
    for (int it = 0; it < 60; ++it) {
      for (size_t i = 0; i < a.graph.variable_nodes_.size(); ++i) {
        std::vector<int> mask(a.graph.variable_nodes_[i]->numberOfEdges());
        for (auto& m : mask) {
          m = send(rng);
        }
        a.graph.variable_nodes_[i]->sendMessages(mask);
        b.graph.variable_nodes_[i]->sendMessages(mask);
        a.graph.variable_nodes_[i]->markMessagesProcessed();
        b.graph.variable_nodes_[i]->markMessagesProcessed();
      }
      for (auto* l : {&a, &b}) {
        for (auto& f : l->graph.factor_nodes_) {
          f->updateFactorIfNew();
          f->sendAllMessagesIfNew();
        }
        for (auto& v : l->graph.variable_nodes_) {
          v->updateMuAndSigma();
        }
      }
      ASSERT_LT(maxDifference(a, b), 1e-9) << "linearise_at_belief " << at_belief << ", iteration " << it;
    }
  }
}

TEST(EdgeDamping, SameAsFactorDamping) {
  // Damping set on every edge (as a scheduler does) equals the same damping set on the factors
  Loop a, b;
  buildLoop(a, 8, 7);
  buildLoop(b, 8, 7);
  for (auto& f : a.graph.factor_nodes_) {
    f->message_update_.damping = 0.4;
  }
  for (auto& v : b.graph.variable_nodes_) {
    auto p = std::dynamic_pointer_cast<SE2Pose>(v);
    for (auto& e : p->edges_) {
      e->damping_ = 0.4;
    }
  }
  for (int it = 0; it < 40; ++it) {
    incrementalTick(a.graph);
    incrementalTick(b.graph);
    ASSERT_LT(maxDifference(a, b), 1e-12) << "iteration " << it;
  }
}
