// Must not compile: an SE3Pose where SE3PointEquality expects an SE3Point.
#include <gsolver/graph/factor_graph.h>
#include <gsolver/graph/types/all_types.h>

using namespace gsolver;

int main() {
  FactorGraph graph;
  auto pose     = std::make_shared<SE3Pose>("P", Eigen::Isometry3d::Identity(), Eigen::Matrix<double, 6, 6>::Identity());
  auto landmark = std::make_shared<SE3Point>("L", Eigen::Vector3d::Zero(), Eigen::Matrix3d::Identity());
  graph.addFactorNode(std::make_shared<SE3PointEquality>("EQ", Eigen::Matrix3d::Identity()), pose, landmark);
}
