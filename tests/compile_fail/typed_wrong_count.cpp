// Must not compile: SE3PointEquality needs two variables.
#include <gsolver/graph/factor_graph.h>
#include <gsolver/graph/types/all_types.h>

using namespace gsolver;

int main() {
  FactorGraph graph;
  auto landmark = std::make_shared<SE3Point>("L", Eigen::Vector3d::Zero(), Eigen::Matrix3d::Identity());
  graph.addFactorNode(std::make_shared<SE3PointEquality>("EQ", Eigen::Matrix3d::Identity()), landmark);
}
