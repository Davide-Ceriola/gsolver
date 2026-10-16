#include <gsolver/graph/types/factors/se2_landmark.h>

namespace gsolver {

  namespace {
    Isometry2d at(const Eigen::Vector2d& position) {
      Isometry2d T    = Isometry2d::Identity();
      T.translation() = position;
      return T;
    }
  } // namespace

  SE2LandmarkBearing::SE2LandmarkBearing(std::string id, const Eigen::Vector2d& landmark, const double& z, MatrixErrorDim Sigma)
      : FactorNode(id, Sigma), landmark_(landmark), z_(z) {
  }

  void SE2LandmarkBearing::updateErrorAndJacobian() {
    evaluate(edges_.at<0>()->variable_to_factor_message_.mu_, landmark_, z_, error_, J_);
  }

  void SE2LandmarkBearing::evaluate(const Isometry2d& x, const Eigen::Vector2d& landmark, const double& z, VectorErrorDim& error, MatrixJacobianDim& J) {
    // The two-pose bearing with the second pose fixed: keep the Jacobian of the first
    SE2Bearing::VectorErrorDim e;
    SE2Bearing::MatrixJacobianDim J_both;
    SE2Bearing::evaluate(x, at(landmark), z, e, J_both);
    error = e;
    J     = J_both.leftCols<3>();
  }

  SE2LandmarkRangeBearing::SE2LandmarkRangeBearing(std::string id, const Eigen::Vector2d& landmark, const Eigen::Vector2d& z, MatrixErrorDim Sigma)
      : FactorNode(id, Sigma), landmark_(landmark), z_(z) {
  }

  void SE2LandmarkRangeBearing::updateErrorAndJacobian() {
    evaluate(edges_.at<0>()->variable_to_factor_message_.mu_, landmark_, z_, error_, J_);
  }

  void SE2LandmarkRangeBearing::evaluate(const Isometry2d& x,
                                         const Eigen::Vector2d& landmark,
                                         const Eigen::Vector2d& z,
                                         VectorErrorDim& error,
                                         MatrixJacobianDim& J) {
    SE2RangeBearing::VectorErrorDim e;
    SE2RangeBearing::MatrixJacobianDim J_both;
    SE2RangeBearing::evaluate(x, at(landmark), z, e, J_both);
    error = e;
    J     = J_both.leftCols<3>();
  }

} // namespace gsolver
