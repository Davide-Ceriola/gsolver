#pragma once

#include <Eigen/Dense>

#include <gsolver/graph/factor_node.h>
#include <gsolver/graph/types/factors/se2_bearing.h>
#include <gsolver/graph/types/factors/se2_range_bearing.h>
#include <gsolver/graph/types/variables/se2_pose.h>
#include <gsolver/maths/geometry2d.h>

namespace gsolver {

  /**
   * @brief Bearing from an SE2Pose to a landmark at a known, fixed position.
   *
   * Same measurement as SE2Bearing, with the observed position fixed: a unary factor that
   * anchors the pose in the landmarks' frame.
   */
  class SE2LandmarkBearing : public FactorNode<1, SE2Pose> {
  public:
    Eigen::Vector2d landmark_;
    double z_;

    SE2LandmarkBearing(std::string id, const Eigen::Vector2d& landmark, const double& z, MatrixErrorDim Sigma);

    void updateErrorAndJacobian() override;

    static void evaluate(const Isometry2d& x, const Eigen::Vector2d& landmark, const double& z, VectorErrorDim& error, MatrixJacobianDim& J);
  };

  /**
   * @brief Range and bearing from an SE2Pose to a landmark at a known, fixed position.
   */
  class SE2LandmarkRangeBearing : public FactorNode<2, SE2Pose> {
  public:
    Eigen::Vector2d landmark_;
    Eigen::Vector2d z_;

    SE2LandmarkRangeBearing(std::string id, const Eigen::Vector2d& landmark, const Eigen::Vector2d& z, MatrixErrorDim Sigma);

    void updateErrorAndJacobian() override;

    static void evaluate(const Isometry2d& x,
                         const Eigen::Vector2d& landmark,
                         const Eigen::Vector2d& z,
                         VectorErrorDim& error,
                         MatrixJacobianDim& J);
  };

} // namespace gsolver
