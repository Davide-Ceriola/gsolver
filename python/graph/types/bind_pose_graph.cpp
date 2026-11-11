// =============================================================================
// gsolver Python bindings - pose-graph types: SE2Pose, SE2Prior, SE2Odometry,
// SE3PosePose, SE3PosePrior (SE3Pose is in bind_se3_pose.cpp)
// =============================================================================

#include <nanobind/nanobind.h>
#include <nanobind/eigen/dense.h>
#include <nanobind/stl/string.h>

#include <gsolver/graph/types/factors/se2_odometry.h>
#include <gsolver/graph/types/factors/se2_prior.h>
#include <gsolver/graph/types/factors/se3_pose_pose.h>
#include <gsolver/graph/types/variables/se2_pose.h>

#include "../bind_accessors.h"

namespace nb = nanobind;

namespace gsolver {
  namespace python {

    namespace {
      Eigen::Isometry2d iso2(const Eigen::Matrix3d& m) {
        Eigen::Isometry2d iso;
        iso.matrix() = m;
        return iso;
      }
      Eigen::Isometry3d iso3(const Eigen::Matrix4d& m) {
        Eigen::Isometry3d iso;
        iso.matrix() = m;
        return iso;
      }
    } // namespace

    void bind_pose_graph(nb::module_& m) {
      auto se2_pose = nb::class_<SE2Pose, VariableNodeBase>(m, "SE2Pose", "SE2 pose variable node")
        .def(
          "__init__",
          [](SE2Pose* self, const std::string& id, const Eigen::Matrix3d& mu, const Eigen::Matrix3d& sigma) {
            new (self) SE2Pose(id, iso2(mu), sigma);
          },
          nb::arg("id"),
          nb::arg("mu"),
          nb::arg("sigma"),
          "Create SE2Pose variable (mu: 3x3 homogeneous transform; sigma: 3x3 over x, y, theta)")
        .def_prop_rw(
          "mu",
          [](const SE2Pose& v) -> Eigen::Matrix3d { return v.mu_.matrix(); },
          [](SE2Pose& v, const Eigen::Matrix3d& mu) { v.mu_ = iso2(mu); },
          "Current pose estimate as 3x3 homogeneous transform")
        .def_rw("sigma", &SE2Pose::Sigma_, "Covariance (x, y, theta)");
      add_variable_accessors<SE2Pose>(se2_pose);

      auto se2_prior = nb::class_<SE2Prior, FactorNodeBase>(m, "SE2Prior", "Prior on an SE2Pose")
        .def(
          "__init__",
          [](SE2Prior* self, const std::string& id, const Eigen::Matrix3d& z, const Eigen::Matrix3d& sigma) {
            new (self) SE2Prior(id, iso2(z), sigma);
          },
          nb::arg("id"),
          nb::arg("z"),
          nb::arg("sigma"));
      add_factor_accessors<SE2Prior>(se2_prior);

      auto se2_odometry =
        nb::class_<SE2Odometry, FactorNodeBase>(m, "SE2Odometry", "Relative pose between two SE2Pose (g2o EDGE_SE2)")
          .def(
            "__init__",
            [](SE2Odometry* self, const std::string& id, const Eigen::Matrix3d& z, const Eigen::Matrix3d& sigma) {
              new (self) SE2Odometry(id, iso2(z), sigma);
            },
            nb::arg("id"),
            nb::arg("z"),
            nb::arg("sigma"),
            "z: measured pose of x_j in the frame of x_i (3x3)");
      add_factor_accessors<SE2Odometry>(se2_odometry);

      auto se3_pose_pose =
        nb::class_<SE3PosePose, FactorNodeBase>(m, "SE3PosePose", "Relative pose between two SE3Pose (g2o EDGE_SE3:QUAT)")
          .def(
            "__init__",
            [](SE3PosePose* self, const std::string& id, const Eigen::Matrix4d& z, const Eigen::Matrix<double, 6, 6>& sigma) {
              new (self) SE3PosePose(id, iso3(z), sigma);
            },
            nb::arg("id"),
            nb::arg("z"),
            nb::arg("sigma"),
            "z: measured pose of x_j in the frame of x_i (4x4); sigma over (translation, rotation)");
      add_factor_accessors<SE3PosePose>(se3_pose_pose);

      auto se3_pose_prior = nb::class_<SE3PosePrior, FactorNodeBase>(m, "SE3PosePrior", "Prior on an SE3Pose")
        .def(
          "__init__",
          [](SE3PosePrior* self, const std::string& id, const Eigen::Matrix4d& z, const Eigen::Matrix<double, 6, 6>& sigma) {
            new (self) SE3PosePrior(id, iso3(z), sigma);
          },
          nb::arg("id"),
          nb::arg("z"),
          nb::arg("sigma"));
      add_factor_accessors<SE3PosePrior>(se3_pose_prior);
    }

  } // namespace python
} // namespace gsolver
