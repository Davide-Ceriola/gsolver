// =============================================================================
// gsolver Python bindings - SE3PoseVelStereoPoint factor (stereo camera to point)
// =============================================================================

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/string.h>

#include <nanobind/eigen/dense.h>

#include <gsolver/graph/types/factors/se3_posevel_stereo_point.h>

#include "../../bind_accessors.h"

namespace nb = nanobind;

namespace gsolver {
  namespace python {

    void bind_se3_posevel_stereo_point(nb::module_& m) {
      auto cls = nb::class_<SE3PoseVelStereoPoint, FactorNodeBase>(m, "SE3PoseVelStereoPoint",
                                                                    "Stereo camera factor between SE3PoseVel and SE3Point")
        .def(
          "__init__",
          [](SE3PoseVelStereoPoint* self,
             const std::string& id,
             const Eigen::Vector3d& z,
             const Eigen::Matrix<double, 3, 4>& P,
             double baseline,
             const Eigen::Matrix3d& sigma) { new (self) SE3PoseVelStereoPoint(id, z, P, baseline, sigma); },
          nb::arg("id"),
          nb::arg("z"),
          nb::arg("P"),
          nb::arg("baseline"),
          nb::arg("sigma"),
          "Create stereo factor\n\n"
          "Args:\n"
          "    id: Unique identifier\n"
          "    z: Stereo observation [u_left, v, disparity]\n"
          "    P: 3x4 camera projection matrix [fx 0 cx 0; 0 fy cy 0; 0 0 1 0]\n"
          "    baseline: Stereo baseline (metres)\n"
          "    sigma: 3x3 measurement covariance (pixels^2)")
        .def_ro("z", &SE3PoseVelStereoPoint::z_, "Stereo observation")
        .def_ro("inlier", &SE3PoseVelStereoPoint::inlier_, "False when the point is behind the camera or outside the image");
      add_factor_accessors<SE3PoseVelStereoPoint>(cls);
    }

  } // namespace python
} // namespace gsolver
