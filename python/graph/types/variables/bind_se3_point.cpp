// =============================================================================
// gsolver Python bindings - SE3Point variable
// =============================================================================

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/string.h>

#include <nanobind/eigen/dense.h>

#include <gsolver/graph/types/variables/se3_point.h>

#include "../../bind_accessors.h"

namespace nb = nanobind;

namespace gsolver {
  namespace python {

    void bind_se3_point(nb::module_& m) {
      auto cls = nb::class_<SE3Point, VariableNodeBase>(m, "SE3Point", "3D point variable node")
        .def(
          "__init__",
          [](SE3Point* self,
             const std::string& id,
             nb::ndarray<double, nb::shape<3>> mu,
             nb::ndarray<double, nb::shape<3, 3>> sigma) {
            Eigen::Vector3d mu_eigen;
            Eigen::Matrix3d sigma_eigen;
            for (int i = 0; i < 3; ++i) {
              mu_eigen(i) = mu(i);
              for (int j = 0; j < 3; ++j) {
                sigma_eigen(i, j) = sigma(i, j);
              }
            }
            new (self) SE3Point(id, mu_eigen, sigma_eigen);
          },
          nb::arg("id"),
          nb::arg("mu"),
          nb::arg("sigma"),
          "Create SE3Point variable\n\n"
          "Args:\n"
          "    id: Unique identifier\n"
          "    mu: Initial point estimate (Vector3d)\n"
          "    sigma: 3x3 covariance matrix")
        .def_prop_ro(
          "id", [](const SE3Point& self) { return std::string(self.id_); }, "Variable node ID")
        .def_rw("mu", &SE3Point::mu_, "Current point estimate")
        .def_rw("sigma", &SE3Point::Sigma_, "Covariance matrix")
        .def(
          "incoming_message",
          [](const SE3Point& self, size_t j) {
            const auto& msg = self.edges_.at(j)->factor_to_variable_message_;
            return std::make_pair(Eigen::Vector3d(static_cast<Eigen::Vector3d>(msg.mu_)), Eigen::Matrix3d(msg.Lambda_));
          },
          nb::arg("edge_index"), "(mean, precision) of the factor-to-variable message on edge j");
      add_variable_accessors<SE3Point>(cls);
    }

  } // namespace python
} // namespace gsolver
