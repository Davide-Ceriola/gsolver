#pragma once

#include <random>

#include <gsolver/maths/geometry2d.h>
#include <gsolver/graph/types/variables/se2_pose.h>

namespace gsolver::test {

  inline Isometry2d se2(double x, double y, double theta) {
    return expMapSE2(Eigen::Vector3d(x, y, theta));
  }

  inline Isometry2d randomSE2(std::mt19937& rng, double extent = 10.0) {
    std::uniform_real_distribution<double> pos(-extent, extent);
    std::uniform_real_distribution<double> angle(-M_PI, M_PI);
    return se2(pos(rng), pos(rng), angle(rng));
  }

  inline Isometry2d perturb(std::mt19937& rng, const Isometry2d& x, double sigma_t, double sigma_theta) {
    std::normal_distribution<double> n(0.0, 1.0);
    return SE2Pose::boxplus(x, Eigen::Vector3d(sigma_t * n(rng), sigma_t * n(rng), sigma_theta * n(rng)));
  }

  // Pose difference as a tangent vector, with the angle wrapped
  inline Eigen::Vector3d se2Error(const Isometry2d& a, const Isometry2d& b) {
    return logMapSE2(a.inverse() * b);
  }

} // namespace gsolver::test
