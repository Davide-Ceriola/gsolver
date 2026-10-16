#pragma once

#include <cmath>

namespace gsolver {

  // Robust kernels, as weights that scale a factor's information (eta and Lambda).
  //
  // Each weight is w(m) = psi(m) / m for the kernel's influence function psi, with m the
  // factor's Mahalanobis distance sqrt(e^T Sigma^-1 e) and n the threshold, both in standard
  // deviations of the factor's own noise. This is the iteratively reweighted least-squares
  // weight: at GBP's fixed point every factor satisfies the weighted stationarity condition,
  // so the solution is the M-estimator of the chosen kernel.
  //
  // Common thresholds (95% efficiency for Gaussian inliers, one dimension):
  // Huber 1.345, Cauchy 2.385, Tukey 4.685.
  enum class RobustKernel {
    NONE,
    HUBER,         // quadratic within n, linear beyond: outliers keep some influence
    CAUCHY,        // influence decays slowly towards zero
    GEMAN_MCCLURE, // influence decays fast (equivalent to dynamic covariance scaling)
    TUKEY,         // zero influence beyond n
  };

  struct RobustKernelConfig {
    RobustKernel kernel = RobustKernel::NONE;
    double threshold    = 2.0;
  };

  inline double robustWeight(RobustKernel kernel, double m, double n) {
    const double r2 = (m / n) * (m / n);
    switch (kernel) {
      case RobustKernel::HUBER:
        return m <= n ? 1.0 : n / m;
      case RobustKernel::CAUCHY:
        return 1.0 / (1.0 + r2);
      case RobustKernel::GEMAN_MCCLURE:
        return 1.0 / ((1.0 + r2) * (1.0 + r2));
      case RobustKernel::TUKEY:
        return m < n ? (1.0 - r2) * (1.0 - r2) : 0.0;
      case RobustKernel::NONE:
      default:
        return 1.0;
    }
  }

  inline double robustWeight(const RobustKernelConfig& config, double mahalanobis) {
    return robustWeight(config.kernel, mahalanobis, config.threshold);
  }

} // namespace gsolver
