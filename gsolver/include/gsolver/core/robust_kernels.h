#pragma once

#include <cmath>

namespace gsolver {

  enum class RobustKernel { NONE, HUBER };

  // Robust kernel of a factor. The threshold is a Mahalanobis distance sqrt(e^T Sigma^-1 e),
  // i.e. in standard deviations of the factor's own noise model.
  struct RobustKernelConfig {
    RobustKernel kernel = RobustKernel::NONE;
    double threshold    = 2.0;
  };

  //! Huber weight for a factor at Mahalanobis distance m. Scaling the factor's information
  //! (eta and Lambda) by it makes the factor's energy w * m^2 equal the Huber energy:
  //! m^2 within the threshold n, 2 n m - n^2 beyond it. Continuous at m = n (w = 1).
  inline double huberWeight(double mahalanobis, double threshold) {
    if (mahalanobis <= threshold) {
      return 1.0;
    }
    return 2.0 * threshold / mahalanobis - (threshold * threshold) / (mahalanobis * mahalanobis);
  }

  inline double robustWeight(const RobustKernelConfig& config, double mahalanobis) {
    switch (config.kernel) {
      case RobustKernel::HUBER:
        return huberWeight(mahalanobis, config.threshold);
      case RobustKernel::NONE:
      default:
        return 1.0;
    }
  }

} // namespace gsolver
