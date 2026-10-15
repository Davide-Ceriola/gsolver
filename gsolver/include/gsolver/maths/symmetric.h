#pragma once

#include <Eigen/Dense>

namespace gsolver {

  //! Precision added (times identity) to messages, beliefs and the blocks marginalized out,
  //! so they stay invertible when a direction carries no information. It is a tiny amount of
  //! artificial information: code that turns a message into a factor (e.g. folding a dropped
  //! pose into a prior) should subtract it, or it is counted again.
  constexpr double kInformationRegularization = 1e-6;

  //! Replace M by its symmetric part, (M + M^T) / 2. Precision and covariance matrices are
  //! symmetric by definition; this removes the antisymmetric part that rounding leaves behind,
  //! so it cannot accumulate across iterations.
  template <typename Derived>
  inline void symmetrize(Eigen::MatrixBase<Derived>& M) {
    M = (0.5 * (M + M.transpose())).eval(); // eval: M appears on both sides
  }

} // namespace gsolver

namespace gsolver {

  //! Solves A x = b for symmetric positive-definite A with one Cholesky factorization,
  //! reusable for several right-hand sides. If rounding has left A slightly indefinite,
  //! Cholesky fails and it falls back to LDLT.
  template <int N>
  class SpdSolver {
  public:
    using MatrixN = Eigen::Matrix<double, N, N>;

    explicit SpdSolver(const MatrixN& A) : llt_(A) {
      if (llt_.info() != Eigen::Success) {
        ldlt_.compute(A);
        use_ldlt_ = true;
      }
    }

    template <int M>
    Eigen::Matrix<double, N, M> solve(const Eigen::Matrix<double, N, M>& b) const {
      if (use_ldlt_) {
        return ldlt_.solve(b);
      }
      return llt_.solve(b);
    }

    bool usedFallback() const {
      return use_ldlt_;
    }

  private:
    Eigen::LLT<MatrixN> llt_;
    Eigen::LDLT<MatrixN> ldlt_;
    bool use_ldlt_ = false;
  };

} // namespace gsolver
