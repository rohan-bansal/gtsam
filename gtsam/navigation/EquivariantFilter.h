/**
 * @file EquivariantFilter.h
 * @brief Equivariant Filter (EqF) implementation
 *
 * @author Darshan Rajasekaran
 * @author Jennifer Oum
 * @author Rohan Bansal
 * @author Frank Dellaert
 * @date 2025
 */

#pragma once

#include <gtsam/base/GroupAction.h>
#include <gtsam/base/Lie.h>
#include <gtsam/base/Matrix.h>
#include <gtsam/base/Vector.h>
#include <gtsam/geometry/Rot3.h>
#include <gtsam/geometry/Unit3.h>
#include <gtsam/inference/Symbol.h>
#include <gtsam/navigation/ImuBias.h>
#include <gtsam/navigation/ManifoldEKF.h>
#include <gtsam/nonlinear/Values.h>

#include <iostream>
#include <type_traits>

namespace gtsam {

/**
 * Equivariant Filter (EqF) for state estimation on Lie groups.
 *
 * The EqF estimates a Lie group state X in G and a manifold state xi in M.
 * It uses a symmetry principle where the error dynamics are autonomous in a
 * specific frame.
 *
 * This implementation supports two modes:
 * 1. **Automatic**: The filter calculates Jacobian A using the input orbit.
 * 2. **Explicit**: You provide the Jacobian A and the manifold covariance Qc
 *    directly.
 *
 * @tparam M Manifold type for the physical state.
 * @tparam Symmetry Functor encoding the group action on the state.
 */
template <typename M, typename Symmetry>
class EquivariantFilter : public ManifoldEKF<M> {
 public:
  using Base = ManifoldEKF<M>;

  // Manifold traits
  static constexpr int DimM = Base::Dim;
  using TangentM = typename Base::TangentVector;
  using MatrixM = typename Base::Jacobian;
  using CovarianceM = typename Base::Covariance;

  // Group traits
  using G = typename Symmetry::Group;
  static constexpr int DimG = traits<G>::dimension;
  using TangentG = typename traits<G>::TangentVector;

  // Cross-dimension helpers: use dynamic matrices when either dimension is
  // dynamic
  static constexpr bool IsDynamic =
      (DimM == Eigen::Dynamic) || (DimG == Eigen::Dynamic);
  using MatrixMG =
      std::conditional_t<IsDynamic, Eigen::MatrixXd,
                         Eigen::Matrix<double, DimM, DimG>>;
  using MatrixGM =
      std::conditional_t<IsDynamic, Eigen::MatrixXd,
                         Eigen::Matrix<double, DimG, DimM>>;

 protected:
  M xi_ref_;  // Origin (reference) state on the manifold
  G g_;       // Group element estimate

  MatrixMG Dphi0_;           // Differential of state action at identity
  MatrixGM InnovationLift_;  // Innovation lift matrix ((Dphi0)^+)

  // Helper to detect whether Symmetry provides an Orbit type
  template <typename S, typename = void>
  struct HasOrbit : std::false_type {};
  template <typename S>
  struct HasOrbit<S, std::void_t<typename S::Orbit>> : std::true_type {};

 public:
  /**
   * @brief Initialize the Equivariant Filter.
   *
   * @param xi_ref Reference manifold state (origin of lifted coordinates).
   * @param Sigma Initial covariance on the manifold.
   * @param X0 Initial group estimate (default: Identity).
   */
  EquivariantFilter(const M& xi_ref, const CovarianceM& Sigma,
                    const G& X0 = traits<G>::Identity())
      : Base(xi_ref, Sigma), xi_ref_(xi_ref), g_(X0) {
    if constexpr (HasOrbit<Symmetry>::value) {
      typename Symmetry::Orbit act_on_ref(xi_ref);
      // Compute differential of action phi at identity (Dphi0)
      act_on_ref(traits<G>::Identity(), &Dphi0_);
      // Precompute the Innovation Lift matrix (pseudo-inverse of Dphi0)
      InnovationLift_ =
          Dphi0_.completeOrthogonalDecomposition().pseudoInverse();
      this->X_ = act_on_ref(g_);
    }
  }

 protected:
  /**
   * @brief Protected constructor for subclasses that manage their own group
   *        element and do not use Dphi0_/InnovationLift_.
   *
   * This constructor initializes ManifoldEKF with the given state and
   * covariance, stores xi_ref_ and g_, but skips Dphi0_ computation.
   * Subclasses should directly manipulate g_, xi_ref_, and P_.
   *
   * @param xi_ref Reference manifold state.
   * @param Sigma Initial covariance.
   * @param X0 Initial group estimate.
   * @param subclassTag Disambiguation tag (unused).
   */
  struct SubclassTag {};
  EquivariantFilter(const M& xi_ref, const CovarianceM& Sigma, const G& X0,
                    SubclassTag)
      : Base(xi_ref, Sigma), xi_ref_(xi_ref), g_(X0) {}

 public:
  /// State on the manifold M is given by the base class
  using Base::state;

  /// errorCovariance that returns P_, on the equivariant filter error
  const typename Base::Covariance& errorCovariance() const { return this->P_; }

  /// Covariance in the tangent space at the current state.
  CovarianceM covariance() const {
    if constexpr (HasOrbit<Symmetry>::value) {
      MatrixM J;
      if constexpr (MatrixM::RowsAtCompileTime == Eigen::Dynamic) {
        J.resize(this->n_, this->n_);
      }
      const typename Symmetry::Diffeomorphism action_at_g(g_);
      action_at_g(xi_ref_, &J);
      return J.transpose() * this->P_ * J;
    } else {
      return this->P_;
    }
  }

  /// @return Current group estimate.
  const G& groupEstimate() const { return g_; }

  /// @return Reference state (origin).
  const M& referenceState() const { return xi_ref_; }

  /**
   * @brief Resize internal state for dynamic-dimension manifolds.
   *
   * Updates P_, I_, and n_ to reflect a new tangent space dimension.
   * Only meaningful when DimM == Eigen::Dynamic.
   *
   * @param newDim New tangent space dimension.
   */
  void resizeDynamic(int newDim) {
    static_assert(DimM == Eigen::Dynamic,
                  "resizeDynamic only valid for dynamic-dimension manifolds");
    this->n_ = newDim;
    // Resize P_ preserving existing data
    int oldDim = this->P_.rows();
    if (oldDim != newDim) {
      this->P_.conservativeResize(newDim, newDim);
      if (newDim > oldDim) {
        this->P_.block(oldDim, 0, newDim - oldDim, oldDim).setZero();
        this->P_.block(0, oldDim, oldDim, newDim - oldDim).setZero();
        this->P_.block(oldDim, oldDim, newDim - oldDim, newDim - oldDim)
            .setZero();
      }
    }
    this->I_ = MatrixM::Identity(newDim, newDim);
  }

  /**
   * @brief Compute the error dynamics matrix A (Automatic).
   *
   * Calculates A = D_phi|_0 * D_lift|_u0, where u0 is the input mapped to the
   * origin.
   *
   * @tparam Lift Functor for the lift Lambda(xi, u).
   * @tparam InputOrbit Functor for the input orbit psi_u.
   * @param psi_u Input Orbit instance.
   * @return MatrixM The calculated error dynamics matrix A.
   */
  template <typename Lift, typename InputOrbit>
  MatrixM computeErrorDynamicsMatrix(const InputOrbit& psi_u) const {
    MatrixGM D_lift;
    // Map current input to origin: u_origin = psi_u(X^-1)
    auto u_origin = psi_u(g_.inverse());

    // Lift at origin: D_lift = d(Lambda(xi_ref, u_origin))/dxi
    Lift lift_u_origin(u_origin);
    lift_u_origin(xi_ref_, &D_lift);

    return Dphi0_ * D_lift;
  }

  /**
   * @brief Discretize continuous-time error dynamics over dt.
   *
   * K=1 gives Euler, K>1 calls expm(A*dt, K).
   */
  template <size_t K = 1>
  MatrixM transitionMatrix(const MatrixM& A, double dt) const {
    if constexpr (K == 1) {
      return this->I_ + A * dt;
    } else {
      return MatrixM(expm(A * dt, K));
    }
  }

  /**
   * @brief Propagate the filter state (Automatic).
   */
  template <size_t K = 1, typename Lift, typename InputOrbit>
  void predict(const Lift& lift_u, const InputOrbit& psi_u, const MatrixM& Qc,
               double dt) {
    MatrixM A = computeErrorDynamicsMatrix<Lift>(psi_u);
    predictWithJacobian<K>(lift_u, A, Qc, dt);
  }

  /**
   * @brief Propagate the filter state (Explicit).
   */
  template <size_t K = 1, typename Lift>
  void predictWithJacobian(const Lift& lift_u, const MatrixM& A,
                           const MatrixM& Qc, double dt) {
    M xi_est = this->state();
    TangentG Lambda = lift_u(xi_est);

    g_ = traits<G>::Compose(g_, traits<G>::Expmap(Lambda * dt));

    if constexpr (HasOrbit<Symmetry>::value) {
      typename Symmetry::Orbit act_on_ref(xi_ref_);
      M xi_next = act_on_ref(g_);
      MatrixM Phi = transitionMatrix<K>(A, dt);
      CovarianceM Q_manifold = Qc * dt;
      Base::predict(xi_next, Phi, Q_manifold);
    }
  }

  /**
   * Measurement update: Corrects the state and covariance using a
   * pre-calculated predicted measurement and its Jacobian.
   *
   * Overwrites ManifoldEKF::update to modify g_ as well.
   */
  template <typename Measurement>
  void update(
      const Measurement& prediction,
      const Eigen::Matrix<double, traits<Measurement>::dimension, DimM>& H,
      const Measurement& z,
      const Eigen::Matrix<double, traits<Measurement>::dimension,
                          traits<Measurement>::dimension>& R) {
    static constexpr int MeasDim = traits<Measurement>::dimension;

    typename traits<Measurement>::TangentVector innovation =
        traits<Measurement>::Local(z, prediction);

    Eigen::Matrix<double, DimM, MeasDim> K = this->KalmanGain(H, R);
    TangentM delta_xi = -K * innovation;

    // Lift correction to Group tangent space
    TangentG delta_x = InnovationLift_ * delta_xi;
    g_ = traits<G>::Compose(traits<G>::Expmap(delta_x), g_);

    if constexpr (HasOrbit<Symmetry>::value) {
      typename Symmetry::Orbit act_on_ref(xi_ref_);
      this->X_ = act_on_ref(g_);
    }

    this->JosephUpdate(K, H, R);
  }

  /// Same API as ManifoldEKF for measurement update with model function.
  template <typename Z, typename Func>
  void update(Func&& h, const Z& z,
              const Eigen::Matrix<double, traits<Z>::dimension,
                                  traits<Z>::dimension>& R) {
    static_assert(IsManifold<Z>::value,
                  "Template parameter Z must be a GTSAM Manifold.");

    Matrix H(traits<Z>::GetDimension(z), this->n_);
    Z prediction = h(this->X_, H);
    update<Z>(prediction, H, z, R);
  }

  /// Same API as ManifoldEKF for measurement update with vector inputs.
  void updateWithVector(const gtsam::Vector& prediction, const Matrix& H,
                        const gtsam::Vector& z, const Matrix& R) {
    this->validateInputs(prediction, H, z, R);
    update<Vector>(prediction, H, z, R);
  }
};

}  // namespace gtsam
