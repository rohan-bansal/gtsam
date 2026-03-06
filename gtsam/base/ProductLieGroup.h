/* ----------------------------------------------------------------------------

 * GTSAM Copyright 2010, Georgia Tech Research Corporation,
 * Atlanta, Georgia 30332-0415
 * All Rights Reserved
 * Authors: Frank Dellaert, et al. (see THANKS for the full author list)

 * See LICENSE for the license information

 * -------------------------------------------------------------------------- */

/**
 * @file ProductLieGroup.h
 * @date May, 2015
 * @author Frank Dellaert
 * @brief Group product of two Lie Groups
 */

#pragma once

#include <gtsam/base/Lie.h>
#include <gtsam/base/Testable.h>
#include <gtsam/base/Vector.h>

#include <algorithm>
#include <array>
#include <iostream>
#include <initializer_list>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace gtsam {

namespace product_lie_group_detail {

template <int N>
using VectorN = Eigen::Matrix<double, N, 1>;

template <int N>
using MatrixN = Eigen::Matrix<double, N, N>;

template <int N>
VectorN<N> ZeroVector(Eigen::Index dim) {
  if constexpr (N == Eigen::Dynamic) {
    return VectorN<N>::Zero(dim);
  } else {
    (void)dim;
    return VectorN<N>::Zero();
  }
}

template <int N>
MatrixN<N> ZeroMatrix(Eigen::Index dim) {
  if constexpr (N == Eigen::Dynamic) {
    return MatrixN<N>::Zero(dim, dim);
  } else {
    (void)dim;
    return MatrixN<N>::Zero();
  }
}

template <int N>
MatrixN<N> IdentityMatrix(Eigen::Index dim) {
  if constexpr (N == Eigen::Dynamic) {
    return MatrixN<N>::Identity(dim, dim);
  } else {
    (void)dim;
    return MatrixN<N>::Identity();
  }
}

template <int N, typename Derived>
VectorN<N> Slice(const Eigen::MatrixBase<Derived>& value, Eigen::Index offset,
                 Eigen::Index dim) {
  VectorN<N> result;
  if constexpr (N == Eigen::Dynamic) {
    result.resize(dim);
  }
  result = value.segment(offset, dim);
  return result;
}

template <typename Derived, typename Block>
void AssignDiagonalBlock(Eigen::MatrixBase<Derived>* result, Eigen::Index row,
                         const Block& block) {
  result->derived().block(row, row, block.rows(), block.cols()) = block;
}

template <typename Derived, typename Segment>
void AssignSegment(Eigen::MatrixBase<Derived>* result, Eigen::Index offset,
                   const Segment& value) {
  result->derived().segment(offset, value.size()) = value;
}

template <typename G>
using JacobianOf = Eigen::Matrix<double, traits<G>::dimension, traits<G>::dimension>;

template <typename G>
int DimensionOf(const G& value) {
  return traits<G>::GetDimension(value);
}

template <typename G>
int UniformElementDimension(const std::vector<G>& elements,
                            const char* context) {
  if constexpr (traits<G>::dimension != Eigen::Dynamic) {
    (void)elements;
    (void)context;
    return traits<G>::dimension;
  } else {
    if (elements.empty()) return 0;
    const int dim = DimensionOf(elements.front());
    for (const auto& element : elements) {
      if (DimensionOf(element) != dim) {
        throw std::invalid_argument(
            std::string(context) +
            ": all elements must share the same tangent dimension");
      }
    }
    return dim;
  }
}

inline void CheckSize(Eigen::Index actual, Eigen::Index expected,
                      const char* context) {
  if (actual != expected) {
    throw std::invalid_argument(std::string(context) + ": unexpected tangent dimension");
  }
}

inline void CheckMultiple(Eigen::Index total, Eigen::Index divisor,
                          const char* context) {
  if (divisor == 0) {
    CheckSize(total, 0, context);
    return;
  }
  if (total % divisor != 0) {
    throw std::invalid_argument(
        std::string(context) + ": tangent dimension is not divisible by element dimension");
  }
}

}  // namespace product_lie_group_detail

/**
 * @brief Template to construct the product Lie group of two other Lie groups
 * Assumes Lie group structure for G and H
 */
template <typename G, typename H>
class ProductLieGroup : public std::pair<G, H> {
  GTSAM_CONCEPT_ASSERT(IsLieGroup<G>);
  GTSAM_CONCEPT_ASSERT(IsLieGroup<H>);
  GTSAM_CONCEPT_ASSERT(IsTestable<G>);
  GTSAM_CONCEPT_ASSERT(IsTestable<H>);

 public:
  using Base = std::pair<G, H>;
  static constexpr int dimension1 = traits<G>::dimension;
  static constexpr int dimension2 = traits<H>::dimension;
  static constexpr int dimension =
      (dimension1 == Eigen::Dynamic || dimension2 == Eigen::Dynamic)
          ? Eigen::Dynamic
          : dimension1 + dimension2;

  using TangentVector = Eigen::Matrix<double, dimension, 1>;
  using ChartJacobian = OptionalJacobian<dimension, dimension>;
  using Jacobian = Eigen::Matrix<double, dimension, dimension>;
  using Jacobian1 = product_lie_group_detail::JacobianOf<G>;
  using Jacobian2 = product_lie_group_detail::JacobianOf<H>;

  /// @name Standard Constructors
  /// @{

  /// Default constructor yields identity
  ProductLieGroup() : Base(traits<G>::Identity(), traits<H>::Identity()) {}

  /// Construct from two subgroup elements
  ProductLieGroup(const G& g, const H& h) : Base(g, h) {}

  /// Construct from base pair
  ProductLieGroup(const Base& base) : Base(base) {}

  /// @}
  /// @name Group Operations
  /// @{

  using group_flavor = multiplicative_group_tag;

  /// Identity element
  static ProductLieGroup Identity() { return ProductLieGroup(); }

  /// Group multiplication
  ProductLieGroup operator*(const ProductLieGroup& other) const {
    return ProductLieGroup(traits<G>::Compose(this->first, other.first),
                           traits<H>::Compose(this->second, other.second));
  }

  /// Group inverse
  ProductLieGroup inverse() const {
    return ProductLieGroup(traits<G>::Inverse(this->first),
                           traits<H>::Inverse(this->second));
  }

  /// Compose with another element (same as operator*)
  ProductLieGroup compose(const ProductLieGroup& g) const { return (*this) * g; }

  /// Calculate relative transformation
  ProductLieGroup between(const ProductLieGroup& g) const {
    return this->inverse() * g;
  }

  /// @}
  /// @name Manifold Operations
  /// @{

  /// Return manifold dimension
  static constexpr int Dim() { return dimension; }

  /// Return manifold dimension
  int dim() const { return dim1() + dim2(); }

  /// Retract to manifold
  ProductLieGroup retract(const TangentVector& v, ChartJacobian H1 = {},
                          ChartJacobian H2 = {}) const {
    const int d1 = dim1();
    const int d2 = dim2();
    ValidateTangentSize(v, d1 + d2, "ProductLieGroup::retract");

    const auto v1 = product_lie_group_detail::Slice<dimension1>(v, 0, d1);
    const auto v2 = product_lie_group_detail::Slice<dimension2>(v, d1, d2);

    Jacobian1 Dg1, Dg2;
    Jacobian2 Dh1, Dh2;
    G g = traits<G>::Retract(this->first, v1, H1 ? &Dg1 : nullptr,
                             H2 ? &Dg2 : nullptr);
    H h = traits<H>::Retract(this->second, v2, H1 ? &Dh1 : nullptr,
                             H2 ? &Dh2 : nullptr);

    if (H1) {
      *H1 = product_lie_group_detail::ZeroMatrix<dimension>(d1 + d2);
      product_lie_group_detail::AssignDiagonalBlock(&(*H1), 0, Dg1);
      product_lie_group_detail::AssignDiagonalBlock(&(*H1), d1, Dh1);
    }
    if (H2) {
      *H2 = product_lie_group_detail::ZeroMatrix<dimension>(d1 + d2);
      product_lie_group_detail::AssignDiagonalBlock(&(*H2), 0, Dg2);
      product_lie_group_detail::AssignDiagonalBlock(&(*H2), d1, Dh2);
    }
    return ProductLieGroup(g, h);
  }

  /// Local coordinates on manifold
  TangentVector localCoordinates(const ProductLieGroup& g, ChartJacobian H1 = {},
                                 ChartJacobian H2 = {}) const {
    const int d1 = dim1();
    const int d2 = dim2();
    TangentVector v = product_lie_group_detail::ZeroVector<dimension>(d1 + d2);

    Jacobian1 Dg1, Dg2;
    Jacobian2 Dh1, Dh2;
    const auto v1 = traits<G>::Local(this->first, g.first, H1 ? &Dg1 : nullptr,
                                     H2 ? &Dg2 : nullptr);
    const auto v2 = traits<H>::Local(this->second, g.second,
                                     H1 ? &Dh1 : nullptr,
                                     H2 ? &Dh2 : nullptr);

    product_lie_group_detail::AssignSegment(&v, 0, v1);
    product_lie_group_detail::AssignSegment(&v, d1, v2);

    if (H1) {
      *H1 = product_lie_group_detail::ZeroMatrix<dimension>(d1 + d2);
      product_lie_group_detail::AssignDiagonalBlock(&(*H1), 0, Dg1);
      product_lie_group_detail::AssignDiagonalBlock(&(*H1), d1, Dh1);
    }
    if (H2) {
      *H2 = product_lie_group_detail::ZeroMatrix<dimension>(d1 + d2);
      product_lie_group_detail::AssignDiagonalBlock(&(*H2), 0, Dg2);
      product_lie_group_detail::AssignDiagonalBlock(&(*H2), d1, Dh2);
    }
    return v;
  }

  /// @}
  /// @name Lie Group Operations
  /// @{

  /// Compose with Jacobians
  ProductLieGroup compose(const ProductLieGroup& other, ChartJacobian H1,
                          ChartJacobian H2 = {}) const {
    const int d1 = dim1();
    const int d2 = dim2();

    Jacobian1 Dg1, Dg2;
    Jacobian2 Dh1, Dh2;
    G g = traits<G>::Compose(this->first, other.first, H1 ? &Dg1 : nullptr,
                             H2 ? &Dg2 : nullptr);
    H h = traits<H>::Compose(this->second, other.second,
                             H1 ? &Dh1 : nullptr,
                             H2 ? &Dh2 : nullptr);

    if (H1) {
      *H1 = product_lie_group_detail::ZeroMatrix<dimension>(d1 + d2);
      product_lie_group_detail::AssignDiagonalBlock(&(*H1), 0, Dg1);
      product_lie_group_detail::AssignDiagonalBlock(&(*H1), d1, Dh1);
    }
    if (H2) {
      *H2 = product_lie_group_detail::ZeroMatrix<dimension>(d1 + d2);
      product_lie_group_detail::AssignDiagonalBlock(&(*H2), 0, Dg2);
      product_lie_group_detail::AssignDiagonalBlock(&(*H2), d1, Dh2);
    }
    return ProductLieGroup(g, h);
  }

  /// Between with Jacobians
  ProductLieGroup between(const ProductLieGroup& other, ChartJacobian H1,
                          ChartJacobian H2 = {}) const {
    const int d1 = dim1();
    const int d2 = dim2();

    Jacobian1 Dg1, Dg2;
    Jacobian2 Dh1, Dh2;
    G g = traits<G>::Between(this->first, other.first, H1 ? &Dg1 : nullptr,
                             H2 ? &Dg2 : nullptr);
    H h = traits<H>::Between(this->second, other.second,
                             H1 ? &Dh1 : nullptr,
                             H2 ? &Dh2 : nullptr);

    if (H1) {
      *H1 = product_lie_group_detail::ZeroMatrix<dimension>(d1 + d2);
      product_lie_group_detail::AssignDiagonalBlock(&(*H1), 0, Dg1);
      product_lie_group_detail::AssignDiagonalBlock(&(*H1), d1, Dh1);
    }
    if (H2) {
      *H2 = product_lie_group_detail::ZeroMatrix<dimension>(d1 + d2);
      product_lie_group_detail::AssignDiagonalBlock(&(*H2), 0, Dg2);
      product_lie_group_detail::AssignDiagonalBlock(&(*H2), d1, Dh2);
    }
    return ProductLieGroup(g, h);
  }

  /// Inverse with Jacobian
  ProductLieGroup inverse(ChartJacobian D) const {
    const int d1 = dim1();
    const int d2 = dim2();

    Jacobian1 Dg;
    Jacobian2 Dh;
    G g = traits<G>::Inverse(this->first, D ? &Dg : nullptr);
    H h = traits<H>::Inverse(this->second, D ? &Dh : nullptr);

    if (D) {
      *D = product_lie_group_detail::ZeroMatrix<dimension>(d1 + d2);
      product_lie_group_detail::AssignDiagonalBlock(&(*D), 0, Dg);
      product_lie_group_detail::AssignDiagonalBlock(&(*D), d1, Dh);
    }
    return ProductLieGroup(g, h);
  }

  /// Exponential map
  static ProductLieGroup Expmap(const TangentVector& v, ChartJacobian Hv = {}) {
    int d1 = 0, d2 = 0;
    InferExpmapDimensions(v.size(), &d1, &d2);

    const auto v1 = product_lie_group_detail::Slice<dimension1>(v, 0, d1);
    const auto v2 = product_lie_group_detail::Slice<dimension2>(v, d1, d2);

    Jacobian1 Dg;
    Jacobian2 Dh;
    G g = traits<G>::Expmap(v1, Hv ? &Dg : nullptr);
    H h = traits<H>::Expmap(v2, Hv ? &Dh : nullptr);

    if (Hv) {
      *Hv = product_lie_group_detail::ZeroMatrix<dimension>(d1 + d2);
      product_lie_group_detail::AssignDiagonalBlock(&(*Hv), 0, Dg);
      product_lie_group_detail::AssignDiagonalBlock(&(*Hv), d1, Dh);
    }
    return ProductLieGroup(g, h);
  }

  /// Logarithmic map
  static TangentVector Logmap(const ProductLieGroup& p, ChartJacobian Hp = {}) {
    const int d1 = product_lie_group_detail::DimensionOf(p.first);
    const int d2 = product_lie_group_detail::DimensionOf(p.second);
    TangentVector v = product_lie_group_detail::ZeroVector<dimension>(d1 + d2);

    Jacobian1 Dg;
    Jacobian2 Dh;
    const auto v1 = traits<G>::Logmap(p.first, Hp ? &Dg : nullptr);
    const auto v2 = traits<H>::Logmap(p.second, Hp ? &Dh : nullptr);

    product_lie_group_detail::AssignSegment(&v, 0, v1);
    product_lie_group_detail::AssignSegment(&v, d1, v2);

    if (Hp) {
      *Hp = product_lie_group_detail::ZeroMatrix<dimension>(d1 + d2);
      product_lie_group_detail::AssignDiagonalBlock(&(*Hp), 0, Dg);
      product_lie_group_detail::AssignDiagonalBlock(&(*Hp), d1, Dh);
    }
    return v;
  }

  /// Local coordinates (same as Logmap)
  static TangentVector LocalCoordinates(const ProductLieGroup& p,
                                        ChartJacobian Hp = {}) {
    return Logmap(p, Hp);
  }

  /// Right multiplication by exponential map
  ProductLieGroup expmap(const TangentVector& v) const {
    return compose(ProductLieGroup::Expmap(v));
  }

  /// Logarithmic map for relative transformation
  TangentVector logmap(const ProductLieGroup& g) const {
    return ProductLieGroup::Logmap(between(g));
  }

  /// Adjoint map
  Jacobian AdjointMap() const {
    const auto adjG = traits<G>::AdjointMap(this->first);
    const auto adjH = traits<H>::AdjointMap(this->second);
    Jacobian adj =
        product_lie_group_detail::ZeroMatrix<dimension>(adjG.rows() + adjH.rows());
    adj.block(0, 0, adjG.rows(), adjG.cols()) = adjG;
    adj.block(adjG.rows(), adjG.rows(), adjH.rows(), adjH.cols()) = adjH;
    return adj;
  }

  /// @}
  /// @name Testable interface
  /// @{

  void print(const std::string& s = "") const {
    std::cout << s << "ProductLieGroup" << std::endl;
    traits<G>::Print(this->first, "  first");
    traits<H>::Print(this->second, "  second");
  }

  bool equals(const ProductLieGroup& other, double tol = 1e-9) const {
    return traits<G>::Equals(this->first, other.first, tol) &&
           traits<H>::Equals(this->second, other.second, tol);
  }

  /// @}

 private:
  int dim1() const { return product_lie_group_detail::DimensionOf(this->first); }
  int dim2() const { return product_lie_group_detail::DimensionOf(this->second); }

  static void ValidateTangentSize(const TangentVector& v, int expected,
                                  const char* context) {
    if constexpr (dimension == Eigen::Dynamic) {
      product_lie_group_detail::CheckSize(v.size(), expected, context);
    } else {
      (void)v;
      (void)expected;
      (void)context;
    }
  }

  static void InferExpmapDimensions(Eigen::Index total, int* d1, int* d2) {
    if constexpr (dimension1 != Eigen::Dynamic && dimension2 != Eigen::Dynamic) {
      product_lie_group_detail::CheckSize(total, dimension1 + dimension2,
                                          "ProductLieGroup::Expmap");
      *d1 = dimension1;
      *d2 = dimension2;
    } else if constexpr (dimension1 != Eigen::Dynamic) {
      if (total < dimension1) {
        throw std::invalid_argument(
            "ProductLieGroup::Expmap: tangent dimension is smaller than the fixed first factor");
      }
      *d1 = dimension1;
      *d2 = static_cast<int>(total) - dimension1;
    } else if constexpr (dimension2 != Eigen::Dynamic) {
      if (total < dimension2) {
        throw std::invalid_argument(
            "ProductLieGroup::Expmap: tangent dimension is smaller than the fixed second factor");
      }
      *d1 = static_cast<int>(total) - dimension2;
      *d2 = dimension2;
    } else {
      throw std::invalid_argument(
          "ProductLieGroup::Expmap: cannot infer two dynamic factor dimensions from a tangent vector alone");
    }
  }
};

/**
 * @brief Template to construct the N-fold power of a Lie group
 * Represents the group G^N = G x G x ... x G (N times)
 * Assumes Lie group structure for G and N >= 1
 */
template <typename G, int N>
class PowerLieGroup : public std::array<G, static_cast<size_t>(N)> {
  static_assert(N != Eigen::Dynamic, "Dynamic power group uses a specialization");
  static_assert(N >= 1, "PowerLieGroup requires N >= 1");
  GTSAM_CONCEPT_ASSERT(IsLieGroup<G>);
  GTSAM_CONCEPT_ASSERT(IsTestable<G>);

 public:
  using Base = std::array<G, static_cast<size_t>(N)>;
  static constexpr int baseDimension = traits<G>::dimension;
  static constexpr int dimension =
      (baseDimension == Eigen::Dynamic) ? Eigen::Dynamic : N * baseDimension;

  using TangentVector = Eigen::Matrix<double, dimension, 1>;
  using ChartJacobian = OptionalJacobian<dimension, dimension>;
  using Jacobian = Eigen::Matrix<double, dimension, dimension>;
  using BaseJacobian = product_lie_group_detail::JacobianOf<G>;

  /// @name Standard Constructors
  /// @{

  /// Default constructor yields identity
  PowerLieGroup() { this->fill(traits<G>::Identity()); }

  /// Construct from array of group elements
  PowerLieGroup(const Base& elements) : Base(elements) { ValidateElementDimensions(); }

  /// Construct from initializer list
  PowerLieGroup(const std::initializer_list<G>& elements) {
    if (elements.size() != static_cast<size_t>(N)) {
      throw std::invalid_argument(
          "PowerLieGroup: initializer list size must equal N");
    }
    std::copy(elements.begin(), elements.end(), this->begin());
    ValidateElementDimensions();
  }

  /// @}
  /// @name Group Operations
  /// @{

  using group_flavor = multiplicative_group_tag;

  /// Identity element
  static PowerLieGroup Identity() { return PowerLieGroup(); }

  /// Group multiplication
  PowerLieGroup operator*(const PowerLieGroup& other) const {
    PowerLieGroup result;
    for (int i = 0; i < N; ++i) {
      result[static_cast<size_t>(i)] = traits<G>::Compose(
          (*this)[static_cast<size_t>(i)], other[static_cast<size_t>(i)]);
    }
    return result;
  }

  /// Group inverse
  PowerLieGroup inverse() const {
    PowerLieGroup result;
    for (int i = 0; i < N; ++i) {
      result[static_cast<size_t>(i)] =
          traits<G>::Inverse((*this)[static_cast<size_t>(i)]);
    }
    return result;
  }

  /// Compose with another element (same as operator*)
  PowerLieGroup compose(const PowerLieGroup& g) const { return (*this) * g; }

  /// Calculate relative transformation
  PowerLieGroup between(const PowerLieGroup& g) const {
    return this->inverse() * g;
  }

  /// @}
  /// @name Manifold Operations
  /// @{

  /// Return manifold dimension
  static constexpr int Dim() { return dimension; }

  /// Return manifold dimension
  int dim() const { return N * elementDim(); }

  /// Retract to manifold
  PowerLieGroup retract(const TangentVector& v, ChartJacobian H1 = {},
                        ChartJacobian H2 = {}) const {
    const int d = elementDim();
    ValidateTangentSize(v, dim(), "PowerLieGroup::retract");

    PowerLieGroup result;
    Jacobian H1out, H2out;
    if (H1) H1out = product_lie_group_detail::ZeroMatrix<dimension>(dim());
    if (H2) H2out = product_lie_group_detail::ZeroMatrix<dimension>(dim());

    for (int i = 0; i < N; ++i) {
      const int row = i * d;
      const auto vi = product_lie_group_detail::Slice<baseDimension>(v, row, d);
      BaseJacobian Di1, Di2;
      result[static_cast<size_t>(i)] =
          traits<G>::Retract((*this)[static_cast<size_t>(i)], vi,
                             H1 ? &Di1 : nullptr, H2 ? &Di2 : nullptr);
      if (H1) product_lie_group_detail::AssignDiagonalBlock(&H1out, row, Di1);
      if (H2) product_lie_group_detail::AssignDiagonalBlock(&H2out, row, Di2);
    }

    if (H1) *H1 = H1out;
    if (H2) *H2 = H2out;
    return result;
  }

  /// Local coordinates on manifold
  TangentVector localCoordinates(const PowerLieGroup& g, ChartJacobian H1 = {},
                                 ChartJacobian H2 = {}) const {
    const int d = elementDim();
    TangentVector v = product_lie_group_detail::ZeroVector<dimension>(dim());
    Jacobian H1out, H2out;
    if (H1) H1out = product_lie_group_detail::ZeroMatrix<dimension>(dim());
    if (H2) H2out = product_lie_group_detail::ZeroMatrix<dimension>(dim());

    for (int i = 0; i < N; ++i) {
      const int row = i * d;
      BaseJacobian Di1, Di2;
      const auto vi = traits<G>::Local((*this)[static_cast<size_t>(i)],
                                       g[static_cast<size_t>(i)],
                                       H1 ? &Di1 : nullptr,
                                       H2 ? &Di2 : nullptr);
      product_lie_group_detail::AssignSegment(&v, row, vi);
      if (H1) product_lie_group_detail::AssignDiagonalBlock(&H1out, row, Di1);
      if (H2) product_lie_group_detail::AssignDiagonalBlock(&H2out, row, Di2);
    }

    if (H1) *H1 = H1out;
    if (H2) *H2 = H2out;
    return v;
  }

  /// @}
  /// @name Lie Group Operations
  /// @{

  /// Compose with Jacobians
  PowerLieGroup compose(const PowerLieGroup& other, ChartJacobian H1,
                        ChartJacobian H2 = {}) const {
    const int d = elementDim();
    PowerLieGroup result;
    Jacobian H1out, H2out;
    if (H1) H1out = product_lie_group_detail::ZeroMatrix<dimension>(dim());
    if (H2) H2out = product_lie_group_detail::ZeroMatrix<dimension>(dim());

    for (int i = 0; i < N; ++i) {
      const int row = i * d;
      BaseJacobian Di1, Di2;
      result[static_cast<size_t>(i)] = traits<G>::Compose(
          (*this)[static_cast<size_t>(i)], other[static_cast<size_t>(i)],
          H1 ? &Di1 : nullptr, H2 ? &Di2 : nullptr);
      if (H1) product_lie_group_detail::AssignDiagonalBlock(&H1out, row, Di1);
      if (H2) product_lie_group_detail::AssignDiagonalBlock(&H2out, row, Di2);
    }

    if (H1) *H1 = H1out;
    if (H2) *H2 = H2out;
    return result;
  }

  /// Between with Jacobians
  PowerLieGroup between(const PowerLieGroup& other, ChartJacobian H1,
                        ChartJacobian H2 = {}) const {
    const int d = elementDim();
    PowerLieGroup result;
    Jacobian H1out, H2out;
    if (H1) H1out = product_lie_group_detail::ZeroMatrix<dimension>(dim());
    if (H2) H2out = product_lie_group_detail::ZeroMatrix<dimension>(dim());

    for (int i = 0; i < N; ++i) {
      const int row = i * d;
      BaseJacobian Di1, Di2;
      result[static_cast<size_t>(i)] = traits<G>::Between(
          (*this)[static_cast<size_t>(i)], other[static_cast<size_t>(i)],
          H1 ? &Di1 : nullptr, H2 ? &Di2 : nullptr);
      if (H1) product_lie_group_detail::AssignDiagonalBlock(&H1out, row, Di1);
      if (H2) product_lie_group_detail::AssignDiagonalBlock(&H2out, row, Di2);
    }

    if (H1) *H1 = H1out;
    if (H2) *H2 = H2out;
    return result;
  }

  /// Inverse with Jacobian
  PowerLieGroup inverse(ChartJacobian D) const {
    const int d = elementDim();
    PowerLieGroup result;
    Jacobian Dout;
    if (D) Dout = product_lie_group_detail::ZeroMatrix<dimension>(dim());

    for (int i = 0; i < N; ++i) {
      const int row = i * d;
      BaseJacobian Di;
      result[static_cast<size_t>(i)] =
          traits<G>::Inverse((*this)[static_cast<size_t>(i)], D ? &Di : nullptr);
      if (D) product_lie_group_detail::AssignDiagonalBlock(&Dout, row, Di);
    }

    if (D) *D = Dout;
    return result;
  }

  /// Exponential map
  static PowerLieGroup Expmap(const TangentVector& v, ChartJacobian Hv = {}) {
    const int d = InferElementDimension(v.size());
    PowerLieGroup result;
    Jacobian Hvout;
    if (Hv) Hvout = product_lie_group_detail::ZeroMatrix<dimension>(N * d);

    for (int i = 0; i < N; ++i) {
      const int row = i * d;
      const auto vi = product_lie_group_detail::Slice<baseDimension>(v, row, d);
      BaseJacobian Di;
      result[static_cast<size_t>(i)] =
          traits<G>::Expmap(vi, Hv ? &Di : nullptr);
      if (Hv) product_lie_group_detail::AssignDiagonalBlock(&Hvout, row, Di);
    }

    if (Hv) *Hv = Hvout;
    return result;
  }

  /// Logarithmic map
  static TangentVector Logmap(const PowerLieGroup& p, ChartJacobian Hp = {}) {
    const int d = p.elementDim();
    TangentVector v = product_lie_group_detail::ZeroVector<dimension>(p.dim());
    Jacobian Hpout;
    if (Hp) Hpout = product_lie_group_detail::ZeroMatrix<dimension>(p.dim());

    for (int i = 0; i < N; ++i) {
      const int row = i * d;
      BaseJacobian Di;
      const auto vi =
          traits<G>::Logmap(p[static_cast<size_t>(i)], Hp ? &Di : nullptr);
      product_lie_group_detail::AssignSegment(&v, row, vi);
      if (Hp) product_lie_group_detail::AssignDiagonalBlock(&Hpout, row, Di);
    }

    if (Hp) *Hp = Hpout;
    return v;
  }

  /// Local coordinates (same as Logmap)
  static TangentVector LocalCoordinates(const PowerLieGroup& p,
                                        ChartJacobian Hp = {}) {
    return Logmap(p, Hp);
  }

  /// Right multiplication by exponential map
  PowerLieGroup expmap(const TangentVector& v) const {
    return compose(PowerLieGroup::Expmap(v));
  }

  /// Logarithmic map for relative transformation
  TangentVector logmap(const PowerLieGroup& g) const {
    return PowerLieGroup::Logmap(between(g));
  }

  /// Adjoint map
  Jacobian AdjointMap() const {
    const int total = dim();
    const int d = elementDim();
    Jacobian adj = product_lie_group_detail::ZeroMatrix<dimension>(total);
    for (int i = 0; i < N; ++i) {
      const int row = i * d;
      const auto adjGi = traits<G>::AdjointMap((*this)[static_cast<size_t>(i)]);
      adj.block(row, row, adjGi.rows(), adjGi.cols()) = adjGi;
    }
    return adj;
  }

  /// @}
  /// @name Testable interface
  /// @{

  void print(const std::string& s = "") const {
    std::cout << s << "PowerLieGroup" << std::endl;
    for (int i = 0; i < N; ++i) {
      traits<G>::Print((*this)[static_cast<size_t>(i)],
                       "  component[" + std::to_string(i) + "]");
    }
  }

  bool equals(const PowerLieGroup& other, double tol = 1e-9) const {
    for (int i = 0; i < N; ++i) {
      if (!traits<G>::Equals((*this)[static_cast<size_t>(i)],
                             other[static_cast<size_t>(i)], tol)) {
        return false;
      }
    }
    return true;
  }

  /// @}

 private:
  int elementDim() const {
    if constexpr (baseDimension != Eigen::Dynamic) {
      return baseDimension;
    } else {
      std::vector<G> elements(this->begin(), this->end());
      return product_lie_group_detail::UniformElementDimension(
          elements, "PowerLieGroup");
    }
  }

  void ValidateElementDimensions() const {
    (void)elementDim();
  }

  static int InferElementDimension(Eigen::Index total) {
    if constexpr (baseDimension != Eigen::Dynamic) {
      product_lie_group_detail::CheckSize(total, N * baseDimension,
                                          "PowerLieGroup::Expmap");
      return baseDimension;
    } else {
      product_lie_group_detail::CheckMultiple(total, N,
                                              "PowerLieGroup::Expmap");
      return N == 0 ? 0 : static_cast<int>(total / N);
    }
  }

  static void ValidateTangentSize(const TangentVector& v, int expected,
                                  const char* context) {
    if constexpr (dimension == Eigen::Dynamic) {
      product_lie_group_detail::CheckSize(v.size(), expected, context);
    } else {
      (void)v;
      (void)expected;
      (void)context;
    }
  }
};

/**
 * @brief Dynamic-count power Lie group specialization
 * Represents the group G^n for runtime n.
 */
template <typename G>
class PowerLieGroup<G, Eigen::Dynamic> {
  GTSAM_CONCEPT_ASSERT(IsLieGroup<G>);
  GTSAM_CONCEPT_ASSERT(IsTestable<G>);

 public:
  using Base = std::vector<G>;
  static constexpr int baseDimension = traits<G>::dimension;
  static constexpr int dimension = Eigen::Dynamic;

  using TangentVector = Vector;
  using ChartJacobian = OptionalJacobian<Eigen::Dynamic, Eigen::Dynamic>;
  using Jacobian = Matrix;
  using BaseJacobian = product_lie_group_detail::JacobianOf<G>;

  /// @name Standard Constructors
  /// @{

  /// Default constructor yields the empty identity
  PowerLieGroup() = default;

  /// Construct identity with n elements
  explicit PowerLieGroup(size_t n) : elements_(n, traits<G>::Identity()) {}

  /// Construct from runtime container
  explicit PowerLieGroup(const Base& elements) : elements_(elements) {
    ValidateElementDimensions();
  }

  /// Construct from initializer list
  PowerLieGroup(const std::initializer_list<G>& elements) : elements_(elements) {
    ValidateElementDimensions();
  }

  /// @}
  /// @name Access
  /// @{

  size_t size() const { return elements_.size(); }
  size_t n() const { return elements_.size(); }

  const G& operator[](size_t i) const { return elements_.at(i); }
  G& operator[](size_t i) { return elements_.at(i); }

  typename Base::const_iterator begin() const { return elements_.begin(); }
  typename Base::const_iterator end() const { return elements_.end(); }
  typename Base::iterator begin() { return elements_.begin(); }
  typename Base::iterator end() { return elements_.end(); }

  /// @}
  /// @name Group Operations
  /// @{

  using group_flavor = multiplicative_group_tag;

  /// Identity element
  static PowerLieGroup Identity(size_t n = 0) { return PowerLieGroup(n); }

  /// Group multiplication
  PowerLieGroup operator*(const PowerLieGroup& other) const {
    CheckSameSize(other, "PowerLieGroup::operator*");
    Base result(size());
    for (size_t i = 0; i < size(); ++i) {
      result[i] = traits<G>::Compose(elements_[i], other.elements_[i]);
    }
    return PowerLieGroup(result);
  }

  /// Group inverse
  PowerLieGroup inverse() const {
    Base result(size());
    for (size_t i = 0; i < size(); ++i) {
      result[i] = traits<G>::Inverse(elements_[i]);
    }
    return PowerLieGroup(result);
  }

  /// Compose with another element (same as operator*)
  PowerLieGroup compose(const PowerLieGroup& g) const { return (*this) * g; }

  /// Calculate relative transformation
  PowerLieGroup between(const PowerLieGroup& g) const {
    return this->inverse() * g;
  }

  /// @}
  /// @name Manifold Operations
  /// @{

  /// Return manifold dimension
  static constexpr int Dim() { return dimension; }

  /// Return manifold dimension
  int dim() const { return static_cast<int>(size()) * elementDim(); }

  /// Retract to manifold
  PowerLieGroup retract(const TangentVector& v, ChartJacobian H1 = {},
                        ChartJacobian H2 = {}) const {
    const int d = elementDim();
    product_lie_group_detail::CheckSize(v.size(), dim(),
                                        "PowerLieGroup::retract");

    Base result(size());
    Jacobian H1out = Matrix::Zero(dim(), dim());
    Jacobian H2out = Matrix::Zero(dim(), dim());

    for (size_t i = 0; i < size(); ++i) {
      const int row = static_cast<int>(i) * d;
      const auto vi = product_lie_group_detail::Slice<baseDimension>(v, row, d);
      BaseJacobian Di1, Di2;
      result[i] = traits<G>::Retract(elements_[i], vi, H1 ? &Di1 : nullptr,
                                     H2 ? &Di2 : nullptr);
      if (H1) H1out.block(row, row, Di1.rows(), Di1.cols()) = Di1;
      if (H2) H2out.block(row, row, Di2.rows(), Di2.cols()) = Di2;
    }

    if (H1) *H1 = H1out;
    if (H2) *H2 = H2out;
    return PowerLieGroup(result);
  }

  /// Local coordinates on manifold
  TangentVector localCoordinates(const PowerLieGroup& g, ChartJacobian H1 = {},
                                 ChartJacobian H2 = {}) const {
    CheckSameSize(g, "PowerLieGroup::localCoordinates");
    const int d = elementDim();
    TangentVector v = Vector::Zero(dim());
    Jacobian H1out = Matrix::Zero(dim(), dim());
    Jacobian H2out = Matrix::Zero(dim(), dim());

    for (size_t i = 0; i < size(); ++i) {
      const int row = static_cast<int>(i) * d;
      BaseJacobian Di1, Di2;
      const auto vi = traits<G>::Local(elements_[i], g.elements_[i],
                                       H1 ? &Di1 : nullptr,
                                       H2 ? &Di2 : nullptr);
      v.segment(row, vi.size()) = vi;
      if (H1) H1out.block(row, row, Di1.rows(), Di1.cols()) = Di1;
      if (H2) H2out.block(row, row, Di2.rows(), Di2.cols()) = Di2;
    }

    if (H1) *H1 = H1out;
    if (H2) *H2 = H2out;
    return v;
  }

  /// @}
  /// @name Lie Group Operations
  /// @{

  /// Compose with Jacobians
  PowerLieGroup compose(const PowerLieGroup& other, ChartJacobian H1,
                        ChartJacobian H2 = {}) const {
    CheckSameSize(other, "PowerLieGroup::compose");
    const int d = elementDim();
    Base result(size());
    Jacobian H1out = Matrix::Zero(dim(), dim());
    Jacobian H2out = Matrix::Zero(dim(), dim());

    for (size_t i = 0; i < size(); ++i) {
      const int row = static_cast<int>(i) * d;
      BaseJacobian Di1, Di2;
      result[i] = traits<G>::Compose(elements_[i], other.elements_[i],
                                     H1 ? &Di1 : nullptr,
                                     H2 ? &Di2 : nullptr);
      if (H1) H1out.block(row, row, Di1.rows(), Di1.cols()) = Di1;
      if (H2) H2out.block(row, row, Di2.rows(), Di2.cols()) = Di2;
    }

    if (H1) *H1 = H1out;
    if (H2) *H2 = H2out;
    return PowerLieGroup(result);
  }

  /// Between with Jacobians
  PowerLieGroup between(const PowerLieGroup& other, ChartJacobian H1,
                        ChartJacobian H2 = {}) const {
    CheckSameSize(other, "PowerLieGroup::between");
    const int d = elementDim();
    Base result(size());
    Jacobian H1out = Matrix::Zero(dim(), dim());
    Jacobian H2out = Matrix::Zero(dim(), dim());

    for (size_t i = 0; i < size(); ++i) {
      const int row = static_cast<int>(i) * d;
      BaseJacobian Di1, Di2;
      result[i] = traits<G>::Between(elements_[i], other.elements_[i],
                                     H1 ? &Di1 : nullptr,
                                     H2 ? &Di2 : nullptr);
      if (H1) H1out.block(row, row, Di1.rows(), Di1.cols()) = Di1;
      if (H2) H2out.block(row, row, Di2.rows(), Di2.cols()) = Di2;
    }

    if (H1) *H1 = H1out;
    if (H2) *H2 = H2out;
    return PowerLieGroup(result);
  }

  /// Inverse with Jacobian
  PowerLieGroup inverse(ChartJacobian D) const {
    const int d = elementDim();
    Base result(size());
    Jacobian Dout = Matrix::Zero(dim(), dim());

    for (size_t i = 0; i < size(); ++i) {
      const int row = static_cast<int>(i) * d;
      BaseJacobian Di;
      result[i] = traits<G>::Inverse(elements_[i], D ? &Di : nullptr);
      if (D) Dout.block(row, row, Di.rows(), Di.cols()) = Di;
    }

    if (D) *D = Dout;
    return PowerLieGroup(result);
  }

  /// Exponential map
  static PowerLieGroup Expmap(const TangentVector& v, ChartJacobian Hv = {}) {
    const int d = InferElementDimension(v.size());
    const size_t count = InferCount(v.size(), d);
    Base result(count);
    Jacobian Hvout = Matrix::Zero(v.size(), v.size());

    for (size_t i = 0; i < count; ++i) {
      const int row = static_cast<int>(i) * d;
      const auto vi = product_lie_group_detail::Slice<baseDimension>(v, row, d);
      BaseJacobian Di;
      result[i] = traits<G>::Expmap(vi, Hv ? &Di : nullptr);
      if (Hv) Hvout.block(row, row, Di.rows(), Di.cols()) = Di;
    }

    if (Hv) *Hv = Hvout;
    return PowerLieGroup(result);
  }

  /// Logarithmic map
  static TangentVector Logmap(const PowerLieGroup& p, ChartJacobian Hp = {}) {
    const int d = p.elementDim();
    TangentVector v = Vector::Zero(p.dim());
    Jacobian Hpout = Matrix::Zero(p.dim(), p.dim());

    for (size_t i = 0; i < p.size(); ++i) {
      const int row = static_cast<int>(i) * d;
      BaseJacobian Di;
      const auto vi = traits<G>::Logmap(p.elements_[i], Hp ? &Di : nullptr);
      v.segment(row, vi.size()) = vi;
      if (Hp) Hpout.block(row, row, Di.rows(), Di.cols()) = Di;
    }

    if (Hp) *Hp = Hpout;
    return v;
  }

  /// Local coordinates (same as Logmap)
  static TangentVector LocalCoordinates(const PowerLieGroup& p,
                                        ChartJacobian Hp = {}) {
    return Logmap(p, Hp);
  }

  /// Right multiplication by exponential map
  PowerLieGroup expmap(const TangentVector& v) const {
    return compose(PowerLieGroup::Expmap(v));
  }

  /// Logarithmic map for relative transformation
  TangentVector logmap(const PowerLieGroup& g) const {
    return PowerLieGroup::Logmap(between(g));
  }

  /// Adjoint map
  Jacobian AdjointMap() const {
    const int d = elementDim();
    Jacobian adj = Matrix::Zero(dim(), dim());
    for (size_t i = 0; i < size(); ++i) {
      const int row = static_cast<int>(i) * d;
      const auto adjGi = traits<G>::AdjointMap(elements_[i]);
      adj.block(row, row, adjGi.rows(), adjGi.cols()) = adjGi;
    }
    return adj;
  }

  /// @}
  /// @name Testable interface
  /// @{

  void print(const std::string& s = "") const {
    std::cout << s << "PowerLieGroup" << std::endl;
    for (size_t i = 0; i < size(); ++i) {
      traits<G>::Print(elements_[i], "  component[" + std::to_string(i) + "]");
    }
  }

  bool equals(const PowerLieGroup& other, double tol = 1e-9) const {
    if (size() != other.size()) return false;
    for (size_t i = 0; i < size(); ++i) {
      if (!traits<G>::Equals(elements_[i], other.elements_[i], tol)) {
        return false;
      }
    }
    return true;
  }

  /// @}

 private:
  Base elements_;

  int elementDim() const {
    return product_lie_group_detail::UniformElementDimension(
        elements_, "PowerLieGroup");
  }

  void ValidateElementDimensions() const { (void)elementDim(); }

  void CheckSameSize(const PowerLieGroup& other, const char* context) const {
    if (size() != other.size()) {
      throw std::invalid_argument(std::string(context) + ": incompatible group sizes");
    }
  }

  static int InferElementDimension(Eigen::Index total) {
    if constexpr (baseDimension != Eigen::Dynamic) {
      product_lie_group_detail::CheckMultiple(total, baseDimension,
                                              "PowerLieGroup::Expmap");
      return baseDimension;
    } else {
      if (total == 0) return 0;
      throw std::invalid_argument(
          "PowerLieGroup::Expmap: cannot infer element count for a dynamic-dimensional base group");
    }
  }

  static size_t InferCount(Eigen::Index total, Eigen::Index elementDim) {
    if (elementDim == 0) {
      product_lie_group_detail::CheckSize(total, 0, "PowerLieGroup::Expmap");
      return 0;
    }
    product_lie_group_detail::CheckMultiple(total, elementDim,
                                            "PowerLieGroup::Expmap");
    return static_cast<size_t>(total / elementDim);
  }
};

/// Traits specialization for ProductLieGroup
template <typename G, typename H>
struct traits<ProductLieGroup<G, H>>
    : internal::LieGroup<ProductLieGroup<G, H>> {};

template <typename G, typename H>
struct traits<const ProductLieGroup<G, H>>
    : internal::LieGroup<ProductLieGroup<G, H>> {};

/// Traits specialization for PowerLieGroup
template <typename G, int N>
struct traits<PowerLieGroup<G, N>> : internal::LieGroup<PowerLieGroup<G, N>> {};

template <typename G, int N>
struct traits<const PowerLieGroup<G, N>>
    : internal::LieGroup<PowerLieGroup<G, N>> {};

}  // namespace gtsam
