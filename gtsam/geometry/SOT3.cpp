/* ----------------------------------------------------------------------------

 * GTSAM Copyright 2010, Georgia Tech Research Corporation,
 * Atlanta, Georgia 30332-0415
 * All Rights Reserved
 * Authors: Frank Dellaert, et al. (see THANKS for the full author list)

 * See LICENSE for the license information

 * -------------------------------------------------------------------------- */

/**
 * @file    SOT3.cpp
 * @brief   The scaled orthogonal transforms SOT(3) = SO(3) x R>0
 * @author  Rohan Bansal
 */

#include <gtsam/geometry/SOT3.h>

#include <cmath>
#include <iostream>
#include <stdexcept>

namespace gtsam {

SOT3::SOT3(const SO3& R, double c) : Base() {
  if (c <= 0.0) {
    throw std::invalid_argument("SOT3: scale must be strictly positive");
  }
  this->first = R;
  this->second = Vector1::Constant(std::log(c));
}

SOT3::SOT3(const MatrixNN& M) : SOT3(SO3(M.topLeftCorner<3, 3>()), M(3, 3)) {}

void SOT3::print(const std::string& s) const {
  if (!s.empty()) std::cout << s << "\n";
  std::cout << "SOT3:\n";
  rotation().print("  R: ");
  std::cout << "  c: " << scalar() << "\n";
}

bool SOT3::equals(const SOT3& other, double tol) const {
  return rotation().equals(other.rotation(), tol) &&
         std::abs(scalar() - other.scalar()) < tol;
}

// LieGroup interface

SOT3::MatrixNN SOT3::matrix() const {
  MatrixNN M = MatrixNN::Zero();
  M.topLeftCorner<3, 3>() = rotation().matrix();
  M(3, 3) = scalar();
  return M;
}

SOT3::MatrixNN SOT3::Hat(const TangentVector& xi) {
  // (Omega, s) -> [[Omega^x, 0], [0, s]]
  MatrixNN X = MatrixNN::Zero();
  X.topLeftCorner<3, 3>() = SO3::Hat(Vector3(xi.head<3>()));
  X(3, 3) = xi(3);
  return X;
}

SOT3::TangentVector SOT3::Vee(const MatrixNN& X) {
  // [[Omega^x, 0], [0, s]] -> (Omega, s)
  TangentVector xi;
  xi.head<3>() = SO3::Vee(Matrix3(X.topLeftCorner<3, 3>()));
  xi(3) = X(3, 3);
  return xi;
}

//******************************************************************************
// Lie Group

SOT3 SOT3::Expmap(const TangentVector& xi, ChartJacobian H) {
  return SOT3(Base::Expmap(xi, H));
}

SOT3::TangentVector SOT3::Logmap(const SOT3& Q, ChartJacobian H) {
  return Base::Logmap(Q, H);
}

SOT3::Jacobian SOT3::AdjointMap() const {
  return Base::AdjointMap();
}

SOT3 SOT3::ChartAtOrigin::Retract(const TangentVector& xi, ChartJacobian H) {
  return SOT3::Expmap(xi, H);
}

SOT3::TangentVector SOT3::ChartAtOrigin::Local(const SOT3& Q,
                                                ChartJacobian H) {
  return SOT3::Logmap(Q, H);
}

}  // namespace gtsam
