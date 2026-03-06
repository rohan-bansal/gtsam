/* ----------------------------------------------------------------------------

 * GTSAM Copyright 2010, Georgia Tech Research Corporation,
 * Atlanta, Georgia 30332-0415
 * All Rights Reserved
 * Authors: Frank Dellaert, et al. (see THANKS for the full author list)

 * See LICENSE for the license information

 * -------------------------------------------------------------------------- */

/**
 * @file    VIOEqFMatrices.cpp
 * @brief   Euclidean EqF matrix suite for VIO foundations
 */

#include <gtsam_unstable/navigation/VIOEqFMatrices.h>

#include <gtsam_unstable/navigation/VIOSymmetry.h>

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace gtsam {

namespace {

Pose3 APose(const VIOGroup& X) { return Pose3(X.A().rotation(), X.A().x(0)); }

Rot3 RotationFromTwoVectors(const Vector3& from, const Vector3& to) {
  const double eps = 1e-12;
  const Vector3 a = from.normalized();
  const Vector3 b = to.normalized();
  const double c = std::clamp(a.dot(b), -1.0, 1.0);

  if (c > 1.0 - eps) {
    return Rot3::Identity();
  }

  if (c < -1.0 + eps) {
    Vector3 axis = a.unitOrthogonal();
    axis.normalize();
    return Rot3::Expmap(std::acos(-1.0) * axis);
  }

  Vector3 axis = a.cross(b);
  const double s = axis.norm();
  axis /= s;
  const double angle = std::atan2(s, c);
  return Rot3::Expmap(angle * axis);
}

Matrix NumericalDifferential(const std::function<Vector(const Vector&)>& f,
                             const Vector& x, double h = 1e-6) {
  const int n = static_cast<int>(x.size());
  const Vector y = f(x);
  const int m = static_cast<int>(y.size());

  Matrix J = Matrix::Zero(m, n);
  for (int j = 0; j < n; ++j) {
    Vector dx = Vector::Zero(n);
    dx(j) = h;
    J.col(j) = (f(x + dx) - f(x - dx)) / (2.0 * h);
  }
  return J;
}

Matrix EqFInputMatrixB_euclid(const VIOGroup& X, const VIOState& xi0);

Matrix EqFStateMatrixA_euclid(const VIOGroup& X, const VIOState& xi0,
                              const IMUVelocity& imuVel) {
  const int N = static_cast<int>(xi0.n());
  Matrix A0t = Matrix::Zero(xi0.dim(), xi0.dim());

  A0t.block(0, 0, xi0.dim(), 6) = -EqFInputMatrixB_euclid(X, xi0).block(
      0, 0, xi0.dim(), 6);
  A0t.block<3, 3>(9, 12).setIdentity();
  A0t.block<3, 3>(12, 6) = -GRAVITY_CONSTANT * Rot3::Hat(xi0.sensor.gravityDir());

  const VIOState xiHat = stateGroupAction(X, xi0);
  const IMUVelocity vEst = imuVel - xiHat.sensor.inputBias;
  Vector6 U_I;
  U_I << vEst.gyr, xiHat.sensor.velocity;

  const Pose3 A = APose(X);
  const Vector6 commonTwist =
      xi0.sensor.cameraOffset.inverse().AdjointMap() * A.AdjointMap() * U_I;
  A0t.block<6, 6>(15, 15) = Pose3::adjointMap(commonTwist);

  const Matrix3 R_IC = xiHat.sensor.cameraOffset.rotation().matrix();
  const Matrix3 R_A = A.rotation().matrix();
  for (int i = 0; i < N; ++i) {
    const Matrix3 Qhat_i = X.Q()[static_cast<size_t>(i)].rotation().matrix() *
                           X.Q()[static_cast<size_t>(i)].scalar();
    A0t.block<3, 3>(VIOSensorState::CompDim + 3 * i, 12) =
        -Qhat_i * R_IC.transpose() * R_A.transpose();
  }

  const Matrix66 commonTerm =
      X.B().inverse().AdjointMap() * Pose3::adjointMap(commonTwist);
  for (int i = 0; i < N; ++i) {
    Matrix36 temp;
    temp << Rot3::Hat(xi0.cameraLandmarks[static_cast<size_t>(i)].p) *
                X.Q()[static_cast<size_t>(i)].rotation().matrix(),
        -X.Q()[static_cast<size_t>(i)].scalar() *
            X.Q()[static_cast<size_t>(i)].rotation().matrix();
    A0t.block<3, 6>(VIOSensorState::CompDim + 3 * i, 15) = temp * commonTerm;
  }

  const Vector6 U_C = xiHat.sensor.cameraOffset.inverse().AdjointMap() * U_I;
  const Vector3 v_C = U_C.tail<3>();
  for (int i = 0; i < N; ++i) {
    const Matrix3 Qhat_i = X.Q()[static_cast<size_t>(i)].rotation().matrix() *
                           X.Q()[static_cast<size_t>(i)].scalar();
    const Vector3 qhat_i = xiHat.cameraLandmarks[static_cast<size_t>(i)].p;
    const Matrix3 A_qi =
        -Qhat_i *
        (Rot3::Hat(qhat_i) * Rot3::Hat(v_C) - 2.0 * v_C * qhat_i.transpose() +
         qhat_i * v_C.transpose()) *
        Qhat_i.inverse() * (1.0 / qhat_i.squaredNorm());
    A0t.block<3, 3>(VIOSensorState::CompDim + 3 * i,
                    VIOSensorState::CompDim + 3 * i) = A_qi;
  }

  return A0t;
}

Matrix EqFInputMatrixB_euclid(const VIOGroup& X, const VIOState& xi0) {
  const int N = static_cast<int>(xi0.n());
  Matrix Bt = Matrix::Zero(xi0.dim(), IMUVelocity::CompDim);

  const VIOState xiHat = stateGroupAction(X, xi0);
  const Pose3 A = APose(X);

  Bt.block<6, 6>(0, 6).setIdentity();

  const Matrix3 R_A = A.rotation().matrix();
  Bt.block<3, 3>(6, 0) = R_A;
  Bt.block<3, 3>(9, 0) = Rot3::Hat(A.translation()) * R_A;
  Bt.block<3, 3>(12, 0) = R_A * Rot3::Hat(xiHat.sensor.velocity);
  Bt.block<3, 3>(12, 3) = R_A;

  const Matrix3 RT_IC = xiHat.sensor.cameraOffset.rotation().matrix().transpose();
  const Vector3 x_IC = xiHat.sensor.cameraOffset.translation();
  for (int i = 0; i < N; ++i) {
    const Matrix3 Qhat_i = X.Q()[static_cast<size_t>(i)].rotation().matrix() *
                           X.Q()[static_cast<size_t>(i)].scalar();
    const Vector3 qhat_i = xiHat.cameraLandmarks[static_cast<size_t>(i)].p;
    Bt.block<3, 3>(VIOSensorState::CompDim + 3 * i, 0) =
        Qhat_i * (Rot3::Hat(qhat_i) * RT_IC + RT_IC * Rot3::Hat(x_IC));
  }

  return Bt;
}

Matrix23 EqFoutputMatrixCiStar_euclid(
    const Point3& q0, const SOT3& QHat,
    const std::shared_ptr<const VIOCameraModel>& camera, const Point2& y) {
  if (!camera) {
    throw std::invalid_argument("EqFoutputMatrixCiStar_euclid: null camera");
  }

  const Vector3 qHat = QHat.applyInverse(q0);
  const Vector3 yHat = qHat.normalized();

  Eigen::Matrix<double, 4, 3> m2g;
  m2g.block<3, 3>(0, 0) = -Rot3::Hat(q0);
  m2g.block<1, 3>(3, 0) = -q0.transpose();
  m2g /= q0.squaredNorm();

  auto DRho = [&camera](const Vector3& yVec) {
    Eigen::Matrix<double, 3, 4> DRhoVec;
    DRhoVec << Rot3::Hat(yVec), Vector3::Zero();
    return camera->projectionJacobian(yVec) * DRhoVec;
  };

  const Vector3 yTru = camera->undistortPoint(y);
  return 0.5 * (DRho(yTru) + DRho(yHat)) * QHat.inverse().AdjointMap() * m2g;
}

Vector liftInnovation_euclid(const Vector& totalInnovation, const VIOState& xi0) {
  if (totalInnovation.size() != xi0.dim()) {
    throw std::invalid_argument(
        "liftInnovation_euclid: innovation dimension mismatch");
  }

  const int N = static_cast<int>(xi0.n());
  Vector lift = Vector::Zero(21 + 4 * N);

  lift.segment<6>(9) = totalInnovation.segment<6>(0);   // beta
  lift.segment<6>(0) = totalInnovation.segment<6>(6);   // U_A

  const Vector3 gammaV = totalInnovation.segment<3>(12);
  lift.segment<3>(6) =
      -gammaV - Rot3::Hat(lift.segment<3>(0)) * xi0.sensor.velocity;  // u_w

  lift.segment<6>(15) =
      totalInnovation.segment<6>(15) +
      xi0.sensor.cameraOffset.inverse().AdjointMap() * lift.segment<6>(0);  // U_B

  for (int i = 0; i < N; ++i) {
    const Vector3 gammaQi0 =
        totalInnovation.segment<3>(VIOSensorState::CompDim + 3 * i);
    const Vector3 qi0 = xi0.cameraLandmarks[static_cast<size_t>(i)].p;

    lift.segment<3>(21 + 4 * i) = -qi0.cross(gammaQi0) / qi0.squaredNorm();
    lift(21 + 4 * i + 3) = -qi0.dot(gammaQi0) / qi0.squaredNorm();
  }

  return lift;
}

VIOGroup liftInnovationDiscrete_euclid(const Vector& totalInnovation,
                                       const VIOState& xi0) {
  if (totalInnovation.size() != xi0.dim()) {
    throw std::invalid_argument(
        "liftInnovationDiscrete_euclid: innovation dimension mismatch");
  }

  const Vector6 beta = totalInnovation.segment<6>(0);
  const Pose3 A_pose = Pose3::Expmap(totalInnovation.segment<6>(6));
  const Vector3 w = xi0.sensor.velocity -
                    A_pose.rotation().matrix() *
                        (xi0.sensor.velocity + totalInnovation.segment<3>(12));

  VIOGroup::SE23::Matrix3K x;
  x.col(0) = A_pose.translation();
  x.col(1) = w;
  const VIOGroup::SE23 A(A_pose.rotation(), x);

  const Pose3 B = xi0.sensor.cameraOffset.inverse()
                      .compose(A_pose)
                      .compose(xi0.sensor.cameraOffset)
                      .compose(Pose3::Expmap(totalInnovation.segment<6>(15)));

  std::vector<SOT3> Q;
  std::vector<int> ids;
  Q.reserve(xi0.n());
  ids.reserve(xi0.n());

  for (size_t i = 0; i < xi0.n(); ++i) {
    const Vector3 qi = xi0.cameraLandmarks[i].p;
    const Vector3 GammaQi =
        totalInnovation.segment<3>(VIOSensorState::CompDim + 3 * static_cast<int>(i));
    const Vector3 qi1 = qi + GammaQi;
    const Rot3 R = RotationFromTwoVectors(qi1, qi);
    const double a = qi.norm() / qi1.norm();
    Q.emplace_back(SOT3(SO3(R.matrix()), a));
    ids.push_back(xi0.cameraLandmarks[i].id);
  }

  return VIOGroup(A, beta, B, VIOGroup::LandmarkGroup(Q), ids);
}

}  // namespace

const EqFCoordinateSuite EqFCoordinateSuite_euclid{
    [](const VIOState& Xi, const VIOState& Xi0) { return Xi0.localCoordinates(Xi); },
    [](const Vector& eps, const VIOState& Xi0) { return Xi0.retract(eps); },
    EqFStateMatrixA_euclid,
    EqFInputMatrixB_euclid,
    EqFoutputMatrixCiStar_euclid,
    liftInnovation_euclid,
    liftInnovationDiscrete_euclid};

Matrix EqFCoordinateSuite::outputMatrixC(const VIOState& xi0, const VIOGroup& X,
                                         const VisionMeasurement& y,
                                         bool useEquivariance) const {
  const int M = static_cast<int>(xi0.n());
  const std::vector<int> yIds = y.getIds();
  const int N = static_cast<int>(yIds.size());

  Matrix C = Matrix::Zero(2 * N, VIOSensorState::CompDim + Landmark::CompDim * M);
  const VisionMeasurement yHat = measureSystemState(stateGroupAction(X, xi0), y.camera);

  for (int i = 0; i < M; ++i) {
    const int idNum = xi0.cameraLandmarks[static_cast<size_t>(i)].id;
    const auto itY = std::find(yIds.begin(), yIds.end(), idNum);
    if (itY == yIds.end()) continue;

    size_t k = static_cast<size_t>(i);
    if (!X.ids().empty()) {
      const auto itQ = std::find(X.ids().begin(), X.ids().end(), idNum);
      if (itQ == X.ids().end()) {
        throw std::invalid_argument("EqFCoordinateSuite::outputMatrixC: id not found in group");
      }
      k = static_cast<size_t>(std::distance(X.ids().begin(), itQ));
    }

    const int j = static_cast<int>(std::distance(yIds.begin(), itY));
    C.block<2, 3>(2 * j, VIOSensorState::CompDim + 3 * i) =
        useEquivariance ? outputMatrixCiStar(xi0.cameraLandmarks[static_cast<size_t>(i)].p,
                                             X.Q()[k], y.camera,
                                             y.camCoordinates.at(idNum))
                        : outputMatrixCi(xi0.cameraLandmarks[static_cast<size_t>(i)].p,
                                         X.Q()[k], y.camera);
  }

  (void)yHat;
  return C;
}

Matrix EqFCoordinateSuite::stateMatrixADiscrete(const VIOGroup& X,
                                                const VIOState& xi0,
                                                const IMUVelocity& imuVel,
                                                double dt) const {
  auto a0Discrete = [&](const Vector& epsilon) {
    const VIOState xiE = stateChartInv(epsilon, xi0);
    const VIOState xiHat = stateGroupAction(X, xi0);
    const VIOState xi = stateGroupAction(X, xiE);
    const VIOGroup lambdaTilde =
        liftVelocityDiscrete(xi, imuVel, dt) *
        liftVelocityDiscrete(xiHat, imuVel, dt).inverse();
    const VIOState xiE1 = stateGroupAction(X * lambdaTilde * X.inverse(), xiE);
    return stateChart(xiE1, xi0);
  };

  return NumericalDifferential(a0Discrete, Vector::Zero(xi0.dim()));
}

Matrix23 EqFCoordinateSuite::outputMatrixCi(
    const Point3& q0, const SOT3& QHat,
    const std::shared_ptr<const VIOCameraModel>& camera) const {
  const Vector3 qHat = QHat.applyInverse(q0);
  const Point2 yHat = camera->projectPoint(qHat);
  return outputMatrixCiStar(q0, QHat, camera, yHat);
}

const EqFCoordinateSuite* getCoordinates(CoordinateChoice coordinateChoice) {
  if (coordinateChoice == CoordinateChoice::Euclidean) {
    return &EqFCoordinateSuite_euclid;
  }
  return nullptr;
}

}  // namespace gtsam
