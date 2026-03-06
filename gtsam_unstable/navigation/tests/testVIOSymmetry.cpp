/* ----------------------------------------------------------------------------

 * GTSAM Copyright 2010, Georgia Tech Research Corporation,
 * Atlanta, Georgia 30332-0415
 * All Rights Reserved
 * Authors: Frank Dellaert, et al. (see THANKS for the full author list)

 * See LICENSE for the license information

 * -------------------------------------------------------------------------- */

/**
 * @file   testVIOSymmetry.cpp
 * @brief  Unit tests for VIO symmetry actions and lift helpers
 */

#include <CppUnitLite/TestHarness.h>
#include <gtsam/base/GroupAction.h>
#include <gtsam/base/TestableAssertions.h>
#include <gtsam_unstable/navigation/VIOSymmetry.h>

#include <cmath>
#include <memory>
#include <stdexcept>
#include <vector>

using namespace gtsam;

namespace {

class SimplePinholeCamera final : public VIOCameraModel {
 public:
  Point2 projectPoint(const Point3& p) const override {
    if (std::abs(p.z()) < 1e-12) {
      throw std::invalid_argument("SimplePinholeCamera: z is near zero");
    }
    return Point2(p.x() / p.z(), p.y() / p.z());
  }

  Vector3 undistortPoint(const Point2& y) const override {
    return Vector3(y.x(), y.y(), 1.0).normalized();
  }

  Matrix23 projectionJacobian(const Vector3& y) const override {
    if (std::abs(y.z()) < 1e-12) {
      throw std::invalid_argument("SimplePinholeCamera: z is near zero");
    }
    Matrix23 J;
    const double z2 = y.z() * y.z();
    J << 1.0 / y.z(), 0.0, -y.x() / z2, 0.0, 1.0 / y.z(), -y.y() / z2;
    return J;
  }
};

VIOGroup::SE23 MakeA(const Rot3& R, const Point3& t, const Vector3& w) {
  VIOGroup::SE23::Matrix3K x;
  x.col(0) = t;
  x.col(1) = w;
  return VIOGroup::SE23(R, x);
}

std::vector<Landmark> Lms0() { return {}; }
std::vector<Landmark> Lms1() { return {{Point3(1.0, -0.3, 4.2), 11}}; }
std::vector<Landmark> Lms3() {
  return {{Point3(1.0, -0.3, 4.2), 11},
          {Point3(-0.7, 0.4, 3.1), 22},
          {Point3(0.2, 0.8, 5.5), 33}};
}

VIOSensorState SensorA() {
  VIOSensorState s;
  s.inputBias = (Vector6() << 0.05, -0.04, 0.03, 0.01, -0.02, 0.04).finished();
  s.pose = Pose3(Rot3::RzRyRx(0.1, -0.2, 0.25), Point3(0.4, -0.1, 1.0));
  s.velocity = Vector3(0.3, -0.2, 0.1);
  s.cameraOffset =
      Pose3(Rot3::RzRyRx(-0.03, 0.05, -0.02), Point3(0.1, 0.02, 0.07));
  return s;
}

VIOState State0() { return VIOState(SensorA(), Lms0()); }
VIOState State1() { return VIOState(SensorA(), Lms1()); }
VIOState State3() { return VIOState(SensorA(), Lms3()); }

VIOGroup Group0() {
  const Rot3 R = Rot3::RzRyRx(0.02, -0.03, 0.04);
  const Point3 t(0.05, -0.02, 0.03);
  const Vector3 w(0.01, -0.03, 0.02);
  return VIOGroup(MakeA(R, t, w),
                  (Vector6() << 0.01, -0.01, 0.02, -0.02, 0.01, 0.0).finished(),
                  Pose3(Rot3::RzRyRx(-0.02, 0.01, 0.03), Point3(0.02, 0.0, -0.01)),
                  VIOGroup::LandmarkGroup::Identity(0));
}

VIOGroup Group1() {
  const Rot3 R = Rot3::RzRyRx(0.02, -0.03, 0.04);
  const Point3 t(0.05, -0.02, 0.03);
  const Vector3 w(0.01, -0.03, 0.02);
  const SOT3 q1(SO3::Expmap((Vector3() << 0.03, -0.02, 0.01).finished()), 1.1);
  return VIOGroup(MakeA(R, t, w),
                  (Vector6() << 0.01, -0.01, 0.02, -0.02, 0.01, 0.0).finished(),
                  Pose3(Rot3::RzRyRx(-0.02, 0.01, 0.03), Point3(0.02, 0.0, -0.01)),
                  VIOGroup::LandmarkGroup({q1}), {11});
}

VIOGroup Group3() {
  const Rot3 R = Rot3::RzRyRx(0.02, -0.03, 0.04);
  const Point3 t(0.05, -0.02, 0.03);
  const Vector3 w(0.01, -0.03, 0.02);
  const SOT3 q1(SO3::Expmap((Vector3() << 0.03, -0.02, 0.01).finished()), 1.1);
  const SOT3 q2(SO3::Expmap((Vector3() << -0.01, 0.04, -0.02).finished()), 0.95);
  const SOT3 q3(SO3::Expmap((Vector3() << 0.02, 0.01, 0.03).finished()), 1.05);
  return VIOGroup(MakeA(R, t, w),
                  (Vector6() << 0.01, -0.01, 0.02, -0.02, 0.01, 0.0).finished(),
                  Pose3(Rot3::RzRyRx(-0.02, 0.01, 0.03), Point3(0.02, 0.0, -0.01)),
                  VIOGroup::LandmarkGroup({q1, q2, q3}), {11, 22, 33});
}

VIOGroup Group3b() {
  const Rot3 R = Rot3::RzRyRx(-0.04, 0.01, -0.02);
  const Point3 t(-0.03, 0.04, -0.01);
  const Vector3 w(-0.02, 0.01, 0.03);
  const SOT3 q1(SO3::Expmap((Vector3() << -0.02, 0.01, 0.04).finished()), 1.02);
  const SOT3 q2(SO3::Expmap((Vector3() << 0.02, -0.03, 0.01).finished()), 0.98);
  const SOT3 q3(SO3::Expmap((Vector3() << 0.01, 0.02, -0.02).finished()), 1.08);
  return VIOGroup(MakeA(R, t, w),
                  (Vector6() << -0.02, 0.03, -0.01, 0.02, 0.01, -0.01).finished(),
                  Pose3(Rot3::RzRyRx(0.01, -0.02, 0.04), Point3(-0.01, 0.03, 0.02)),
                  VIOGroup::LandmarkGroup({q1, q2, q3}), {11, 22, 33});
}

Matrix NumericalDerivativeWrtGroup(const VIOSymmetry& phi, const VIOGroup& X,
                                   const VIOState& xi, const VIOState& y0,
                                   double h = 1e-6) {
  Matrix H = Matrix::Zero(y0.dim(), static_cast<int>(X.dim()));
  for (int j = 0; j < static_cast<int>(X.dim()); ++j) {
    Vector dx = Vector::Zero(X.dim());
    dx(j) = h;
    const VIOState yPlus = phi(xi, X.retract(dx));
    dx(j) = -h;
    const VIOState yMinus = phi(xi, X.retract(dx));
    H.col(j) =
        (y0.localCoordinates(yPlus) - y0.localCoordinates(yMinus)) / (2.0 * h);
  }
  return H;
}

Matrix NumericalDerivativeWrtState(const VIOSymmetry& phi, const VIOGroup& X,
                                   const VIOState& xi, const VIOState& y0,
                                   double h = 1e-6) {
  Matrix H = Matrix::Zero(y0.dim(), xi.dim());
  for (int j = 0; j < xi.dim(); ++j) {
    Vector dxi = Vector::Zero(xi.dim());
    dxi(j) = h;
    const VIOState yPlus = phi(xi.retract(dxi), X);
    dxi(j) = -h;
    const VIOState yMinus = phi(xi.retract(dxi), X);
    H.col(j) =
        (y0.localCoordinates(yPlus) - y0.localCoordinates(yMinus)) / (2.0 * h);
  }
  return H;
}

}  // namespace

//******************************************************************************
TEST(VIOSymmetry, RightActionLaw) {
  const VIOSymmetry phi;
  const VIOState xi = State3();
  const VIOGroup X1 = Group3();
  const VIOGroup X2 = Group3b();
  EXPECT_RIGHT_ACTION(phi, xi, X1, X2);
}

//******************************************************************************
TEST(VIOSymmetry, JacobiansN0) {
  const VIOSymmetry phi;
  const VIOGroup X = Group0();
  const VIOState xi = State0();

  Matrix HX, HXi;
  const VIOState y = phi(xi, X, HXi, HX);
  const Matrix HXNum = NumericalDerivativeWrtGroup(phi, X, xi, y);
  const Matrix HXiNum = NumericalDerivativeWrtState(phi, X, xi, y);

  EXPECT(assert_equal(HXNum, HX, 1e-5));
  EXPECT(assert_equal(HXiNum, HXi, 1e-5));
}

//******************************************************************************
TEST(VIOSymmetry, JacobiansN1) {
  const VIOSymmetry phi;
  const VIOGroup X = Group1();
  const VIOState xi = State1();

  Matrix HX, HXi;
  const VIOState y = phi(xi, X, HXi, HX);
  const Matrix HXNum = NumericalDerivativeWrtGroup(phi, X, xi, y);
  const Matrix HXiNum = NumericalDerivativeWrtState(phi, X, xi, y);

  EXPECT(assert_equal(HXNum, HX, 1e-5));
  EXPECT(assert_equal(HXiNum, HXi, 1e-5));
}

//******************************************************************************
TEST(VIOSymmetry, JacobiansN3) {
  const VIOSymmetry phi;
  const VIOGroup X = Group3();
  const VIOState xi = State3();

  Matrix HX, HXi;
  const VIOState y = phi(xi, X, HXi, HX);
  const Matrix HXNum = NumericalDerivativeWrtGroup(phi, X, xi, y);
  const Matrix HXiNum = NumericalDerivativeWrtState(phi, X, xi, y);

  EXPECT(assert_equal(HXNum, HX, 1e-5));
  EXPECT(assert_equal(HXiNum, HXi, 1e-5));
}

//******************************************************************************
TEST(VIOSymmetry, OutputEquivariance) {
  const auto camera = std::make_shared<SimplePinholeCamera>();
  const VIOState xi = State3();
  const VIOGroup X = Group3();

  const VisionMeasurement y = measureSystemState(xi, camera);
  const VisionMeasurement lhs = outputGroupAction(X, y);
  const VisionMeasurement rhs = measureSystemState(stateGroupAction(X, xi), camera);

  EXPECT(lhs.equals(rhs, 1e-8));
}

//******************************************************************************
TEST(VIOSymmetry, LiftAndIntegrationSanity) {
  const VIOState xi = State3();
  IMUVelocity imu;
  imu.gyr = Vector3(0.03, -0.02, 0.01);
  imu.acc = Vector3(0.2, -0.1, 9.75);
  imu.gyrBiasVel = Vector3(0.01, 0.0, -0.02);
  imu.accBiasVel = Vector3(-0.01, 0.02, 0.0);

  const Vector l = liftVelocity(xi, imu);
  EXPECT_LONGS_EQUAL(33, l.size());

  const double dt = 1e-3;
  const VIOGroup delta = liftVelocityDiscrete(xi, imu, dt);
  const VIOState xiLifted = stateGroupAction(delta, xi);
  const VIOState xiIntegrated = integrateSystemFunction(xi, imu, dt);
  EXPECT(assert_equal(xiIntegrated, xiLifted, 1e-7));
}

//******************************************************************************
TEST(VIOOutputSymmetry, JacobianSizes) {
  const auto camera = std::make_shared<SimplePinholeCamera>();
  const VIOState xi = State3();
  const VIOGroup X = Group3();
  const VisionMeasurement y = measureSystemState(xi, camera);
  const VIOOutputSymmetry rho;

  Matrix HX, Hy;
  const VisionMeasurement out = rho(y, X, Hy, HX);

  EXPECT_LONGS_EQUAL(out.dim(), HX.rows());
  EXPECT_LONGS_EQUAL(static_cast<long>(X.dim()), HX.cols());
  EXPECT_LONGS_EQUAL(out.dim(), Hy.rows());
  EXPECT_LONGS_EQUAL(y.dim(), Hy.cols());
  EXPECT(HX.array().isFinite().all());
  EXPECT(Hy.array().isFinite().all());
}

int main() {
  TestResult tr;
  return TestRegistry::runAllTests(tr);
}
