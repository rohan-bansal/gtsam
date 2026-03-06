/* ----------------------------------------------------------------------------

 * GTSAM Copyright 2010, Georgia Tech Research Corporation,
 * Atlanta, Georgia 30332-0415
 * All Rights Reserved
 * Authors: Frank Dellaert, et al. (see THANKS for the full author list)

 * See LICENSE for the license information

 * -------------------------------------------------------------------------- */

/**
 * @file   VIOEqvioTestUtils.h
 * @brief  Utilities for eqvio-style parity tests in unstable navigation
 */

#pragma once

#include <gtsam/base/Matrix.h>
#include <gtsam/base/Vector.h>
#include <gtsam/geometry/Pose3.h>
#include <gtsam/geometry/SO3.h>
#include <gtsam_unstable/navigation/VIOCommon.h>
#include <gtsam_unstable/navigation/VIOGroup.h>
#include <gtsam_unstable/navigation/VIOState.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <functional>
#include <limits>
#include <memory>
#include <stdexcept>
#include <vector>

namespace gtsam {
namespace eqvio_test_util {

class SimplePinholeCamera final : public VIOCameraModel {
 public:
  explicit SimplePinholeCamera(double fx = 450.0, double fy = 450.0,
                               double cx = 400.0, double cy = 240.0)
      : fx_(fx), fy_(fy), cx_(cx), cy_(cy) {}

  Point2 projectPoint(const Point3& p) const override {
    if (std::abs(p.z()) < 1e-12) {
      throw std::invalid_argument("SimplePinholeCamera: z is near zero");
    }
    return Point2(fx_ * p.x() / p.z() + cx_, fy_ * p.y() / p.z() + cy_);
  }

  Vector3 undistortPoint(const Point2& y) const override {
    return Vector3((y.x() - cx_) / fx_, (y.y() - cy_) / fy_, 1.0);
  }

  Matrix23 projectionJacobian(const Vector3& y) const override {
    if (std::abs(y.z()) < 1e-12) {
      throw std::invalid_argument("SimplePinholeCamera: z is near zero");
    }
    Matrix23 J;
    const double z2 = y.z() * y.z();
    J << fx_ / y.z(), 0.0, -fx_ * y.x() / z2, 0.0, fy_ / y.z(),
        -fy_ * y.y() / z2;
    return J;
  }

 private:
  double fx_, fy_, cx_, cy_;
};

inline std::shared_ptr<const VIOCameraModel> CreateDefaultCamera() {
  return std::make_shared<SimplePinholeCamera>();
}

inline VIOGroup::SE23 MakeA(const Rot3& R, const Point3& t, const Vector3& w) {
  VIOGroup::SE23::Matrix3K x;
  x.col(0) = t;
  x.col(1) = w;
  return VIOGroup::SE23(R, x);
}

inline VIOState RandomStateElement(const std::vector<int>& ids) {
  VIOSensorState sensor;
  sensor.inputBias = Vector6::Random();
  sensor.pose = Pose3::Expmap(Vector6::Random());
  sensor.velocity = Vector3::Random();
  sensor.cameraOffset = Pose3::Expmap(Vector6::Random());

  std::vector<Landmark> lms(ids.size());
  for (size_t i = 0; i < ids.size(); ++i) {
    Point3 p = 10.0 * Vector3::Random();
    if (std::abs(p.z()) < 1e-3) p.z() += (p.z() >= 0.0 ? 1.0 : -1.0);
    lms[i] = Landmark{p, ids[i]};
  }
  return VIOState(sensor, lms);
}

inline VIOGroup RandomGroupElement(const std::vector<int>& ids) {
  const Pose3 Apose = Pose3::Expmap(Vector6::Random());
  const Vector3 w = Vector3::Random();
  const Pose3 B = Pose3::Expmap(Vector6::Random());
  const Vector6 beta = Vector6::Random();

  std::vector<SOT3> Q(ids.size());
  for (size_t i = 0; i < ids.size(); ++i) {
    Q[i] = SOT3(SO3::Expmap(Vector3::Random()),
                2.0 * static_cast<double>(rand()) / RAND_MAX + 1.0);
  }

  return VIOGroup(MakeA(Apose.rotation(), Apose.translation(), w), beta, B,
                  VIOGroup::LandmarkGroup(Q), ids);
}

inline IMUVelocity RandomVelocityElement() {
  IMUVelocity vel;
  vel.gyr = Vector3::Random();
  vel.acc = Vector3::Random();
  vel.gyrBiasVel = Vector3::Random();
  vel.accBiasVel = Vector3::Random();
  vel.stamp = 0.0;
  return vel;
}

inline VisionMeasurement RandomVisionMeasurement(
    const std::vector<int>& ids,
    const std::shared_ptr<const VIOCameraModel>& camera) {
  VisionMeasurement measurement;
  measurement.camera = camera;
  measurement.stamp = 0.0;

  for (int id : ids) {
    Vector3 p;
    do {
      p = Vector3::Random();
    } while (p.norm() < 1e-9);
    p.normalize();
    while (p.z() < 1e-1) {
      p = Vector3::Random().normalized();
    }
    measurement.camCoordinates[id] = camera->projectPoint(p);
  }
  return measurement;
}

inline double LogNorm(const VIOGroup& X) { return VIOGroup::Logmap(X).norm(); }

inline double StateDistance(const VIOState& xi1, const VIOState& xi2) {
  if (xi1.n() != xi2.n()) {
    throw std::invalid_argument("StateDistance: landmark count mismatch");
  }

  double dist = 0.0;
  dist += (xi1.sensor.inputBias - xi2.sensor.inputBias).norm();
  dist += xi1.sensor.pose.localCoordinates(xi2.sensor.pose).norm();
  dist += xi1.sensor.cameraOffset.localCoordinates(xi2.sensor.cameraOffset).norm();
  dist += (xi1.sensor.velocity - xi2.sensor.velocity).norm();

  for (size_t i = 0; i < xi1.n(); ++i) {
    if (xi1.cameraLandmarks[i].id != xi2.cameraLandmarks[i].id) {
      throw std::invalid_argument("StateDistance: landmark ids mismatch");
    }
    dist += (xi1.cameraLandmarks[i].p - xi2.cameraLandmarks[i].p).norm();
  }
  return dist;
}

inline double MeasurementDistance(const VisionMeasurement& y1,
                                  const VisionMeasurement& y2) {
  Vector y1vec = Vector(y1);
  Vector y2vec = Vector(y2);
  const double scale = std::max(1.0, std::max(y1vec.norm(), y2vec.norm()));
  const Vector diff = Vector(y1 - y2);
  return diff.norm() / scale;
}

inline Matrix NumericalDifferential(const std::function<Vector(const Vector&)>& f,
                                    const Vector& x0, double h) {
  const int n = static_cast<int>(x0.size());
  const Vector y0 = f(x0);
  const int m = static_cast<int>(y0.size());
  Matrix J = Matrix::Zero(m, n);
  for (int j = 0; j < n; ++j) {
    Vector dx = Vector::Zero(n);
    dx(j) = h;
    J.col(j) = (f(x0 + dx) - f(x0 - dx)) / (2.0 * h);
  }
  return J;
}

inline bool MatrixClose(const Matrix& A, const Matrix& B, double h = -1.0) {
  if (A.rows() != B.rows() || A.cols() != B.cols()) return false;
  if (!A.array().isFinite().all() || !B.array().isFinite().all()) return false;

  if (h < 0.0) h = std::cbrt(std::numeric_limits<double>::epsilon());

  for (int i = 0; i < A.rows(); ++i) {
    for (int j = 0; j < A.cols(); ++j) {
      const double tol = std::max(h, h * 1e1 * std::abs(A(i, j)));
      if (std::abs(A(i, j) - B(i, j)) > tol) return false;
    }
  }
  return true;
}

}  // namespace eqvio_test_util
}  // namespace gtsam

