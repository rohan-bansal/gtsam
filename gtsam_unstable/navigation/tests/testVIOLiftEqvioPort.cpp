/* ----------------------------------------------------------------------------

 * GTSAM Copyright 2010, Georgia Tech Research Corporation,
 * Atlanta, Georgia 30332-0415
 * All Rights Reserved
 * Authors: Frank Dellaert, et al. (see THANKS for the full author list)

 * See LICENSE for the license information

 * -------------------------------------------------------------------------- */

/**
 * @file   testVIOLiftEqvioPort.cpp
 * @brief  Eqvio-style port of test_VIOLift.cpp
 */

#include <CppUnitLite/TestHarness.h>

#include <gtsam_unstable/navigation/VIOEqFMatrices.h>
#include <gtsam_unstable/navigation/VIOSymmetry.h>

#include "VIOEqvioTestUtils.h"

#include <cmath>
#include <limits>
#include <vector>

using namespace gtsam;
using namespace gtsam::eqvio_test_util;

namespace {

constexpr int kTestReps = 25;
constexpr double kNearZero = 1e-12;

bool InnovationLiftTest(const EqFCoordinateSuite* suite) {
  if (!suite) return false;

  const std::vector<int> ids = {0, 1, 2, 3, 4};
  for (int rep = 0; rep < kTestReps; ++rep) {
    const VIOState xi0 = RandomStateElement(ids);

    const auto reprojectionFunc = [&](const Vector& eps) {
      const Vector liftedInnovation = suite->liftInnovation(eps, xi0);
      const VIOGroup Delta = VIOGroup::Expmap(liftedInnovation);
      const VIOState xi1 = stateGroupAction(Delta, xi0);
      return suite->stateChart(xi1, xi0);
    };

    const int d = xi0.dim();
    const double h = std::cbrt(std::numeric_limits<double>::epsilon());
    const Matrix numericalDf =
        NumericalDifferential(reprojectionFunc, Vector::Zero(d), h);
    const Matrix expected = Matrix::Identity(d, d);
    if (!MatrixClose(expected, numericalDf, h)) return false;
  }
  return true;
}

bool DiscreteInnovationLiftTest(const EqFCoordinateSuite* suite) {
  if (!suite) return false;

  const std::vector<int> ids = {0, 1, 2, 3, 4};
  for (int rep = 0; rep < kTestReps; ++rep) {
    const VIOState xi0 = RandomStateElement(ids);

    const auto reprojectionFunc = [&](const Vector& eps) {
      const VIOGroup discreteInn = suite->liftInnovationDiscrete(eps, xi0);
      const VIOState xi1 = stateGroupAction(discreteInn, xi0);
      return suite->stateChart(xi1, xi0);
    };

    for (int j = 0; j < xi0.dim(); ++j) {
      const Vector ej = Vector::Unit(xi0.dim(), j);
      const Vector reproj = reprojectionFunc(ej);
      if ((reproj - ej).norm() > 1e-5) return false;
    }
  }
  return true;
}

}  // namespace

//******************************************************************************
TEST(VIOLiftEqvioPort, Lift) {
  srand(0);
  const std::vector<int> ids = {0, 1, 2, 3, 4};

  for (int rep = 0; rep < kTestReps; ++rep) {
    const VIOState xi0 = RandomStateElement(ids);
    const IMUVelocity velocity = RandomVelocityElement();

    double previousDist = 1e8;
    for (int i = 0; i < 8; ++i) {
      const double dt = std::pow(10.0, -i);
      const VIOState xi1 = integrateSystemFunction(xi0, velocity, dt);

      const Vector lambda = liftVelocity(xi0, velocity);
      const VIOGroup lambdaExp = VIOGroup::Expmap(dt * lambda);
      const VIOState xi2 = stateGroupAction(lambdaExp, xi0);

      const double diffDist = StateDistance(xi1, xi2) / dt;
      EXPECT(diffDist <= previousDist);
      previousDist = diffDist;
    }
  }
}

//******************************************************************************
TEST(VIOLiftEqvioPort, DiscreteLift) {
  srand(0);
  const std::vector<int> ids = {0, 1, 2, 3, 4};
  const double dt = 0.1;

  for (int rep = 0; rep < kTestReps; ++rep) {
    const VIOState xi0 = RandomStateElement(ids);
    const IMUVelocity velocity = RandomVelocityElement();

    const VIOState xi1 = integrateSystemFunction(xi0, velocity, dt);
    const VIOGroup X = liftVelocityDiscrete(xi0, velocity, dt);
    const VIOState xi2 = stateGroupAction(X, xi0);

    const double dist12 = StateDistance(xi1, xi2);
    EXPECT(dist12 <= kNearZero);
  }
}

//******************************************************************************
TEST(VIOLiftEqvioPort, InnovationLiftsEuclid) {
  EXPECT(InnovationLiftTest(getCoordinates(CoordinateChoice::Euclidean)));
  EXPECT(
      DiscreteInnovationLiftTest(getCoordinates(CoordinateChoice::Euclidean)));
}

//******************************************************************************
TEST(VIOLiftEqvioPort, InnovationLiftsInvDepth) {
  EXPECT(InnovationLiftTest(getCoordinates(CoordinateChoice::InvDepth)));
  EXPECT(
      DiscreteInnovationLiftTest(getCoordinates(CoordinateChoice::InvDepth)));
}

int main() {
  TestResult tr;
  return TestRegistry::runAllTests(tr);
}
