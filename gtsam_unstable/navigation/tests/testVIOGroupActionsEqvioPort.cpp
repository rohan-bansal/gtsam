/* ----------------------------------------------------------------------------

 * GTSAM Copyright 2010, Georgia Tech Research Corporation,
 * Atlanta, Georgia 30332-0415
 * All Rights Reserved
 * Authors: Frank Dellaert, et al. (see THANKS for the full author list)

 * See LICENSE for the license information

 * -------------------------------------------------------------------------- */

/**
 * @file   testVIOGroupActionsEqvioPort.cpp
 * @brief  Eqvio-style port of test_VIOGroupActions.cpp
 */

#include <CppUnitLite/TestHarness.h>

#include <gtsam_unstable/navigation/VIOSymmetry.h>

#include "VIOEqvioTestUtils.h"

#include <vector>

using namespace gtsam;
using namespace gtsam::eqvio_test_util;

namespace {

constexpr int kTestReps = 25;
constexpr double kNearZero = 1e-12;

}  // namespace

//******************************************************************************
TEST(VIOGroupActionsEqvioPort, StateAction) {
  srand(0);
  const std::vector<int> ids = {0, 1, 2, 3, 4};
  const VIOGroup groupId = VIOGroup::Identity(ids);

  for (int rep = 0; rep < kTestReps; ++rep) {
    const VIOGroup X1 = RandomGroupElement(ids);
    const VIOGroup X2 = RandomGroupElement(ids);
    const VIOState xi0 = RandomStateElement(ids);

    const double dist00 = StateDistance(xi0, xi0);
    EXPECT(dist00 <= kNearZero);

    const VIOState xi0Id = stateGroupAction(groupId, xi0);
    const double dist0Id = StateDistance(xi0Id, xi0);
    EXPECT(dist0Id <= kNearZero);

    const VIOState xi1 = stateGroupAction(X2, stateGroupAction(X1, xi0));
    const VIOState xi2 = stateGroupAction(X1 * X2, xi0);
    const double dist12 = StateDistance(xi1, xi2);
    EXPECT(dist12 <= kNearZero);
  }
}

//******************************************************************************
TEST(VIOGroupActionsEqvioPort, OutputAction) {
  srand(0);
  const std::vector<int> ids = {0, 1, 2, 3, 4};
  const VIOGroup groupId = VIOGroup::Identity(ids);
  const auto camera = CreateDefaultCamera();

  for (int rep = 0; rep < kTestReps; ++rep) {
    const VIOGroup X1 = RandomGroupElement(ids);
    const VIOGroup X2 = RandomGroupElement(ids);
    const VisionMeasurement y0 = RandomVisionMeasurement(ids, camera);

    const double dist00 = MeasurementDistance(y0, y0);
    EXPECT(dist00 <= 1e-5);

    const VisionMeasurement y0Id = outputGroupAction(groupId, y0);
    const double dist0Id = MeasurementDistance(y0Id, y0);
    EXPECT(dist0Id <= 1e-5);

    const VisionMeasurement y1 = outputGroupAction(X2, outputGroupAction(X1, y0));
    const VisionMeasurement y2 = outputGroupAction(X1 * X2, y0);
    const double dist12 = MeasurementDistance(y1, y2);
    EXPECT(dist12 <= 1e-5);
  }
}

//******************************************************************************
TEST(VIOGroupActionsEqvioPort, OutputEquivariance) {
  srand(0);
  const std::vector<int> ids = {5, 0, 1, 2, 3, 4};
  const auto camera = CreateDefaultCamera();

  for (int rep = 0; rep < kTestReps; ++rep) {
    const VIOGroup X = RandomGroupElement(ids);
    const VIOState xi0 = RandomStateElement(ids);

    const VisionMeasurement y1 = measureSystemState(stateGroupAction(X, xi0), camera);
    const VisionMeasurement y2 = outputGroupAction(X, measureSystemState(xi0, camera));
    const double dist12 = MeasurementDistance(y1, y2);
    EXPECT(dist12 <= 1e-5);
  }
}

int main() {
  TestResult tr;
  return TestRegistry::runAllTests(tr);
}

