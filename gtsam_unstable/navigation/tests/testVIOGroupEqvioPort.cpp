/* ----------------------------------------------------------------------------

 * GTSAM Copyright 2010, Georgia Tech Research Corporation,
 * Atlanta, Georgia 30332-0415
 * All Rights Reserved
 * Authors: Frank Dellaert, et al. (see THANKS for the full author list)

 * See LICENSE for the license information

 * -------------------------------------------------------------------------- */

/**
 * @file   testVIOGroupEqvioPort.cpp
 * @brief  Eqvio-style port of test_VIOGroup.cpp
 */

#include <CppUnitLite/TestHarness.h>

#include "VIOEqvioTestUtils.h"

#include <vector>

using namespace gtsam;
using namespace gtsam::eqvio_test_util;

namespace {

constexpr int kTestReps = 25;
constexpr double kNearZero = 1e-12;

}  // namespace

//******************************************************************************
TEST(VIOGroupEqvioPort, BasicOperations) {
  srand(0);
  const std::vector<int> allIds = {0, 1, 2, 3, 4};
  const VIOGroup groupId = VIOGroup::Identity(allIds);

  for (int rep = 0; rep < kTestReps; ++rep) {
    const VIOGroup X1 = RandomGroupElement(allIds);
    const VIOGroup X2 = RandomGroupElement(allIds);
    const VIOGroup X3 = RandomGroupElement(allIds);

    const double inverseError1 = LogNorm(X1.inverse() * X1);
    const double inverseError2 = LogNorm(X1 * X1.inverse());
    EXPECT(inverseError1 <= kNearZero);
    EXPECT(inverseError2 <= kNearZero);

    const VIOGroup result12 = (X1 * X2) * X3;
    const VIOGroup result23 = X1 * (X2 * X3);
    const double assocError1 = LogNorm(result12.inverse() * result23);
    const double assocError2 = LogNorm(result23.inverse() * result12);
    const double assocError3 = LogNorm(result12 * result23.inverse());
    const double assocError4 = LogNorm(result23 * result12.inverse());
    EXPECT(assocError1 <= kNearZero);
    EXPECT(assocError2 <= kNearZero);
    EXPECT(assocError3 <= kNearZero);
    EXPECT(assocError4 <= kNearZero);

    const double idError1 = LogNorm(groupId);
    const double idError2 = LogNorm((X1 * groupId) * X1.inverse());
    const double idError3 = LogNorm(X1.inverse() * (groupId * X1));
    EXPECT(idError1 <= kNearZero);
    EXPECT(idError2 <= kNearZero);
    EXPECT(idError3 <= kNearZero);
  }
}

int main() {
  TestResult tr;
  return TestRegistry::runAllTests(tr);
}

