/* ----------------------------------------------------------------------------

 * GTSAM Copyright 2010, Georgia Tech Research Corporation,
 * Atlanta, Georgia 30332-0415
 * All Rights Reserved
 * Authors: Frank Dellaert, et al. (see THANKS for the full author list)

 * See LICENSE for the license information

 * -------------------------------------------------------------------------- */

/**
 * @file   testPowerLieGroup.cpp
 * @brief  Unit tests for PowerLieGroup
 */

#include <CppUnitLite/TestHarness.h>
#include <gtsam/base/ProductLieGroup.h>
#include <gtsam/base/TestableAssertions.h>
#include <gtsam/base/numericalDerivative.h>
#include <gtsam/base/testLie.h>
#include <gtsam/geometry/Rot3.h>
#include <gtsam/geometry/SOT3.h>

#include <functional>
#include <stdexcept>

using namespace gtsam;

using FixedPower = PowerLieGroup<Rot3, 3>;
using DynamicPower = PowerLieGroup<SOT3, Eigen::Dynamic>;

namespace {

template <int N, typename G>
void testLieGroupDerivativesN(TestResult& result_, const std::string& name_,
                              const G& t1, const G& t2) {
  Matrix H1, H2;
  using T = traits<G>;
  using OJ = OptionalJacobian<T::dimension, T::dimension>;

  OJ none;
  EXPECT(assert_equal<G>(t1.inverse(), T::Inverse(t1, H1)));
  EXPECT(assert_equal(numericalDerivative21<G, G, OJ, N>(T::Inverse, t1, none),
                      H1));

  EXPECT(assert_equal<G>(t2.inverse(), T::Inverse(t2, H1)));
  EXPECT(assert_equal(numericalDerivative21<G, G, OJ, N>(T::Inverse, t2, none),
                      H1));

  EXPECT(assert_equal<G>(t1 * t2, T::Compose(t1, t2, H1, H2)));
  EXPECT(assert_equal(
      numericalDerivative41<G, G, G, OJ, OJ, N>(T::Compose, t1, t2, none, none),
      H1));
  EXPECT(assert_equal(
      numericalDerivative42<G, G, G, OJ, OJ, N>(T::Compose, t1, t2, none, none),
      H2));

  EXPECT(assert_equal<G>(t1.inverse() * t2, T::Between(t1, t2, H1, H2)));
  EXPECT(assert_equal(
      numericalDerivative41<G, G, G, OJ, OJ, N>(T::Between, t1, t2, none, none),
      H1));
  EXPECT(assert_equal(
      numericalDerivative42<G, G, G, OJ, OJ, N>(T::Between, t1, t2, none, none),
      H2));
}

template <int N, typename G>
void testChartDerivativesN(TestResult& result_, const std::string& name_,
                           const G& t1, const G& t2) {
  Matrix H1, H2;
  using T = traits<G>;
  using V = typename T::TangentVector;
  using OJ = OptionalJacobian<T::dimension, T::dimension>;

  OJ none;
  const V w12 = T::Local(t1, t2);
  EXPECT(assert_equal<G>(t2, T::Retract(t1, w12, H1, H2)));
  EXPECT(assert_equal(
      numericalDerivative41<G, G, V, OJ, OJ, N>(T::Retract, t1, w12, none, none),
      H1));
  EXPECT(assert_equal(
      numericalDerivative42<G, G, V, OJ, OJ, N>(T::Retract, t1, w12, none, none),
      H2));

  EXPECT(assert_equal(w12, T::Local(t1, t2, H1, H2)));
  EXPECT(assert_equal(
      numericalDerivative41<V, G, G, OJ, OJ, N>(T::Local, t1, t2, none, none),
      H1));
  EXPECT(assert_equal(
      numericalDerivative42<V, G, G, OJ, OJ, N>(T::Local, t1, t2, none, none),
      H2));
}

const Rot3 kR1 = Rot3::RzRyRx(0.1, -0.2, 0.3);
const Rot3 kR2 = Rot3::RzRyRx(-0.2, 0.1, -0.4);
const Rot3 kR3 = Rot3::RzRyRx(0.3, 0.2, -0.1);
const Rot3 kR4 = Rot3::RzRyRx(-0.1, -0.25, 0.15);
const Rot3 kR5 = Rot3::RzRyRx(0.05, 0.07, -0.09);
const Rot3 kR6 = Rot3::RzRyRx(-0.08, 0.04, 0.12);

const SOT3 kQ1(SO3::Expmap((Vector3() << 0.12, -0.05, 0.08).finished()), 1.2);
const SOT3 kQ2(SO3::Expmap((Vector3() << -0.07, 0.03, -0.09).finished()), 0.95);
const SOT3 kQ3(SO3::Expmap((Vector3() << 0.05, 0.11, -0.04).finished()), 1.1);
const SOT3 kQ4(SO3::Expmap((Vector3() << -0.04, -0.08, 0.1).finished()), 1.05);

DynamicPower MakeDynamic12() { return DynamicPower({kQ1, kQ2}); }
DynamicPower MakeDynamic34() { return DynamicPower({kQ3, kQ4}); }

}  // namespace

//******************************************************************************
TEST(PowerLieGroup, Concept) {
  GTSAM_CONCEPT_ASSERT(IsGroup<FixedPower>);
  GTSAM_CONCEPT_ASSERT(IsManifold<FixedPower>);
  GTSAM_CONCEPT_ASSERT(IsLieGroup<FixedPower>);

  GTSAM_CONCEPT_ASSERT(IsGroup<DynamicPower>);
  GTSAM_CONCEPT_ASSERT(IsManifold<DynamicPower>);
  GTSAM_CONCEPT_ASSERT(IsLieGroup<DynamicPower>);
}

//******************************************************************************
TEST(PowerLieGroup, FixedRegression) {
  const FixedPower g1({kR1, kR2, kR3});
  const FixedPower g2({kR4, kR5, kR6});
  const FixedPower expected({kR1.compose(kR4), kR2.compose(kR5), kR3.compose(kR6)});

  EXPECT(assert_equal(expected, g1.compose(g2)));
  EXPECT(assert_equal(g1.inverse() * g2, g1.between(g2)));

  Matrix expectedAdj = Matrix::Zero(9, 9);
  expectedAdj.block(0, 0, 3, 3) = kR1.AdjointMap();
  expectedAdj.block(3, 3, 3, 3) = kR2.AdjointMap();
  expectedAdj.block(6, 6, 3, 3) = kR3.AdjointMap();
  EXPECT(assert_equal(expectedAdj, Matrix(g1.AdjointMap()), 1e-9));
}

//******************************************************************************
TEST(PowerLieGroup, FixedDerivatives) {
  const FixedPower id;
  const FixedPower g1({kR1, kR2, kR3});
  const FixedPower g2({kR4, kR5, kR6});

  CHECK_LIE_GROUP_DERIVATIVES(id, id);
  CHECK_LIE_GROUP_DERIVATIVES(id, g1);
  CHECK_LIE_GROUP_DERIVATIVES(g1, id);
  CHECK_LIE_GROUP_DERIVATIVES(g1, g2);

  CHECK_CHART_DERIVATIVES(id, id);
  CHECK_CHART_DERIVATIVES(id, g1);
  CHECK_CHART_DERIVATIVES(g1, id);
  CHECK_CHART_DERIVATIVES(g1, g2);
}

//******************************************************************************
TEST(PowerLieGroup, DynamicConstructors) {
  const DynamicPower empty;
  EXPECT_LONGS_EQUAL(0, empty.size());
  EXPECT_LONGS_EQUAL(0, empty.dim());
  EXPECT_LONGS_EQUAL(0, empty.AdjointMap().rows());
  EXPECT_LONGS_EQUAL(0, empty.AdjointMap().cols());

  const DynamicPower identity = DynamicPower::Identity(3);
  EXPECT_LONGS_EQUAL(3, identity.size());
  EXPECT_LONGS_EQUAL(12, identity.dim());

  const DynamicPower explicitElements({kQ1, kQ2});
  EXPECT_LONGS_EQUAL(2, explicitElements.size());
  EXPECT_LONGS_EQUAL(8, explicitElements.dim());
}

//******************************************************************************
TEST(PowerLieGroup, DynamicRegression) {
  const DynamicPower g1 = MakeDynamic12();
  const DynamicPower g2 = MakeDynamic34();

  EXPECT(assert_equal(kQ1 * kQ3, g1.compose(g2)[0]));
  EXPECT(assert_equal(kQ2 * kQ4, g1.compose(g2)[1]));
  EXPECT(assert_equal(g1.inverse() * g2, g1.between(g2)));

  const Vector xi =
      (Vector(8) << 0.02, -0.03, 0.04, 0.01, -0.02, 0.05, -0.01, 0.03)
          .finished();
  const DynamicPower roundTrip = DynamicPower::Expmap(xi);
  EXPECT(assert_equal(xi, DynamicPower::Logmap(roundTrip), 1e-9));

  Matrix expectedAdj = Matrix::Zero(8, 8);
  expectedAdj.block(0, 0, 4, 4) = g1[0].AdjointMap();
  expectedAdj.block(4, 4, 4, 4) = g1[1].AdjointMap();
  EXPECT(assert_equal(expectedAdj, g1.AdjointMap(), 1e-9));

  CHECK_EXCEPTION(g1.compose(DynamicPower::Identity(1)), std::invalid_argument);
}

//******************************************************************************
TEST(PowerLieGroup, DynamicDerivatives) {
  const DynamicPower id = DynamicPower::Identity(2);
  const DynamicPower g1 = MakeDynamic12();
  const DynamicPower g2 = MakeDynamic34();

  testLieGroupDerivativesN<8>(result_, name_, id, id);
  testLieGroupDerivativesN<8>(result_, name_, id, g1);
  testLieGroupDerivativesN<8>(result_, name_, g1, id);
  testLieGroupDerivativesN<8>(result_, name_, g1, g2);

  testChartDerivativesN<8>(result_, name_, id, id);
  testChartDerivativesN<8>(result_, name_, id, g1);
  testChartDerivativesN<8>(result_, name_, g1, id);
  testChartDerivativesN<8>(result_, name_, g1, g2);
}

int main() {
  TestResult tr;
  return TestRegistry::runAllTests(tr);
}
