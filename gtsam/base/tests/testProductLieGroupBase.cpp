/* ----------------------------------------------------------------------------

 * GTSAM Copyright 2010, Georgia Tech Research Corporation,
 * Atlanta, Georgia 30332-0415
 * All Rights Reserved
 * Authors: Frank Dellaert, et al. (see THANKS for the full author list)

 * See LICENSE for the license information

 * -------------------------------------------------------------------------- */

/**
 * @file   testProductLieGroup.cpp
 * @brief  Unit tests for ProductLieGroup
 */

#include <CppUnitLite/TestHarness.h>
#include <gtsam/base/ProductLieGroup.h>
#include <gtsam/base/TestableAssertions.h>
#include <gtsam/base/numericalDerivative.h>
#include <gtsam/base/testLie.h>
#include <gtsam/geometry/Pose3.h>
#include <gtsam/geometry/SOT3.h>

#include <functional>

using namespace gtsam;

using FixedProduct = ProductLieGroup<Pose3, Vector3>;
using DynamicPower = PowerLieGroup<SOT3, Eigen::Dynamic>;
using DynamicProduct = ProductLieGroup<Pose3, DynamicPower>;

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

const Pose3 kPose1(Rot3::RzRyRx(0.1, -0.2, 0.3), Point3(1.0, -2.0, 0.5));
const Pose3 kPose2(Rot3::RzRyRx(-0.4, 0.1, -0.2), Point3(-0.7, 1.5, 0.9));
const Vector3 kVel1(0.2, -0.4, 0.6);
const Vector3 kVel2(-0.1, 0.5, -0.3);

const SOT3 kQ1(SO3::Expmap((Vector3() << 0.2, -0.1, 0.05).finished()), 1.2);
const SOT3 kQ2(SO3::Expmap((Vector3() << -0.15, 0.25, 0.1).finished()), 0.9);
const SOT3 kQ3(SO3::Expmap((Vector3() << 0.05, 0.07, -0.2).finished()), 1.1);
const SOT3 kQ4(SO3::Expmap((Vector3() << -0.08, -0.12, 0.18).finished()),
               1.05);

DynamicPower MakePower12() { return DynamicPower({kQ1, kQ2}); }
DynamicPower MakePower34() { return DynamicPower({kQ3, kQ4}); }

}  // namespace

//******************************************************************************
TEST(ProductLieGroup, Concept) {
  GTSAM_CONCEPT_ASSERT(IsGroup<FixedProduct>);
  GTSAM_CONCEPT_ASSERT(IsManifold<FixedProduct>);
  GTSAM_CONCEPT_ASSERT(IsLieGroup<FixedProduct>);

  GTSAM_CONCEPT_ASSERT(IsGroup<DynamicProduct>);
  GTSAM_CONCEPT_ASSERT(IsManifold<DynamicProduct>);
  GTSAM_CONCEPT_ASSERT(IsLieGroup<DynamicProduct>);
}

//******************************************************************************
TEST(ProductLieGroup, FixedRegression) {
  const FixedProduct g1(kPose1, kVel1), g2(kPose2, kVel2);
  const FixedProduct expected(kPose1.compose(kPose2), kVel1 + kVel2);

  EXPECT(assert_equal(expected, g1.compose(g2)));
  EXPECT(assert_equal(g1.inverse() * g2, g1.between(g2)));
  EXPECT(assert_equal(FixedProduct::Logmap(g1),
                      FixedProduct::Identity().localCoordinates(g1)));

  Matrix expectedAdj = Matrix::Zero(9, 9);
  expectedAdj.block(0, 0, 6, 6) = kPose1.AdjointMap();
  expectedAdj.block(6, 6, 3, 3) = Matrix3::Identity();
  EXPECT(assert_equal(expectedAdj, Matrix(g1.AdjointMap()), 1e-9));
}

//******************************************************************************
TEST(ProductLieGroup, FixedDerivatives) {
  const FixedProduct id;
  const FixedProduct g1(kPose1, kVel1), g2(kPose2, kVel2);

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
TEST(ProductLieGroup, DynamicFactor) {
  const DynamicProduct g1(kPose1, MakePower12());
  const DynamicProduct g2(kPose2, MakePower34());

  EXPECT_LONGS_EQUAL(14, g1.dim());
  EXPECT_LONGS_EQUAL(14, traits<DynamicProduct>::GetDimension(g1));

  const DynamicProduct composed = g1.compose(g2);
  EXPECT(assert_equal(kPose1.compose(kPose2), composed.first));
  EXPECT(assert_equal(MakePower12().compose(MakePower34()), composed.second));

  const Vector xi =
      (Vector(14) << 0.05, -0.04, 0.03, 0.1, -0.2, 0.3, 0.01, -0.02, 0.03,
       -0.04, -0.03, 0.02, -0.01, 0.05)
          .finished();
  const DynamicProduct roundTrip = DynamicProduct::Expmap(xi);
  EXPECT(assert_equal(xi, DynamicProduct::Logmap(roundTrip), 1e-9));

  Matrix expectedAdj = Matrix::Zero(14, 14);
  expectedAdj.block(0, 0, 6, 6) = g1.first.AdjointMap();
  expectedAdj.block(6, 6, 8, 8) = g1.second.AdjointMap();
  EXPECT(assert_equal(expectedAdj, g1.AdjointMap(), 1e-9));
}

//******************************************************************************
TEST(ProductLieGroup, DynamicDerivatives) {
  const DynamicProduct id(Pose3::Identity(), DynamicPower::Identity(2));
  const DynamicProduct g1(kPose1, MakePower12());
  const DynamicProduct g2(kPose2, MakePower34());

  testLieGroupDerivativesN<14>(result_, name_, id, id);
  testLieGroupDerivativesN<14>(result_, name_, id, g1);
  testLieGroupDerivativesN<14>(result_, name_, g1, id);
  testLieGroupDerivativesN<14>(result_, name_, g1, g2);

  testChartDerivativesN<14>(result_, name_, id, id);
  testChartDerivativesN<14>(result_, name_, id, g1);
  testChartDerivativesN<14>(result_, name_, g1, id);
  testChartDerivativesN<14>(result_, name_, g1, g2);
}

int main() {
  TestResult tr;
  return TestRegistry::runAllTests(tr);
}
