#include <gtest/gtest.h>

// Example of a simple test case
TEST(SimpleTest, BasicAssertions) {
    EXPECT_EQ(1 + 1, 2);
    EXPECT_TRUE(true);
}

// Example of a parameterized test case
class ParameterizedTest : public ::testing::TestWithParam<int> {};

TEST_P(ParameterizedTest, IsEven) {
    int n = GetParam();
    EXPECT_EQ(n % 2, 0);
}

INSTANTIATE_TEST_SUITE_P(EvenNumbers, ParameterizedTest, ::testing::Values(2, 4, 6, 8, 10));

// Example of a templated test case
template <typename T>
class TemplatedTest : public ::testing::Test {};

typedef ::testing::Types<int, float, double> MyTypes;
TYPED_TEST_SUITE(TemplatedTest, MyTypes);

TYPED_TEST(TemplatedTest, IsPositive) {
    TypeParam n = static_cast<TypeParam>(5);
    EXPECT_GT(n, 0);
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}