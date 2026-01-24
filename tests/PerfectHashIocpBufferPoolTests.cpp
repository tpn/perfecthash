#include <gtest/gtest.h>

#include <cstring>

#define _PERFECT_HASH_INTERNAL_BUILD
#include "PerfectHash/stdafx.h"
#include "PerfectHash/PerfectHashIocpBufferPool.h"

namespace {

void NTAPI
TestRtlFillMemory(
    PVOID Destination,
    ULONG_PTR Length,
    BYTE Fill
    )
{
  std::memset(Destination, Fill, static_cast<size_t>(Length));
}

void NTAPI
TestRtlZeroMemory(
    PVOID Destination,
    ULONG_PTR Length
    )
{
  std::memset(Destination, 0, static_cast<size_t>(Length));
}

class PerfectHashIocpBufferPoolTests : public ::testing::Test {
protected:
  void SetUp() override {
    std::memset(&rtl_, 0, sizeof(rtl_));
    std::memset(&vtbl_, 0, sizeof(vtbl_));
    vtbl_.CreateMultipleBuffers = RtlCreateMultipleBuffers;
    vtbl_.DestroyBuffer = RtlDestroyBuffer;
    rtl_.Vtbl = &vtbl_;
    rtl_.RtlFillMemory = TestRtlFillMemory;
    rtl_.RtlZeroMemory = TestRtlZeroMemory;
  }

  void TearDown() override {
    rtl_.Vtbl = nullptr;
  }

  RTL rtl_;
  RTL_VTBL vtbl_;
};

TEST_F(PerfectHashIocpBufferPoolTests, CreatePopPush) {
  PERFECT_HASH_IOCP_BUFFER_POOL pool;
  std::memset(&pool, 0, sizeof(pool));
  EXPECT_EQ(reinterpret_cast<ULONG_PTR>(&pool) & 0xF, 0u);
  EXPECT_EQ(reinterpret_cast<ULONG_PTR>(&pool.ListHead) & 0xF, 0u);

  HRESULT result = PerfectHashIocpBufferPoolCreate(
      &rtl_,
      nullptr,
      4096,
      2,
      1,
      nullptr,
      nullptr,
      &pool);

  ASSERT_GE(result, 0);
  EXPECT_GT(pool.PayloadSizeInBytes, 0u);
  EXPECT_GT(pool.PayloadOffset, 0u);
  EXPECT_EQ(pool.PayloadSizeInBytes + pool.PayloadOffset,
            pool.UsableBufferSizeInBytes);

  PPERFECT_HASH_IOCP_BUFFER buffer = PerfectHashIocpBufferPoolPop(&pool);
  ASSERT_NE(buffer, nullptr);
  EXPECT_EQ(buffer->PayloadSize, pool.PayloadSizeInBytes);
  EXPECT_EQ(buffer->PayloadOffset, pool.PayloadOffset);

  PVOID payload = PerfectHashIocpBufferPayload(buffer);
  EXPECT_EQ(reinterpret_cast<ULONG_PTR>(payload) -
                reinterpret_cast<ULONG_PTR>(buffer),
            pool.PayloadOffset);

  PerfectHashIocpBufferPoolPush(&pool, buffer);
  PerfectHashIocpBufferPoolDestroy(&rtl_, &pool);
}

TEST_F(PerfectHashIocpBufferPoolTests, PopAllReturnsNull) {
  PERFECT_HASH_IOCP_BUFFER_POOL pool;
  std::memset(&pool, 0, sizeof(pool));
  EXPECT_EQ(reinterpret_cast<ULONG_PTR>(&pool) & 0xF, 0u);
  EXPECT_EQ(reinterpret_cast<ULONG_PTR>(&pool.ListHead) & 0xF, 0u);

  HRESULT result = PerfectHashIocpBufferPoolCreate(
      &rtl_,
      nullptr,
      4096,
      1,
      1,
      nullptr,
      nullptr,
      &pool);

  ASSERT_GE(result, 0);

  PPERFECT_HASH_IOCP_BUFFER first = PerfectHashIocpBufferPoolPop(&pool);
  ASSERT_NE(first, nullptr);
  EXPECT_EQ(PerfectHashIocpBufferPoolPop(&pool), nullptr);

  PerfectHashIocpBufferPoolPush(&pool, first);
  EXPECT_NE(PerfectHashIocpBufferPoolPop(&pool), nullptr);

  PerfectHashIocpBufferPoolDestroy(&rtl_, &pool);
}

} // namespace
