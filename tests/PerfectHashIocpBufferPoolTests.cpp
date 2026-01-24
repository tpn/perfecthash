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

ULONGLONG
TestRoundUpPowerOfTwo64(
    ULONGLONG Value
    )
{
  if (Value == 0) {
    return 0;
  }

  Value--;
  Value |= Value >> 1;
  Value |= Value >> 2;
  Value |= Value >> 4;
  Value |= Value >> 8;
  Value |= Value >> 16;
  Value |= Value >> 32;
  Value++;

  return Value;
}

ULONGLONG
TestTrailingZeros64(
    ULONGLONG Value
    )
{
  ULONGLONG Count = 0;

  if (Value == 0) {
    return 64;
  }

  while ((Value & 1) == 0) {
    Count++;
    Value >>= 1;
  }

  return Count;
}

class PerfectHashIocpBufferPoolTests : public ::testing::Test {
protected:
  void SetUp() override {
    std::memset(&rtl_, 0, sizeof(rtl_));
    std::memset(&vtbl_, 0, sizeof(vtbl_));
    vtbl_.CreateBuffer = RtlCreateBuffer;
    vtbl_.DestroyBuffer = RtlDestroyBuffer;
    rtl_.Vtbl = &vtbl_;
    rtl_.RtlFillMemory = TestRtlFillMemory;
    rtl_.RtlZeroMemory = TestRtlZeroMemory;
    rtl_.RoundUpPowerOfTwo64 = TestRoundUpPowerOfTwo64;
    rtl_.TrailingZeros64 = TestTrailingZeros64;
  }

  void TearDown() override {
    rtl_.Vtbl = nullptr;
  }

  RTL rtl_;
  RTL_VTBL vtbl_;
};

TEST_F(PerfectHashIocpBufferPoolTests, InitializeAcquireRelease) {
  PERFECT_HASH_IOCP_BUFFER_POOL pool;
  std::memset(&pool, 0, sizeof(pool));
  EXPECT_EQ(reinterpret_cast<ULONG_PTR>(&pool) & 0xF, 0u);

  HRESULT result = PerfectHashIocpBufferPoolInitialize(
      &rtl_,
      &pool,
      4096,
      0,
      0,
      nullptr,
      nullptr);

  ASSERT_GE(result, 0);
  EXPECT_GT(pool.PayloadSize, 0u);
  EXPECT_GT(pool.AllocationSize, 0u);

  PPERFECT_HASH_IOCP_BUFFER buffer = nullptr;
  result = PerfectHashIocpBufferPoolAcquire(&rtl_,
                                            nullptr,
                                            &pool,
                                            &buffer);
  ASSERT_GE(result, 0);
  ASSERT_NE(buffer, nullptr);
  EXPECT_EQ(buffer->PayloadSize, pool.PayloadSize);
  EXPECT_EQ(buffer->PayloadOffset, PERFECT_HASH_IOCP_BUFFER_HEADER_SIZE);

  PVOID payload = PerfectHashIocpBufferPayload(buffer);
  EXPECT_EQ(reinterpret_cast<ULONG_PTR>(payload) -
                reinterpret_cast<ULONG_PTR>(buffer),
            buffer->PayloadOffset);

  PerfectHashIocpBufferPoolRelease(&pool, buffer);

  PPERFECT_HASH_IOCP_BUFFER second = nullptr;
  result = PerfectHashIocpBufferPoolAcquire(&rtl_,
                                            nullptr,
                                            &pool,
                                            &second);
  ASSERT_GE(result, 0);
  ASSERT_NE(second, nullptr);
  EXPECT_EQ(second, buffer);
  PerfectHashIocpBufferPoolRelease(&pool, second);

  PerfectHashIocpBufferPoolRundown(&rtl_, nullptr, &pool);
}

TEST_F(PerfectHashIocpBufferPoolTests, SizeClassHelpers) {
  EXPECT_EQ(PerfectHashIocpBufferGetClassIndex(&rtl_, 0), 0);
  EXPECT_EQ(PerfectHashIocpBufferGetClassIndex(&rtl_, 1), 0);
  EXPECT_EQ(PerfectHashIocpBufferGetClassIndex(&rtl_, 4096), 0);
  EXPECT_EQ(PerfectHashIocpBufferGetClassIndex(&rtl_, 4097), 1);
  EXPECT_EQ(PerfectHashIocpBufferGetClassIndex(&rtl_, 8192), 1);
  EXPECT_EQ(PerfectHashIocpBufferGetClassIndex(
                &rtl_,
                PERFECT_HASH_IOCP_BUFFER_MAX_SIZE),
            PERFECT_HASH_IOCP_BUFFER_CLASS_COUNT - 1);
  EXPECT_EQ(PerfectHashIocpBufferGetClassIndex(
                &rtl_,
                PERFECT_HASH_IOCP_BUFFER_MAX_SIZE + 1),
            -1);

  EXPECT_EQ(PerfectHashIocpBufferGetPayloadSizeFromClassIndex(0), 4096u);
  EXPECT_EQ(PerfectHashIocpBufferGetPayloadSizeFromClassIndex(1), 8192u);
}

} // namespace
