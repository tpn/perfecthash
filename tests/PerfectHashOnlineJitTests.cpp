#include <gtest/gtest.h>

#include <PerfectHash/PerfectHashOnlineJit.h>
#include <PerfectHash/PerfectHash.h>

#include <array>
#include <cstdint>
#include <cstdlib>
#include <future>
#include <memory>
#include <string>
#include <unordered_set>
#include <vector>

namespace {

#ifndef PERFECTHASH_TEST_BUILD_LIB_DIR
#define PERFECTHASH_TEST_BUILD_LIB_DIR ""
#endif

class ScopedEnvVar {
public:
  ScopedEnvVar(const char *name, const char *value) : name_(name) {
    const char *existing = std::getenv(name);
    if (existing) {
      had_old_ = true;
      old_ = existing;
    }
    Set(value);
  }

  ~ScopedEnvVar() {
    if (had_old_) {
      Set(old_.c_str());
    } else {
      Unset();
    }
  }

private:
  void Set(const char *value) {
#ifdef PH_WINDOWS
    _putenv_s(name_.c_str(), value ? value : "");
#else
    if (value) {
      setenv(name_.c_str(), value, 1);
    } else {
      unsetenv(name_.c_str());
    }
#endif
  }

  void Unset() {
#ifdef PH_WINDOWS
    _putenv_s(name_.c_str(), "");
#else
    unsetenv(name_.c_str());
#endif
  }

  std::string name_;
  bool had_old_ = false;
  std::string old_;
};

std::vector<uint64_t> MakeDownsized64Keys(size_t count) {
  std::vector<uint64_t> keys;
  keys.reserve(count);
  for (uint64_t i = 0; i < count; ++i) {
    keys.push_back((1ULL << 40) | (i << 20));
  }
  return keys;
}

uint64_t Mix64(uint64_t value) {
  value ^= value >> 30;
  value *= 0xbf58476d1ce4e5b9ULL;
  value ^= value >> 27;
  value *= 0x94d049bb133111ebULL;
  value ^= value >> 31;
  return value;
}

std::vector<uint64_t> MakeNonDownsized64Keys(size_t count) {
  std::vector<uint64_t> keys;
  keys.reserve(count);
  for (uint64_t i = 0; i < count; ++i) {
    keys.push_back(Mix64(i + 1));
  }
  return keys;
}

}  // namespace

#if (defined(PH_HAS_RAWDOG_JIT) || defined(PH_HAS_LLVM)) && \
    (defined(_M_ARM64) || defined(__aarch64__))
static void VerifyArm64BatchWidths(PH_ONLINE_JIT_BACKEND requested) {
  ScopedEnvVar concurrency_cap("PERFECT_HASH_ONLINE_MAX_CONCURRENCY", "4");
  const std::array<uint32_t, 16> keys = {
      1, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53};
  PH_ONLINE_JIT_CONTEXT *raw_context = nullptr;
  ASSERT_GE(PhOnlineJitOpen(&raw_context), 0);
  std::unique_ptr<PH_ONLINE_JIT_CONTEXT, decltype(&PhOnlineJitClose)> context(
      raw_context, &PhOnlineJitClose);

  for (auto hash : {PhOnlineJitHashMultiplyShiftR, PhOnlineJitHashMultiplyShiftRX,
                    PhOnlineJitHashMulshrolate1RX, PhOnlineJitHashMulshrolate2RX,
                    PhOnlineJitHashMulshrolate3RX}) {
    SCOPED_TRACE(hash);
    PH_ONLINE_JIT_TABLE *raw_table = nullptr;
    ASSERT_GE(PhOnlineJitCreateTable32(context.get(), hash, keys.data(),
                                       keys.size(), &raw_table), 0);
    std::unique_ptr<PH_ONLINE_JIT_TABLE, decltype(&PhOnlineJitReleaseTable)>
        table(raw_table, &PhOnlineJitReleaseTable);
    std::array<uint32_t, 16> expected;
    for (size_t i = 0; i < keys.size(); ++i) {
      ASSERT_GE(PhOnlineJitIndex32(table.get(), keys[i], &expected[i]), 0);
    }

    for (uint32_t requested_width : {0u, 1u, 4u, 8u, 16u}) {
      SCOPED_TRACE(requested_width);
      PH_ONLINE_JIT_BACKEND backend = PhOnlineJitBackendAuto;
      uint32_t width = 0;
      ASSERT_GE(PhOnlineJitCompileTableEx(
                    context.get(), table.get(), requested,
                    requested_width, PhOnlineJitMaxIsaAuto,
                    PH_ONLINE_JIT_COMPILE_FLAG_STRICT_VECTOR_WIDTH,
                    &backend, &width), 0);
      ASSERT_EQ(backend, requested);
      ASSERT_EQ(width, requested_width ? requested_width : 1u);

      std::array<uint32_t, 16> actual = {};
      for (size_t i = 0; i < keys.size(); i += width) {
        if (width == 1) {
          ASSERT_GE(PhOnlineJitIndex32(table.get(), keys[i], &actual[i]), 0);
        } else {
          const auto index = width == 4 ? PhOnlineJitIndex32x4 :
                             width == 8 ? PhOnlineJitIndex32x8 : PhOnlineJitIndex32x16;
          ASSERT_GE(index(table.get(), &keys[i], &actual[i]), 0);
        }
      }
      EXPECT_EQ(actual, expected);
    }
  }
}
#ifdef PH_HAS_RAWDOG_JIT
TEST(PerfectHashOnlineJitTests, Arm64RawDogStrictWidthsMatchScalar) {
  VerifyArm64BatchWidths(PhOnlineJitBackendRawDogJit);
}
#endif
#ifdef PH_HAS_LLVM
TEST(PerfectHashOnlineJitTests, Arm64LlvmStrictWidthsMatchScalar) {
  VerifyArm64BatchWidths(PhOnlineJitBackendLlvmJit);
}
#endif
#endif

#if defined(PH_HAS_LLVM) && defined(PH_HAS_RAWDOG_JIT)
TEST(PerfectHashOnlineJitTests, SwitchBackendsAndReportCompiledWidth) {
  ScopedEnvVar concurrency_cap("PERFECT_HASH_ONLINE_MAX_CONCURRENCY", "4");
  std::array<uint32_t, 16> keys = {
      1, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53};
  PH_ONLINE_JIT_CONTEXT *raw_context = nullptr;
  ASSERT_GE(PhOnlineJitOpen(&raw_context), 0);
  std::unique_ptr<PH_ONLINE_JIT_CONTEXT, decltype(&PhOnlineJitClose)> context(
      raw_context, &PhOnlineJitClose);
  PH_ONLINE_JIT_TABLE *raw_table = nullptr;
  ASSERT_GE(PhOnlineJitCreateTable32(context.get(), PhOnlineJitHashMultiplyShiftR,
                                     keys.data(), keys.size(), &raw_table), 0);
  std::unique_ptr<PH_ONLINE_JIT_TABLE, decltype(&PhOnlineJitReleaseTable)>
      table(raw_table, &PhOnlineJitReleaseTable);
  std::array<uint32_t, 16> expected;
  for (size_t i = 0; i < keys.size(); ++i) {
    ASSERT_GE(PhOnlineJitIndex32(table.get(), keys[i], &expected[i]), 0);
  }

  PH_ONLINE_JIT_BACKEND effective_backend = PhOnlineJitBackendAuto;
  uint32_t effective_width = 0;
  auto compile = [&](PH_ONLINE_JIT_BACKEND backend, uint32_t width,
                     PH_ONLINE_JIT_MAX_ISA isa, uint32_t flags) {
    return PhOnlineJitCompileTableEx(context.get(), table.get(), backend, width,
                                     isa, flags, &effective_backend, &effective_width);
  };
  for (auto backend : {PhOnlineJitBackendRawDogJit, PhOnlineJitBackendLlvmJit,
                       PhOnlineJitBackendRawDogJit}) {
    ASSERT_GE(compile(backend, 1, PhOnlineJitMaxIsaAuto,
                      PH_ONLINE_JIT_COMPILE_FLAG_STRICT_VECTOR_WIDTH), 0);
    ASSERT_EQ(effective_backend, backend);
    ASSERT_EQ(effective_width, 1u);
    for (size_t i = 0; i < keys.size(); ++i) {
      uint32_t index = 0;
      ASSERT_GE(PhOnlineJitIndex32(table.get(), keys[i], &index), 0);
      EXPECT_EQ(index, expected[i]);
    }
  }

  const auto strict = PH_ONLINE_JIT_COMPILE_FLAG_STRICT_VECTOR_WIDTH;
  ASSERT_GE(compile(PhOnlineJitBackendLlvmJit, 8, PhOnlineJitMaxIsaAuto, strict), 0);
  auto check_x8 = [&]() {
    std::array<uint32_t, 16> actual = {};
    for (size_t i = 0; i < keys.size(); i += 8) {
      ASSERT_GE(PhOnlineJitIndex32x8(table.get(), &keys[i], &actual[i]), 0);
    }
    EXPECT_EQ(actual, expected);
  };

  // An unsupported request for either backend must keep the existing
  // x8 routine and its executable pages available.
  ASSERT_EQ(compile(PhOnlineJitBackendRawDogJit, 2, PhOnlineJitMaxIsaAuto,
                    strict), PH_E_NOT_IMPLEMENTED);
  check_x8();
  ASSERT_EQ(compile(PhOnlineJitBackendLlvmJit, 16, PhOnlineJitMaxIsaAvx2,
                    strict), PH_E_NOT_IMPLEMENTED);
  check_x8();
#if defined(_M_ARM64) || defined(__aarch64__)
  ASSERT_GE(compile(PhOnlineJitBackendRawDogJit, 8, PhOnlineJitMaxIsaAuto, strict), 0);
  ASSERT_EQ(compile(PhOnlineJitBackendLlvmJit, 16, PhOnlineJitMaxIsaAvx2,
                    strict), PH_E_NOT_IMPLEMENTED);
  check_x8();
#endif
  ASSERT_GE(compile(PhOnlineJitBackendLlvmJit, 16, PhOnlineJitMaxIsaAvx2, 0), 0);
  ASSERT_EQ(effective_backend, PhOnlineJitBackendLlvmJit);
  ASSERT_EQ(effective_width, 8u);
  check_x8();

  // Concurrent compiles either complete or return PH_E_TABLE_LOCKED. Query
  // the final x8 binding once both compilers have finished.
  auto recompile = [&]() {
    for (int pass = 0; pass < 8; ++pass) {
      auto result = PhOnlineJitCompileTableEx(
          context.get(), table.get(), PhOnlineJitBackendLlvmJit, 8,
          PhOnlineJitMaxIsaAuto, strict, nullptr, nullptr);
      if (result < 0 && result != PH_E_TABLE_LOCKED) {
        return false;
      }
    }
    return true;
  };
  auto first = std::async(std::launch::async, recompile);
  auto second = std::async(std::launch::async, recompile);
  EXPECT_TRUE(first.get());
  EXPECT_TRUE(second.get());
  check_x8();
}
#endif

TEST(PerfectHashOnlineJitTests, CreateTable64AndIndex64) {
  ScopedEnvVar concurrency_cap("PERFECT_HASH_ONLINE_MAX_CONCURRENCY", "4");
  ScopedEnvVar library_path("LD_LIBRARY_PATH", PERFECTHASH_TEST_BUILD_LIB_DIR);

  auto keys = MakeDownsized64Keys(256);
  auto context = std::unique_ptr<PH_ONLINE_JIT_CONTEXT, decltype(&PhOnlineJitClose)>(
      nullptr, &PhOnlineJitClose);
  auto table = std::unique_ptr<PH_ONLINE_JIT_TABLE, decltype(&PhOnlineJitReleaseTable)>(
      nullptr, &PhOnlineJitReleaseTable);
  PH_ONLINE_JIT_CONTEXT *raw_context = nullptr;
  PH_ONLINE_JIT_TABLE *raw_table = nullptr;

  ASSERT_GE(PhOnlineJitOpen(&raw_context), 0);
  ASSERT_NE(raw_context, nullptr);
  context.reset(raw_context);

  ASSERT_GE(PhOnlineJitCreateTable64(context.get(),
                                     PhOnlineJitHashMulshrolate3RX,
                                     keys.data(),
                                     static_cast<uint64_t>(keys.size()),
                                     &raw_table),
            0);
  ASSERT_NE(raw_table, nullptr);
  table.reset(raw_table);

  auto result = PhOnlineJitCompileTableEx(context.get(),
                                          table.get(),
                                          PhOnlineJitBackendRawDogJit,
                                          1,
                                          PhOnlineJitMaxIsaAuto,
                                          0,
                                          nullptr,
                                          nullptr);
  if (result < 0) {
    result = PhOnlineJitCompileTableEx(context.get(),
                                       table.get(),
                                       PhOnlineJitBackendLlvmJit,
                                       1,
                                       PhOnlineJitMaxIsaAuto,
                                       0,
                                       nullptr,
                                       nullptr);
  }

  std::unordered_set<uint32_t> seen;
  seen.reserve(keys.size());

  for (auto key : keys) {
    uint32_t index = 0;
    ASSERT_GE(PhOnlineJitIndex64(table.get(), key, &index), 0);
    ASSERT_TRUE(seen.insert(index).second);
  }
}

TEST(PerfectHashOnlineJitTests, CreateTable32AndIndex32) {
  auto context = std::unique_ptr<PH_ONLINE_JIT_CONTEXT, decltype(&PhOnlineJitClose)>(
      nullptr, &PhOnlineJitClose);
  auto table = std::unique_ptr<PH_ONLINE_JIT_TABLE, decltype(&PhOnlineJitReleaseTable)>(
      nullptr, &PhOnlineJitReleaseTable);
  PH_ONLINE_JIT_CONTEXT *raw_context = nullptr;
  PH_ONLINE_JIT_TABLE *raw_table = nullptr;
  std::vector<uint32_t> keys = {1, 3, 5, 7, 11, 13, 17, 19};

  ASSERT_GE(PhOnlineJitOpen(&raw_context), 0);
  ASSERT_NE(raw_context, nullptr);
  context.reset(raw_context);

  ASSERT_GE(PhOnlineJitCreateTable32(context.get(),
                                     PhOnlineJitHashMulshrolate3RX,
                                     keys.data(),
                                     static_cast<uint64_t>(keys.size()),
                                     &raw_table),
            0);
  ASSERT_NE(raw_table, nullptr);
  table.reset(raw_table);

  auto result = PhOnlineJitCompileTableEx(context.get(),
                                          table.get(),
                                          PhOnlineJitBackendRawDogJit,
                                          1,
                                          PhOnlineJitMaxIsaAuto,
                                          0,
                                          nullptr,
                                          nullptr);
  if (result < 0) {
    result = PhOnlineJitCompileTableEx(context.get(),
                                       table.get(),
                                       PhOnlineJitBackendLlvmJit,
                                       1,
                                       PhOnlineJitMaxIsaAuto,
                                       0,
                                       nullptr,
                                       nullptr);
  }
#if defined(PH_HAS_RAWDOG_JIT) || defined(PH_HAS_LLVM)
  ASSERT_GE(result, 0);
#else
  // With no JIT backend, retain coverage of the scalar index fallback.
  ASSERT_TRUE(result >= 0 || result == PH_E_LLVM_BACKEND_NOT_FOUND ||
              result == PH_E_NOT_IMPLEMENTED);
#endif

  std::unordered_set<uint32_t> seen;
  for (auto key : keys) {
    uint32_t index = 0;
    ASSERT_GE(PhOnlineJitIndex32(table.get(), key, &index), 0);
    ASSERT_TRUE(seen.insert(index).second);
  }
}

TEST(PerfectHashOnlineJitTests, Index64On32BitTableIsNotImplemented) {
  auto context = std::unique_ptr<PH_ONLINE_JIT_CONTEXT, decltype(&PhOnlineJitClose)>(
      nullptr, &PhOnlineJitClose);
  auto table = std::unique_ptr<PH_ONLINE_JIT_TABLE, decltype(&PhOnlineJitReleaseTable)>(
      nullptr, &PhOnlineJitReleaseTable);
  PH_ONLINE_JIT_CONTEXT *raw_context = nullptr;
  PH_ONLINE_JIT_TABLE *raw_table = nullptr;
  std::vector<uint32_t> keys = {1, 3, 5, 7, 11, 13, 17, 19};
  uint32_t index = 0;

  ASSERT_GE(PhOnlineJitOpen(&raw_context), 0);
  context.reset(raw_context);
  ASSERT_GE(PhOnlineJitCreateTable32(context.get(),
                                     PhOnlineJitHashMulshrolate3RX,
                                     keys.data(),
                                     static_cast<uint64_t>(keys.size()),
                                     &raw_table),
            0);
  table.reset(raw_table);

  ASSERT_EQ(PhOnlineJitIndex64(table.get(), 1ULL, &index), PH_E_NOT_IMPLEMENTED);
}

TEST(PerfectHashOnlineJitTests, CreateTable64RejectsNonDownsizedKeys) {
  auto context = std::unique_ptr<PH_ONLINE_JIT_CONTEXT, decltype(&PhOnlineJitClose)>(
      nullptr, &PhOnlineJitClose);
  PH_ONLINE_JIT_CONTEXT *raw_context = nullptr;
  PH_ONLINE_JIT_TABLE *raw_table = nullptr;
  auto keys = MakeNonDownsized64Keys(64);

  ASSERT_GE(PhOnlineJitOpen(&raw_context), 0);
  context.reset(raw_context);

  ASSERT_EQ(PhOnlineJitCreateTable64(context.get(),
                                     PhOnlineJitHashMulshrolate3RX,
                                     keys.data(),
                                     static_cast<uint64_t>(keys.size()),
                                     &raw_table),
            PH_E_NOT_IMPLEMENTED);
  ASSERT_EQ(raw_table, nullptr);
}
