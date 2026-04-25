#include <gtest/gtest.h>

#include <PerfectHash/PerfectHashOnlineJit.h>
#include <PerfectHash/PerfectHash.h>

#include <cstdint>
#include <cstdlib>
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

  //
  // Mulshrolate3RX is in the curated good hash set, so CreateTable64
  // intentionally selects GraphImpl4 for the compact-key JIT backend.
  //

  ASSERT_GE(PhOnlineJitCreateTable64(context.get(),
                                     PhOnlineJitHashMulshrolate3RX,
                                     keys.data(),
                                     static_cast<uint64_t>(keys.size()),
                                     &raw_table),
            0);
  ASSERT_NE(raw_table, nullptr);
  table.reset(raw_table);

  std::vector<uint32_t> fallback_indexes;
  fallback_indexes.reserve(keys.size());
  for (auto key : keys) {
    uint32_t index = 0;
    ASSERT_GE(PhOnlineJitIndex64(table.get(), key, &index), 0);
    fallback_indexes.push_back(index);
  }

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

  for (size_t i = 0; i < keys.size(); ++i) {
    uint32_t index = 0;
    ASSERT_GE(PhOnlineJitIndex64(table.get(), keys[i], &index), 0);
    EXPECT_EQ(fallback_indexes[i], index);
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
  ASSERT_GE(result, 0);

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

TEST(PerfectHashOnlineJitTests, Index64OnNonGraphImpl64BitTableUsesMetadata) {
  auto context = std::unique_ptr<PH_ONLINE_JIT_CONTEXT, decltype(&PhOnlineJitClose)>(
      nullptr, &PhOnlineJitClose);
  auto table = std::unique_ptr<PH_ONLINE_JIT_TABLE, decltype(&PhOnlineJitReleaseTable)>(
      nullptr, &PhOnlineJitReleaseTable);
  PH_ONLINE_JIT_CONTEXT *raw_context = nullptr;
  PH_ONLINE_JIT_TABLE *raw_table = nullptr;
  auto keys = MakeDownsized64Keys(64);

  ASSERT_GE(PhOnlineJitOpen(&raw_context), 0);
  context.reset(raw_context);
  ASSERT_GE(PhOnlineJitCreateTable64(context.get(),
                                     PhOnlineJitHashMultiplyShiftLR,
                                     keys.data(),
                                     static_cast<uint64_t>(keys.size()),
                                     &raw_table),
            0);
  table.reset(raw_table);

  std::unordered_set<uint32_t> seen;
  seen.reserve(keys.size());
  for (auto key : keys) {
    uint32_t index = 0;
    ASSERT_GE(PhOnlineJitIndex64(table.get(), key, &index), 0);
    ASSERT_TRUE(seen.insert(index).second);
  }
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
