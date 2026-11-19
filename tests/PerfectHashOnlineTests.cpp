#include <gtest/gtest.h>

#include <PerfectHash/PerfectHash.h>
#include <PerfectHash/PerfectHashOnlineJit.h>
#include <PerfectHash/PerfectHashOnlineJitPrivate.h>

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <fstream>
#include <memory>
#include <string>
#include <unordered_set>
#include <vector>

#if defined(PH_HAS_RAWDOG_JIT) && (defined(_M_X64) || defined(__x86_64__))
extern "C" int32_t CompileRawDogX4WithoutAvx2(
    PH_ONLINE_JIT_CONTEXT *, PH_ONLINE_JIT_TABLE *, PH_ONLINE_JIT_MAX_ISA);
#endif

#if defined(PH_LINUX) && defined(__aarch64__)
#include <asm/hwcap.h>
#include <sys/auxv.h>
#endif

namespace {

struct PerfectHashTableShim {
  PPERFECT_HASH_TABLE_VTBL Vtbl;
};

ULONGLONG BuildDownsizeMask(const std::vector<ULONGLONG> &keys) {
  ULONGLONG mask = 0;
  for (auto key : keys) {
    mask |= key;
  }
  return mask;
}

ULONG DownsizeKey(ULONGLONG key, ULONGLONG mask) {
  ULONGLONG result = 0;
  ULONGLONG bitpos = 0;

  while (mask != 0) {
    if (mask & 1) {
      result |= (key & 1) << bitpos;
      bitpos++;
    }

    mask >>= 1;
    key >>= 1;
  }

  return static_cast<ULONG>(result);
}

std::string TrimWhitespace(std::string value) {
  auto not_space = [](unsigned char ch) { return !std::isspace(ch); };
  auto begin = std::find_if(value.begin(), value.end(), not_space);
  auto end = std::find_if(value.rbegin(), value.rend(), not_space).base();
  if (begin >= end) {
    return std::string();
  }
  return std::string(begin, end);
}

bool LoadKeysFromFile(const std::string &path, std::vector<ULONG> *keys) {
  if (!keys) {
    return false;
  }

  std::ifstream file(path, std::ios::binary | std::ios::ate);
  if (!file) {
    return false;
  }

  std::streamsize size = file.tellg();
  if (size <= 0 || (size % static_cast<std::streamsize>(sizeof(ULONG))) != 0) {
    return false;
  }

  keys->resize(static_cast<size_t>(size / sizeof(ULONG)));
  file.seekg(0, std::ios::beg);
  if (!file.read(reinterpret_cast<char *>(keys->data()), size)) {
    return false;
  }

  return true;
}

bool ParseSeedLine(const std::string &line, size_t *index, ULONG *value) {
  if (!index || !value) {
    return false;
  }

  std::string trimmed = TrimWhitespace(line);
  if (trimmed.empty() || trimmed[0] == '#') {
    return false;
  }

  auto comment = trimmed.find('#');
  if (comment != std::string::npos) {
    trimmed = TrimWhitespace(trimmed.substr(0, comment));
    if (trimmed.empty()) {
      return false;
    }
  }

  auto equals = trimmed.find('=');
  if (equals == std::string::npos) {
    return false;
  }

  std::string left = TrimWhitespace(trimmed.substr(0, equals));
  std::string right = TrimWhitespace(trimmed.substr(equals + 1));
  if (left.rfind("Seed", 0) != 0 || left.find("Seed3_") == 0) {
    return false;
  }

  std::string suffix = left.substr(4);
  if (suffix.empty() || suffix.size() > 2) {
    return false;
  }
  for (char ch : suffix) {
    if (ch < '0' || ch > '9') {
      return false;
    }
  }

  size_t parsed_index = static_cast<size_t>(std::stoul(suffix));
  if (parsed_index == 0 || parsed_index > 8) {
    return false;
  }

  char *end = nullptr;
  unsigned long parsed_value = std::strtoul(right.c_str(), &end, 0);
  if (!end || end == right.c_str() || *end != '\0') {
    return false;
  }

  *index = parsed_index;
  *value = static_cast<ULONG>(parsed_value);
  return true;
}

bool LoadSeedsFromFile(const std::string &path, std::vector<ULONG> *seeds) {
  if (!seeds) {
    return false;
  }

  std::ifstream file(path);
  if (!file) {
    return false;
  }

  std::vector<ULONG> local_seeds(8, 0);
  size_t max_index = 0;
  std::string line;
  while (std::getline(file, line)) {
    size_t index = 0;
    ULONG value = 0;
    if (!ParseSeedLine(line, &index, &value)) {
      continue;
    }
    local_seeds[index - 1] = value;
    if (index > max_index) {
      max_index = index;
    }
  }

  if (max_index == 0) {
    return false;
  }

  local_seeds.resize(max_index);
  *seeds = std::move(local_seeds);
  return true;
}

std::vector<ULONG> MakePseudoRandomKeys(size_t count, ULONG salt = 0xA5A5A5A5u) {
  std::vector<ULONG> keys;
  keys.reserve(count);
  constexpr ULONG kMultiplier = 2654435761u;
  for (ULONG index = 1; keys.size() < count; ++index) {
    ULONG value = (index * kMultiplier) ^ salt;
    if (value == 0 || value == 0xffffffffu) {
      continue;
    }
    keys.push_back(value);
  }
  std::sort(keys.begin(), keys.end());
  return keys;
}

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
  std::string old_;
  bool had_old_ = false;
};

class PerfectHashOnlineTests : public ::testing::Test {
protected:
  void SetUp() override {
    HRESULT result = PerfectHashBootstrap(
        &classFactory_,
        &printError_,
        &printMessage_,
        &module_);
    ASSERT_GE(result, 0);

    auto createInstance = classFactory_->Vtbl->CreateInstance;
    result = createInstance(
        classFactory_,
        nullptr,
#ifdef PH_WINDOWS
        IID_PERFECT_HASH_ONLINE,
#else
        &IID_PERFECT_HASH_ONLINE,
#endif
        reinterpret_cast<void **>(&online_));
    ASSERT_GE(result, 0);
  }

  void TearDown() override {
    if (online_) {
      online_->Vtbl->Release(online_);
      online_ = nullptr;
    }
    if (classFactory_) {
      classFactory_->Vtbl->Release(classFactory_);
      classFactory_ = nullptr;
    }
#ifdef PH_WINDOWS
    if (module_) {
      FreeLibrary(module_);
      module_ = nullptr;
    }
#endif
  }

  PPERFECT_HASH_TABLE CreateTableFromKeys(
      const std::vector<ULONG> &keys,
      PERFECT_HASH_HASH_FUNCTION_ID hashFunctionId,
      bool allowAssigned16 = false,
      ULONG graphImpl = 0,
      const std::vector<ULONG> *seeds = nullptr,
      ULONG maxSolveTimeInSeconds = 0,
      bool useSystemRng = false,
      bool useRandomStartSeed = false,
      bool skipGraphVerification = false) {
    PERFECT_HASH_KEYS_LOAD_FLAGS keysFlags = {0};
    PERFECT_HASH_TABLE_CREATE_FLAGS tableFlags = {0};
    std::vector<PERFECT_HASH_TABLE_CREATE_PARAMETER> table_params;
    PERFECT_HASH_TABLE_CREATE_PARAMETERS tableCreateParams = {};
    PPERFECT_HASH_TABLE_CREATE_PARAMETERS tableCreateParamsPtr = nullptr;
    PPERFECT_HASH_TABLE table = nullptr;

    tableFlags.NoFileIo = TRUE;
    tableFlags.Quiet = TRUE;
    tableFlags.DoNotTryUseHash16Impl = allowAssigned16 ? FALSE : TRUE;
    tableFlags.RngUseRandomStartSeed = useRandomStartSeed ? TRUE : FALSE;
    tableFlags.SkipGraphVerification = skipGraphVerification ? TRUE : FALSE;

    if (graphImpl != 0) {
      PERFECT_HASH_TABLE_CREATE_PARAMETER param = {};
      param.Id = TableCreateParameterGraphImplId;
      param.AsULong = graphImpl;
      table_params.push_back(param);
    }

    if (seeds && !seeds->empty()) {
      PERFECT_HASH_TABLE_CREATE_PARAMETER param = {};
      param.Id = TableCreateParameterSeedsId;
      param.AsValueArray.Values = const_cast<ULONG *>(seeds->data());
      param.AsValueArray.NumberOfValues = static_cast<ULONG>(seeds->size());
      param.AsValueArray.ValueSizeInBytes = sizeof(ULONG);
      table_params.push_back(param);
    }

    if (maxSolveTimeInSeconds > 0) {
      PERFECT_HASH_TABLE_CREATE_PARAMETER param = {};
      param.Id = TableCreateParameterMaxSolveTimeInSecondsId;
      param.AsULong = maxSolveTimeInSeconds;
      table_params.push_back(param);
    }

    if (useSystemRng) {
      PERFECT_HASH_TABLE_CREATE_PARAMETER param = {};
      param.Id = TableCreateParameterRngId;
      param.AsULong = PerfectHashRngSystemId;
      table_params.push_back(param);
    }

    if (!table_params.empty()) {
      tableCreateParams.SizeOfStruct = sizeof(tableCreateParams);
      tableCreateParams.NumberOfElements =
          static_cast<ULONG>(table_params.size());
      tableCreateParams.Params = table_params.data();
      tableCreateParamsPtr = &tableCreateParams;
    }

    HRESULT result = online_->Vtbl->CreateTableFromKeys(
        online_,
        PerfectHashChm01AlgorithmId,
        hashFunctionId,
        PerfectHashAndMaskFunctionId,
        sizeof(ULONG),
        static_cast<ULONGLONG>(keys.size()),
        const_cast<ULONG *>(keys.data()),
        &keysFlags,
        &tableFlags,
        tableCreateParamsPtr,
        &table);
    EXPECT_GE(result, 0);
    return table;
  }

  PPERFECT_HASH_TABLE CreateTableFromKeys64(
      const std::vector<ULONGLONG> &keys,
      PERFECT_HASH_HASH_FUNCTION_ID hashFunctionId,
      bool allowAssigned16 = false) {
    PERFECT_HASH_KEYS_LOAD_FLAGS keysFlags = {0};
    PERFECT_HASH_TABLE_CREATE_FLAGS tableFlags = {0};
    PPERFECT_HASH_TABLE table = nullptr;

    tableFlags.NoFileIo = TRUE;
    tableFlags.Quiet = TRUE;
    tableFlags.DoNotTryUseHash16Impl = allowAssigned16 ? FALSE : TRUE;

    HRESULT result = online_->Vtbl->CreateTableFromKeys(
        online_,
        PerfectHashChm01AlgorithmId,
        hashFunctionId,
        PerfectHashAndMaskFunctionId,
        sizeof(ULONGLONG),
        static_cast<ULONGLONG>(keys.size()),
        const_cast<ULONGLONG *>(keys.data()),
        &keysFlags,
        &tableFlags,
        nullptr,
        &table);
    EXPECT_GE(result, 0);
    return table;
  }

protected:
  PICLASSFACTORY classFactory_ = nullptr;
  PPERFECT_HASH_ONLINE online_ = nullptr;
  PPERFECT_HASH_PRINT_ERROR printError_ = nullptr;
  PPERFECT_HASH_PRINT_MESSAGE printMessage_ = nullptr;
  HMODULE module_ = nullptr;
};

TEST_F(PerfectHashOnlineTests, UnsolvedTableDoesNotReturnAHandle) {
  // Identical fixed multipliers produce identical vertices for every key.
  std::vector<ULONG> keys = {1, 3, 5, 7};
  ULONG seeds[] = {1, 1, 0x1010};
  PERFECT_HASH_TABLE_CREATE_FLAGS flags = {};
  flags.NoFileIo = TRUE;
  flags.Quiet = TRUE;
  flags.DoNotTryUseHash16Impl = TRUE;
  flags.HashAllKeysFirst = TRUE;
  PERFECT_HASH_TABLE_CREATE_PARAMETER params[2] = {};
  params[0].Id = TableCreateParameterSeedsId;
  params[0].AsValueArray.Values = seeds;
  params[0].AsValueArray.NumberOfValues = 3;
  params[0].AsValueArray.ValueSizeInBytes = sizeof(ULONG);
  params[1].Id = TableCreateParameterFixedAttemptsId;
  params[1].AsULong = 1;
  PERFECT_HASH_TABLE_CREATE_PARAMETERS parameters = {};
  parameters.SizeOfStruct = sizeof(parameters);
  parameters.NumberOfElements = 2;
  parameters.Params = params;
  PPERFECT_HASH_TABLE table = nullptr;
  HRESULT result = online_->Vtbl->CreateTableFromKeys(
      online_, PerfectHashChm01AlgorithmId,
      PerfectHashHashMultiplyShiftRFunctionId, PerfectHashAndMaskFunctionId,
      sizeof(ULONG), keys.size(), keys.data(), nullptr, &flags, &parameters,
      &table);
  EXPECT_EQ(result, PH_E_ONLINE_CREATE_TABLE_ROUTINE_FAILED_TO_FIND_SOLUTION);
  EXPECT_EQ(table, nullptr);
  if (table) {
    reinterpret_cast<PerfectHashTableShim *>(table)->Vtbl->Release(table);
  }
}

TEST_F(PerfectHashOnlineTests, SolveTimeoutDoesNotReturnAHandle) {
  // Identical fixed multipliers produce identical vertices for every key.
  std::vector<ULONG> keys = {1, 3, 5, 7};
  ULONG seeds[] = {1, 1, 0x1010};
  PERFECT_HASH_TABLE_CREATE_FLAGS flags = {};
  flags.NoFileIo = TRUE;
  flags.Quiet = TRUE;
  flags.DoNotTryUseHash16Impl = TRUE;
  PERFECT_HASH_TABLE_CREATE_PARAMETER params[2] = {};
  params[0].Id = TableCreateParameterSeedsId;
  params[0].AsValueArray.Values = seeds;
  params[0].AsValueArray.NumberOfValues = 3;
  params[0].AsValueArray.ValueSizeInBytes = sizeof(ULONG);
  params[1].Id = TableCreateParameterMaxSolveTimeInSecondsId;
  params[1].AsULong = 1;
  PERFECT_HASH_TABLE_CREATE_PARAMETERS parameters = {};
  parameters.SizeOfStruct = sizeof(parameters);
  parameters.NumberOfElements = 2;
  parameters.Params = params;
  PPERFECT_HASH_TABLE table = nullptr;
  HRESULT result = online_->Vtbl->CreateTableFromKeys(
      online_, PerfectHashChm01AlgorithmId,
      PerfectHashHashMultiplyShiftRFunctionId, PerfectHashAndMaskFunctionId,
      sizeof(ULONG), keys.size(), keys.data(), nullptr, &flags, &parameters,
      &table);
  EXPECT_EQ(result, PH_E_SOLVE_TIMEOUT_EXPIRED);
  EXPECT_EQ(table, nullptr);
  if (table) {
    reinterpret_cast<PerfectHashTableShim *>(table)->Vtbl->Release(table);
  }
}

TEST_F(PerfectHashOnlineTests, CreateTableFromKeysReturnsUniqueIndexes) {
  const std::vector<ULONG> keys = {
      1, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53,
  };

  PPERFECT_HASH_TABLE table =
      CreateTableFromKeys(keys, PerfectHashHashMultiplyShiftRFunctionId);
  ASSERT_NE(table, nullptr);

  auto *shim = reinterpret_cast<PerfectHashTableShim *>(table);
  std::unordered_set<ULONG> seen;
  seen.reserve(keys.size());

  for (ULONG key : keys) {
    ULONG index = 0;
    HRESULT result = shim->Vtbl->Index(table, key, &index);
    ASSERT_GE(result, 0);
    EXPECT_TRUE(seen.insert(index).second);
  }

  shim->Vtbl->Release(table);
}

TEST_F(PerfectHashOnlineTests, SkipGraphVerificationReturnsUsableTable) {
  const std::vector<ULONG> keys = {
      1, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53,
  };

  PPERFECT_HASH_TABLE table =
      CreateTableFromKeys(keys, PerfectHashHashMultiplyShiftRFunctionId,
                          false, 0, nullptr, 0, false, false, true);
  ASSERT_NE(table, nullptr);

  auto *shim = reinterpret_cast<PerfectHashTableShim *>(table);
  std::unordered_set<ULONG> seen;
  seen.reserve(keys.size());

  for (ULONG key : keys) {
    ULONG index = 0;
    HRESULT result = shim->Vtbl->Index(table, key, &index);
    ASSERT_GE(result, 0);
    EXPECT_TRUE(seen.insert(index).second);
  }

  shim->Vtbl->Release(table);
}

TEST_F(PerfectHashOnlineTests, NeonHashAllKeysHandlesWidthsTailsAndFallback) {
  bool hasNeon = false;
#if defined(PH_WINDOWS) && defined(_M_ARM64)
  hasNeon = IsProcessorFeaturePresent(PF_ARM_NEON_INSTRUCTIONS_AVAILABLE);
#elif defined(PH_LINUX) && defined(__aarch64__) && defined(__ARM_NEON)
  hasNeon = (getauxval(AT_HWCAP) & HWCAP_ASIMD) != 0;
#elif defined(PH_MAC) && defined(__aarch64__) && defined(__ARM_NEON)
  hasNeon = true;
#endif

  const PERFECT_HASH_HASH_FUNCTION_ID hashes[] = {
#define NEON_TEST_HASH(Name) PerfectHashHash##Name##FunctionId,
      PERFECT_HASH_GOOD_HASH_FUNCTION_TABLE_ENTRY(NEON_TEST_HASH)
#undef NEON_TEST_HASH
      PerfectHashHashRotateMultiplyXorRotateFunctionId,
  };
  for (auto hash : hashes) {
    for (bool assigned16 : {false, true}) {
      for (ULONG tail = 0; tail < 4; ++tail) {
        auto keys = MakePseudoRandomKeys(100 + tail, 0x31415926u);
        PERFECT_HASH_TABLE_CREATE_FLAGS createFlags = {};
        createFlags.HashAllKeysFirst = TRUE;
        createFlags.Quiet = TRUE;
        createFlags.DoNotTryUseHash16Impl = !assigned16;
        for (bool disableNeon : {false, true}) {
          SCOPED_TRACE(::testing::Message() << "hash " << hash << ", width "
              << (assigned16 ? 16 : 32) << ", tail " << tail
              << ", disableNeon " << disableNeon);
          createFlags.DoNotTryUseNeonHashFunction = disableNeon;
          PPERFECT_HASH_TABLE table = nullptr;
          ASSERT_GE(online_->Vtbl->CreateTableFromKeys(
              online_, PerfectHashChm01AlgorithmId, hash, PerfectHashAndMaskFunctionId,
              sizeof(ULONG), keys.size(), keys.data(), nullptr, &createFlags,
              nullptr, &table), 0);
          ASSERT_NE(table, nullptr);
          auto *shim = reinterpret_cast<PerfectHashTableShim *>(table);
          PERFECT_HASH_TABLE_FLAGS flags = {};
          EXPECT_GE(shim->Vtbl->GetFlags(table, sizeof(flags), &flags), 0);
          EXPECT_EQ(flags.AssignedElementSizeInBits << 3, assigned16 ? 16u : 32u);
          EXPECT_EQ(!!flags.UsedNeonHashFunction,
                    hasNeon && !disableNeon && IsGoodPerfectHashHashFunctionId(hash));
          std::unordered_set<ULONG> seen;
          for (ULONG key : keys) {
            ULONG index = 0;
            EXPECT_GE(shim->Vtbl->Index(table, key, &index), 0);
            EXPECT_TRUE(seen.insert(index).second);
          }
          shim->Vtbl->Release(table);
        }
      }
    }
  }
}

#if defined(PH_HAS_LLVM)
// Every solver graph hashes through the table's one hash JIT with its own
// seeds, at the context's default concurrency. The key counts leave tails
// for the 16-, 8- and 4-key routines and the scalar routine.
TEST_F(PerfectHashOnlineTests, HashJitHashAllKeysServesEveryGraphAndWidth) {
  const PERFECT_HASH_HASH_FUNCTION_ID hashes[] = {
#define HASH_JIT_TEST_HASH(Name) PerfectHashHash##Name##FunctionId,
      PERFECT_HASH_GOOD_HASH_FUNCTION_TABLE_ENTRY(HASH_JIT_TEST_HASH)
#undef HASH_JIT_TEST_HASH
  };
  for (auto hash : hashes) {
    for (bool assigned16 : {false, true}) {
      for (size_t count : {5000u, 5015u}) {
        SCOPED_TRACE(::testing::Message() << "hash " << hash << ", width "
            << (assigned16 ? 16 : 32) << ", keys " << count);
        auto keys = MakePseudoRandomKeys(count, 0x27182818u);
        PERFECT_HASH_TABLE_CREATE_FLAGS createFlags = {};
        createFlags.HashAllKeysFirst = TRUE;
        createFlags.TryUseJitHashFunction = TRUE;
        createFlags.Quiet = TRUE;
        createFlags.DoNotTryUseHash16Impl = !assigned16;
        PPERFECT_HASH_TABLE table = nullptr;
        ASSERT_GE(online_->Vtbl->CreateTableFromKeys(
            online_, PerfectHashChm01AlgorithmId, hash, PerfectHashAndMaskFunctionId,
            sizeof(ULONG), keys.size(), keys.data(), nullptr, &createFlags,
            nullptr, &table), 0);
        ASSERT_NE(table, nullptr);
        auto *shim = reinterpret_cast<PerfectHashTableShim *>(table);
        PERFECT_HASH_TABLE_FLAGS flags = {};
        EXPECT_GE(shim->Vtbl->GetFlags(table, sizeof(flags), &flags), 0);
        EXPECT_EQ(flags.AssignedElementSizeInBits << 3, assigned16 ? 16u : 32u);
        EXPECT_TRUE(flags.UsedJitHashFunction);
        EXPECT_FALSE(flags.UsedNeonHashFunction);
        std::unordered_set<ULONG> seen;
        for (ULONG key : keys) {
          ULONG index = 0;
          EXPECT_GE(shim->Vtbl->Index(table, key, &index), 0);
          EXPECT_TRUE(seen.insert(index).second);
        }
        shim->Vtbl->Release(table);
      }
    }
  }
}
#endif

#if defined(PH_HAS_LLVM)
// The interface's Bind methods return routines that share one set of seeds
// per table: every routine hashes with the seeds of the most recent Bind.
TEST_F(PerfectHashOnlineTests, HashJitBoundRoutinesHashWithTheLatestBoundSeeds) {
  auto keys = MakePseudoRandomKeys(5000, 0x27182818u);
  PERFECT_HASH_TABLE_CREATE_FLAGS createFlags = {};
  createFlags.HashAllKeysFirst = TRUE;
  createFlags.TryUseJitHashFunction = TRUE;
  createFlags.Quiet = TRUE;
  createFlags.DoNotTryUseHash16Impl = TRUE;
  PPERFECT_HASH_TABLE table = nullptr;
  ASSERT_GE(online_->Vtbl->CreateTableFromKeys(
      online_, PerfectHashChm01AlgorithmId,
      PerfectHashHashMultiplyShiftRXFunctionId, PerfectHashAndMaskFunctionId,
      sizeof(ULONG), keys.size(), keys.data(), nullptr, &createFlags,
      nullptr, &table), 0);
  ASSERT_NE(table, nullptr);
  auto *shim = reinterpret_cast<PerfectHashTableShim *>(table);
  PPERFECT_HASH_TABLE_HASH_JIT_INTERFACE jit = nullptr;
  HRESULT result = shim->Vtbl->QueryInterface(
      table,
#ifdef PH_WINDOWS
      IID_PERFECT_HASH_TABLE_HASH_JIT_INTERFACE,
#else
      &IID_PERFECT_HASH_TABLE_HASH_JIT_INTERFACE,
#endif
      reinterpret_cast<void **>(&jit));
  ASSERT_GE(result, 0);
  ASSERT_NE(jit, nullptr);

  ULONG seeds[2][MAX_NUMBER_OF_SEEDS] = {
      {0x9E3779B1u, 0x85EBCA77u, 0x00000B07u},
      {0xC2B2AE3Du, 0x27D4EB2Fu, 0x00000311u},
  };
  PPERFECT_HASH_SEEDED_HASH_EX_BOUND hashEx = nullptr;
  PPERFECT_HASH_SEEDED_HASH_EX_BOUND32X4 hashEx32x4 = nullptr;
  PPERFECT_HASH_SEEDED_HASH_EX_BOUND32X8 hashEx32x8 = nullptr;
  for (auto &bound : seeds) {
    // Rebinding the scalar routine alone also redirects the batch routines.
    if (!hashEx32x4) {
      ASSERT_GE(jit->Vtbl->BindHashEx32x4(jit, bound, ~0u, &hashEx32x4), 0);
      ASSERT_GE(jit->Vtbl->BindHashEx32x8(jit, bound, ~0u, &hashEx32x8), 0);
    }
    ASSERT_GE(jit->Vtbl->BindHashEx(jit, bound, ~0u, &hashEx), 0);
    ASSERT_NE(hashEx, nullptr);
    ASSERT_NE(hashEx32x4, nullptr);
    ASSERT_NE(hashEx32x8, nullptr);
    for (size_t i = 0; i < 64; i += 8) {
      ULONGLONG pairs4[4] = {};
      ULONGLONG pairs8[8] = {};
      hashEx32x4(&keys[i], pairs4);
      hashEx32x8(&keys[i], pairs8);
      for (size_t lane = 0; lane < 8; ++lane) {
        ULONGLONG expected = 0;
        ASSERT_GE(shim->Vtbl->SeededHash(table, keys[i + lane],
                                         MAX_NUMBER_OF_SEEDS, bound,
                                         &expected), 0);
        EXPECT_EQ(hashEx(keys[i + lane]), expected);
        EXPECT_EQ(pairs8[lane], expected);
        if (lane < 4) {
          EXPECT_EQ(pairs4[lane], expected);
        }
      }
    }
  }
  jit->Vtbl->Release(jit);
  shim->Vtbl->Release(table);
}
#endif

TEST_F(PerfectHashOnlineTests, CreateTableFromUnsortedKeysReturnsUniqueIndexes) {
  const std::vector<ULONG> keys = {
      37, 13, 1, 53, 7, 19, 41, 3, 47, 11, 29, 23, 5, 31, 17, 43,
  };

  PERFECT_HASH_TABLE_CREATE_FLAGS tableFlags = {0};
  tableFlags.NoFileIo = TRUE;
  tableFlags.Quiet = TRUE;
  tableFlags.DoNotTryUseHash16Impl = TRUE;

  PPERFECT_HASH_TABLE table = nullptr;
  HRESULT result = PerfectHashOnlineCreateTableFromUnsortedKeys(
      online_,
      PerfectHashChm01AlgorithmId,
      PerfectHashHashMultiplyShiftRFunctionId,
      PerfectHashAndMaskFunctionId,
      sizeof(ULONG),
      static_cast<ULONGLONG>(keys.size()),
      const_cast<ULONG *>(keys.data()),
      nullptr,
      &tableFlags,
      nullptr,
      &table);
  ASSERT_GE(result, 0);
  ASSERT_NE(table, nullptr);

  auto *shim = reinterpret_cast<PerfectHashTableShim *>(table);
  std::unordered_set<ULONG> seen;
  seen.reserve(keys.size());

  for (ULONG key : keys) {
    ULONG index = 0;
    result = shim->Vtbl->Index(table, key, &index);
    ASSERT_GE(result, 0);
    EXPECT_TRUE(seen.insert(index).second);
  }

  shim->Vtbl->Release(table);
}

TEST_F(PerfectHashOnlineTests, Assigned16Boundary32767Vs32768Keys) {
  const PERFECT_HASH_HASH_FUNCTION_ID hashFunctionId =
      PerfectHashHashMultiplyShiftRFunctionId;
  const auto keys32767 = MakePseudoRandomKeys(32767, 0xA5A5A5A5u);
  const auto keys32768 = MakePseudoRandomKeys(32768, 0x5A5A5A5Au);

  PPERFECT_HASH_TABLE table32767 = CreateTableFromKeys(
      keys32767,
      hashFunctionId,
      true,
      3,
      nullptr,
      20);
  ASSERT_NE(table32767, nullptr);

  auto *shim32767 = reinterpret_cast<PerfectHashTableShim *>(table32767);
  PERFECT_HASH_TABLE_FLAGS flags32767 = {0};
  ASSERT_GE(shim32767->Vtbl->GetFlags(table32767, sizeof(flags32767), &flags32767), 0);
  EXPECT_EQ(flags32767.AssignedElementSizeInBits << 3, 16u);
  EXPECT_EQ(shim32767->Vtbl->SlowIndex, nullptr);
  shim32767->Vtbl->Release(table32767);

  PPERFECT_HASH_TABLE table32768 = CreateTableFromKeys(
      keys32768,
      hashFunctionId,
      true,
      3,
      nullptr,
      20);
  ASSERT_NE(table32768, nullptr);

  auto *shim32768 = reinterpret_cast<PerfectHashTableShim *>(table32768);
  PERFECT_HASH_TABLE_FLAGS flags32768 = {0};
  ASSERT_GE(shim32768->Vtbl->GetFlags(table32768, sizeof(flags32768), &flags32768), 0);
  EXPECT_EQ(flags32768.AssignedElementSizeInBits << 3, 32u);
  EXPECT_NE(shim32768->Vtbl->SlowIndex, nullptr);
  shim32768->Vtbl->Release(table32768);
}

TEST_F(PerfectHashOnlineTests, Assigned16RequiresGraphImpl3AndOptIn) {
  const PERFECT_HASH_HASH_FUNCTION_ID hashFunctionId =
      PerfectHashHashMultiplyShiftRFunctionId;
  const auto keys = MakePseudoRandomKeys(2048, 0xDEADBEEFu);

  PPERFECT_HASH_TABLE tableGraphImpl3 = CreateTableFromKeys(
      keys,
      hashFunctionId,
      true,
      3,
      nullptr,
      20);
  ASSERT_NE(tableGraphImpl3, nullptr);

  auto *shimGraphImpl3 = reinterpret_cast<PerfectHashTableShim *>(tableGraphImpl3);
  PERFECT_HASH_TABLE_FLAGS flagsGraphImpl3 = {0};
  ASSERT_GE(
      shimGraphImpl3->Vtbl->GetFlags(
          tableGraphImpl3,
          sizeof(flagsGraphImpl3),
          &flagsGraphImpl3),
      0);
  EXPECT_EQ(flagsGraphImpl3.AssignedElementSizeInBits << 3, 16u);
  EXPECT_EQ(shimGraphImpl3->Vtbl->SlowIndex, nullptr);
  shimGraphImpl3->Vtbl->Release(tableGraphImpl3);

  PPERFECT_HASH_TABLE tableOptOut = CreateTableFromKeys(
      keys,
      hashFunctionId,
      false,
      3,
      nullptr,
      20);
  ASSERT_NE(tableOptOut, nullptr);

  auto *shimOptOut = reinterpret_cast<PerfectHashTableShim *>(tableOptOut);
  PERFECT_HASH_TABLE_FLAGS flagsOptOut = {0};
  ASSERT_GE(shimOptOut->Vtbl->GetFlags(tableOptOut, sizeof(flagsOptOut), &flagsOptOut), 0);
  EXPECT_EQ(flagsOptOut.AssignedElementSizeInBits << 3, 32u);
  EXPECT_NE(shimOptOut->Vtbl->SlowIndex, nullptr);
  shimOptOut->Vtbl->Release(tableOptOut);

  PPERFECT_HASH_TABLE tableGraphImpl2 = CreateTableFromKeys(
      keys,
      hashFunctionId,
      true,
      2,
      nullptr,
      20);
  ASSERT_NE(tableGraphImpl2, nullptr);

  auto *shimGraphImpl2 = reinterpret_cast<PerfectHashTableShim *>(tableGraphImpl2);
  PERFECT_HASH_TABLE_FLAGS flagsGraphImpl2 = {0};
  ASSERT_GE(
      shimGraphImpl2->Vtbl->GetFlags(
          tableGraphImpl2,
          sizeof(flagsGraphImpl2),
          &flagsGraphImpl2),
      0);
  EXPECT_EQ(flagsGraphImpl2.AssignedElementSizeInBits << 3, 32u);
  EXPECT_NE(shimGraphImpl2->Vtbl->SlowIndex, nullptr);
  shimGraphImpl2->Vtbl->Release(tableGraphImpl2);
}

#if defined(PH_HAS_RAWDOG_JIT)
#if defined(PH_HAS_LLVM)
TEST_F(PerfectHashOnlineTests, HeldJitInterfaceFollowsBackendRecompile) {
  const std::vector<ULONG> keys = {1, 3, 5, 7, 11, 13, 17, 19};
  auto *table = CreateTableFromKeys(keys, PerfectHashHashMultiplyShiftRFunctionId);
  ASSERT_NE(table, nullptr);
  auto *shim = reinterpret_cast<PerfectHashTableShim *>(table);
  ULONG expected[2] = {};
  ASSERT_GE(shim->Vtbl->Index(table, keys[0], &expected[0]), 0);
  ASSERT_GE(shim->Vtbl->Index(table, keys[1], &expected[1]), 0);

  PERFECT_HASH_TABLE_COMPILE_FLAGS flags = {};
  flags.Jit = TRUE;
  flags.JitBackendRawDog = TRUE;
  ASSERT_GE(online_->Vtbl->CompileTable(online_, table, &flags), 0);
  PPERFECT_HASH_TABLE_JIT_INTERFACE held = nullptr;
  HRESULT result = shim->Vtbl->QueryInterface(
      table,
#ifdef PH_WINDOWS
      IID_PERFECT_HASH_TABLE_JIT_INTERFACE,
#else
      &IID_PERFECT_HASH_TABLE_JIT_INTERFACE,
#endif
      reinterpret_cast<void **>(&held));
  ASSERT_GE(result, 0);
  ASSERT_NE(held, nullptr);

  flags.JitBackendRawDog = FALSE;
  flags.JitBackendLlvm = TRUE;
  flags.JitIndex32x2 = TRUE;
  ASSERT_GE(online_->Vtbl->CompileTable(online_, table, &flags), 0);
  PERFECT_HASH_TABLE_JIT_INFO info = {};
  ASSERT_GE(held->Vtbl->GetInfo(held, &info), 0);
  ASSERT_TRUE(info.Flags.BackendLlvm);
  ASSERT_TRUE(info.Flags.Index32x2Compiled);

  // This table has only 32-bit keys. The failed request leaves the LLVM x2
  // binding intact. The held interface owns its reference to the table.
  flags.JitIndex64 = TRUE;
  ASSERT_EQ(online_->Vtbl->CompileTable(online_, table, &flags), PH_E_NOT_IMPLEMENTED);
  shim->Vtbl->Release(table);
  ULONG actual[2] = {};
  ASSERT_GE(held->Vtbl->Index32x2(held, keys[0], keys[1], &actual[0], &actual[1]), 0);
  EXPECT_EQ(actual[0], expected[0]);
  EXPECT_EQ(actual[1], expected[1]);
  held->Vtbl->Release(held);
}
#endif

#if defined(_M_ARM64) || defined(__aarch64__)
TEST_F(PerfectHashOnlineTests, FailedRawDogVectorCompilePreservesBatchIndex) {
  const std::vector<ULONG> keys = {1, 3, 5, 7, 11, 13, 17, 19};
  auto *table = CreateTableFromKeys(keys, PerfectHashHashMultiplyShiftRFunctionId);
  ASSERT_NE(table, nullptr);
  auto *shim = reinterpret_cast<PerfectHashTableShim *>(table);
  ULONG expected[4] = {};
  for (size_t i = 0; i < 4; ++i) {
    ASSERT_GE(shim->Vtbl->Index(table, keys[i], &expected[i]), 0);
  }

  PERFECT_HASH_TABLE_COMPILE_FLAGS flags = {};
  flags.Jit = TRUE;
  flags.JitBackendRawDog = TRUE;
  flags.JitIndex32x4 = TRUE;
  ASSERT_GE(online_->Vtbl->CompileTable(online_, table, &flags), 0);

  // RawDog implements the ARM64 x4 entry using scalar instructions.
  // The COM default permits partial widths, but this SIMD request fails.
  flags.JitVectorIndex32x4 = TRUE;
  ASSERT_EQ(online_->Vtbl->CompileTable(online_, table, &flags), PH_E_NOT_IMPLEMENTED);
  PERFECT_HASH_TABLE_FLAGS table_flags = {};
  ASSERT_GE(shim->Vtbl->GetFlags(table, sizeof(table_flags), &table_flags), 0);
  ASSERT_TRUE(table_flags.JitEnabled);
  PPERFECT_HASH_TABLE_JIT_INTERFACE jit = nullptr;
  HRESULT result = shim->Vtbl->QueryInterface(
      table,
#ifdef PH_WINDOWS
      IID_PERFECT_HASH_TABLE_JIT_INTERFACE,
#else
      &IID_PERFECT_HASH_TABLE_JIT_INTERFACE,
#endif
      reinterpret_cast<void **>(&jit));
  ASSERT_GE(result, 0);
  ASSERT_NE(jit, nullptr);
  ULONG actual[4] = {};
  ASSERT_GE(jit->Vtbl->Index32x4(jit, keys[0], keys[1], keys[2], keys[3],
                                  &actual[0], &actual[1], &actual[2], &actual[3]), 0);
  for (size_t i = 0; i < 4; ++i) {
    EXPECT_EQ(actual[i], expected[i]);
  }
  jit->Vtbl->Release(jit);
#if defined(PH_HAS_LLVM)
  // With the default COM flags, LLVM can still install a usable scalar
  // index when the requested ISA cap precludes its x16 routine.
  flags = {};
  flags.Jit = TRUE;
  flags.JitBackendLlvm = TRUE;
  flags.JitIndex32x16 = TRUE;
  flags.JitMaxIsa = PerfectHashJitMaxIsaAvx2;
  ASSERT_GE(online_->Vtbl->CompileTable(online_, table, &flags), 0);
  ASSERT_GE(shim->Vtbl->Index(table, keys[0], &actual[0]), 0);
  EXPECT_EQ(actual[0], expected[0]);
#endif
  shim->Vtbl->Release(table);
}
#endif

TEST_F(PerfectHashOnlineTests, RawDogJitIndexMatchesSlowIndex) {
  const std::vector<ULONG> keys = {
      1, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53,
      59, 61, 67, 71, 73, 79, 83, 89, 97, 101, 103, 107, 109, 113,
  };

  PPERFECT_HASH_TABLE table =
      CreateTableFromKeys(keys, PerfectHashHashMultiplyShiftRFunctionId);
  ASSERT_NE(table, nullptr);

  auto *shim = reinterpret_cast<PerfectHashTableShim *>(table);
  ASSERT_NE(shim->Vtbl->SlowIndex, nullptr);

  PERFECT_HASH_TABLE_COMPILE_FLAGS compileFlags = {0};
  compileFlags.Jit = TRUE;
  compileFlags.JitBackendRawDog = TRUE;

  HRESULT result = online_->Vtbl->CompileTable(online_, table, &compileFlags);
  if (result == PH_E_NOT_IMPLEMENTED) {
    shim->Vtbl->Release(table);
    GTEST_SKIP() << "RawDog Mulshrolate1RX unavailable on this host.";
  }
  ASSERT_GE(result, 0);

  for (ULONG key : keys) {
    ULONG slowIndex = 0;
    ULONG jitIndex = 0;

    result = shim->Vtbl->SlowIndex(table, key, &slowIndex);
    ASSERT_GE(result, 0);

    result = shim->Vtbl->Index(table, key, &jitIndex);
    ASSERT_GE(result, 0);

    EXPECT_EQ(slowIndex, jitIndex);
  }

  PPERFECT_HASH_TABLE_JIT_INTERFACE jitInterface = nullptr;
  result = shim->Vtbl->QueryInterface(
      table,
#ifdef PH_WINDOWS
      IID_PERFECT_HASH_TABLE_JIT_INTERFACE,
#else
      &IID_PERFECT_HASH_TABLE_JIT_INTERFACE,
#endif
      reinterpret_cast<void **>(&jitInterface));
  ASSERT_GE(result, 0);
  ASSERT_NE(jitInterface, nullptr);

  PERFECT_HASH_TABLE_JIT_INFO info = {0};
  result = jitInterface->Vtbl->GetInfo(jitInterface, &info);
  ASSERT_GE(result, 0);
  EXPECT_TRUE(info.Flags.BackendRawDog);

  jitInterface->Vtbl->Release(jitInterface);
  shim->Vtbl->Release(table);
}

TEST_F(PerfectHashOnlineTests, RawDogJitMulshrolate1RXMatchesSlowIndex) {
  const std::vector<ULONG> keys = {
      1, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53,
      59, 61, 67, 71, 73, 79, 83, 89, 97, 101, 103, 107, 109, 113,
  };

  PPERFECT_HASH_TABLE table =
      CreateTableFromKeys(keys, PerfectHashHashMulshrolate1RXFunctionId);
  ASSERT_NE(table, nullptr);

  auto *shim = reinterpret_cast<PerfectHashTableShim *>(table);
  ASSERT_NE(shim->Vtbl->SlowIndex, nullptr);

  PERFECT_HASH_TABLE_COMPILE_FLAGS compileFlags = {0};
  compileFlags.Jit = TRUE;
  compileFlags.JitBackendRawDog = TRUE;

  HRESULT result = online_->Vtbl->CompileTable(online_, table, &compileFlags);
  if (result == PH_E_NOT_IMPLEMENTED) {
    shim->Vtbl->Release(table);
    GTEST_SKIP() << "RawDog Mulshrolate3RX unavailable on this host.";
  }
  ASSERT_GE(result, 0);

  for (ULONG key : keys) {
    ULONG slowIndex = 0;
    ULONG jitIndex = 0;

    result = shim->Vtbl->SlowIndex(table, key, &slowIndex);
    ASSERT_GE(result, 0);

    result = shim->Vtbl->Index(table, key, &jitIndex);
    ASSERT_GE(result, 0);

    EXPECT_EQ(slowIndex, jitIndex);
  }

  PPERFECT_HASH_TABLE_JIT_INTERFACE jitInterface = nullptr;
  result = shim->Vtbl->QueryInterface(
      table,
#ifdef PH_WINDOWS
      IID_PERFECT_HASH_TABLE_JIT_INTERFACE,
#else
      &IID_PERFECT_HASH_TABLE_JIT_INTERFACE,
#endif
      reinterpret_cast<void **>(&jitInterface));
  ASSERT_GE(result, 0);
  ASSERT_NE(jitInterface, nullptr);

  PERFECT_HASH_TABLE_JIT_INFO info = {0};
  result = jitInterface->Vtbl->GetInfo(jitInterface, &info);
  ASSERT_GE(result, 0);
  EXPECT_TRUE(info.Flags.BackendRawDog);

  jitInterface->Vtbl->Release(jitInterface);
  shim->Vtbl->Release(table);
}

TEST_F(PerfectHashOnlineTests, RawDogJitMulshrolate2RXMatchesSlowIndex) {
  const std::vector<ULONG> keys = {
      1, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53,
      59, 61, 67, 71, 73, 79, 83, 89, 97, 101, 103, 107, 109, 113,
  };

  PPERFECT_HASH_TABLE table =
      CreateTableFromKeys(keys, PerfectHashHashMulshrolate2RXFunctionId);
  ASSERT_NE(table, nullptr);

  auto *shim = reinterpret_cast<PerfectHashTableShim *>(table);
  ASSERT_NE(shim->Vtbl->SlowIndex, nullptr);

  PERFECT_HASH_TABLE_COMPILE_FLAGS compileFlags = {0};
  compileFlags.Jit = TRUE;
  compileFlags.JitBackendRawDog = TRUE;

  HRESULT result = online_->Vtbl->CompileTable(online_, table, &compileFlags);
  if (result == PH_E_NOT_IMPLEMENTED) {
    shim->Vtbl->Release(table);
    GTEST_SKIP() << "RawDog Mulshrolate2RX unavailable on this host.";
  }
  ASSERT_GE(result, 0);

  for (ULONG key : keys) {
    ULONG slowIndex = 0;
    ULONG jitIndex = 0;

    result = shim->Vtbl->SlowIndex(table, key, &slowIndex);
    ASSERT_GE(result, 0);

    result = shim->Vtbl->Index(table, key, &jitIndex);
    ASSERT_GE(result, 0);

    EXPECT_EQ(slowIndex, jitIndex);
  }

  PPERFECT_HASH_TABLE_JIT_INTERFACE jitInterface = nullptr;
  result = shim->Vtbl->QueryInterface(
      table,
#ifdef PH_WINDOWS
      IID_PERFECT_HASH_TABLE_JIT_INTERFACE,
#else
      &IID_PERFECT_HASH_TABLE_JIT_INTERFACE,
#endif
      reinterpret_cast<void **>(&jitInterface));
  ASSERT_GE(result, 0);
  ASSERT_NE(jitInterface, nullptr);

  PERFECT_HASH_TABLE_JIT_INFO info = {0};
  result = jitInterface->Vtbl->GetInfo(jitInterface, &info);
  ASSERT_GE(result, 0);
  EXPECT_TRUE(info.Flags.BackendRawDog);

  jitInterface->Vtbl->Release(jitInterface);
  shim->Vtbl->Release(table);
}

TEST_F(PerfectHashOnlineTests,
       RawDogJitMulshrolate1RXIndex32x4MatchesIndex) {
  const std::vector<ULONG> keys = {
      1, 3, 5, 7, 11, 13, 17, 19,
      23, 29, 31, 37, 41, 43, 47, 53,
      59, 61, 67, 71, 73, 79, 83, 89,
      97, 101, 103, 107, 109, 113, 127, 131,
  };

  PPERFECT_HASH_TABLE table =
      CreateTableFromKeys(keys, PerfectHashHashMulshrolate1RXFunctionId);
  ASSERT_NE(table, nullptr);

  auto *shim = reinterpret_cast<PerfectHashTableShim *>(table);

  std::vector<ULONG> expected(keys.size());
  for (size_t i = 0; i < keys.size(); ++i) {
    HRESULT result = shim->Vtbl->Index(table, keys[i], &expected[i]);
    ASSERT_GE(result, 0);
  }

  PERFECT_HASH_TABLE_COMPILE_FLAGS compileFlags = {0};
  compileFlags.Jit = TRUE;
  compileFlags.JitBackendRawDog = TRUE;
  compileFlags.JitIndex32x4 = TRUE;

  HRESULT result = online_->Vtbl->CompileTable(online_, table, &compileFlags);
  if (result == PH_E_NOT_IMPLEMENTED) {
    shim->Vtbl->Release(table);
    GTEST_SKIP() << "RawDog Index32x4 unavailable on this host.";
  }
  ASSERT_GE(result, 0);

  PPERFECT_HASH_TABLE_JIT_INTERFACE jitInterface = nullptr;
  result = shim->Vtbl->QueryInterface(
      table,
#ifdef PH_WINDOWS
      IID_PERFECT_HASH_TABLE_JIT_INTERFACE,
#else
      &IID_PERFECT_HASH_TABLE_JIT_INTERFACE,
#endif
      reinterpret_cast<void **>(&jitInterface));
  ASSERT_GE(result, 0);
  ASSERT_NE(jitInterface, nullptr);

  for (size_t i = 0; i < keys.size(); i += 4) {
    ULONG i1 = 0;
    ULONG i2 = 0;
    ULONG i3 = 0;
    ULONG i4 = 0;

    result = jitInterface->Vtbl->Index32x4(jitInterface,
                                           keys[i],
                                           keys[i + 1],
                                           keys[i + 2],
                                           keys[i + 3],
                                           &i1,
                                           &i2,
                                           &i3,
                                           &i4);
    ASSERT_GE(result, 0);

    EXPECT_EQ(expected[i], i1);
    EXPECT_EQ(expected[i + 1], i2);
    EXPECT_EQ(expected[i + 2], i3);
    EXPECT_EQ(expected[i + 3], i4);
  }

  jitInterface->Vtbl->Release(jitInterface);
  shim->Vtbl->Release(table);
}

TEST_F(PerfectHashOnlineTests,
       RawDogJitMulshrolate1RXIndex32x8MatchesIndex) {
  const std::vector<ULONG> keys = {
      1, 3, 5, 7, 11, 13, 17, 19,
      23, 29, 31, 37, 41, 43, 47, 53,
      59, 61, 67, 71, 73, 79, 83, 89,
      97, 101, 103, 107, 109, 113, 127, 131,
  };

  PPERFECT_HASH_TABLE table =
      CreateTableFromKeys(keys, PerfectHashHashMulshrolate1RXFunctionId);
  ASSERT_NE(table, nullptr);

  auto *shim = reinterpret_cast<PerfectHashTableShim *>(table);

  std::vector<ULONG> expected(keys.size());
  for (size_t i = 0; i < keys.size(); ++i) {
    HRESULT result = shim->Vtbl->Index(table, keys[i], &expected[i]);
    ASSERT_GE(result, 0);
  }

  PERFECT_HASH_TABLE_COMPILE_FLAGS compileFlags = {0};
  compileFlags.Jit = TRUE;
  compileFlags.JitBackendRawDog = TRUE;
  compileFlags.JitIndex32x8 = TRUE;

  HRESULT result = online_->Vtbl->CompileTable(online_, table, &compileFlags);
  if (result == PH_E_NOT_IMPLEMENTED) {
    shim->Vtbl->Release(table);
    GTEST_SKIP() << "RawDog Index32x8 unavailable on this host.";
  }
  ASSERT_GE(result, 0);

  PPERFECT_HASH_TABLE_JIT_INTERFACE jitInterface = nullptr;
  result = shim->Vtbl->QueryInterface(
      table,
#ifdef PH_WINDOWS
      IID_PERFECT_HASH_TABLE_JIT_INTERFACE,
#else
      &IID_PERFECT_HASH_TABLE_JIT_INTERFACE,
#endif
      reinterpret_cast<void **>(&jitInterface));
  ASSERT_GE(result, 0);
  ASSERT_NE(jitInterface, nullptr);

  for (size_t i = 0; i < keys.size(); i += 8) {
    ULONG i1 = 0;
    ULONG i2 = 0;
    ULONG i3 = 0;
    ULONG i4 = 0;
    ULONG i5 = 0;
    ULONG i6 = 0;
    ULONG i7 = 0;
    ULONG i8 = 0;

    result = jitInterface->Vtbl->Index32x8(jitInterface,
                                           keys[i],
                                           keys[i + 1],
                                           keys[i + 2],
                                           keys[i + 3],
                                           keys[i + 4],
                                           keys[i + 5],
                                           keys[i + 6],
                                           keys[i + 7],
                                           &i1,
                                           &i2,
                                           &i3,
                                           &i4,
                                           &i5,
                                           &i6,
                                           &i7,
                                           &i8);
    ASSERT_GE(result, 0);

    EXPECT_EQ(expected[i], i1);
    EXPECT_EQ(expected[i + 1], i2);
    EXPECT_EQ(expected[i + 2], i3);
    EXPECT_EQ(expected[i + 3], i4);
    EXPECT_EQ(expected[i + 4], i5);
    EXPECT_EQ(expected[i + 5], i6);
    EXPECT_EQ(expected[i + 6], i7);
    EXPECT_EQ(expected[i + 7], i8);
  }

  jitInterface->Vtbl->Release(jitInterface);
  shim->Vtbl->Release(table);
}

TEST_F(PerfectHashOnlineTests,
       RawDogJitMultiplyShiftRIndex32x4MatchesIndex) {
  const std::vector<ULONG> keys = {
      1, 3, 5, 7, 11, 13, 17, 19,
      23, 29, 31, 37, 41, 43, 47, 53,
      59, 61, 67, 71, 73, 79, 83, 89,
      97, 101, 103, 107, 109, 113, 127, 131,
  };

  PPERFECT_HASH_TABLE table =
      CreateTableFromKeys(keys, PerfectHashHashMultiplyShiftRFunctionId);
  ASSERT_NE(table, nullptr);

  auto *shim = reinterpret_cast<PerfectHashTableShim *>(table);

  std::vector<ULONG> expected(keys.size());
  for (size_t i = 0; i < keys.size(); ++i) {
    HRESULT result = shim->Vtbl->Index(table, keys[i], &expected[i]);
    ASSERT_GE(result, 0);
  }

  PERFECT_HASH_TABLE_COMPILE_FLAGS compileFlags = {0};
  compileFlags.Jit = TRUE;
  compileFlags.JitBackendRawDog = TRUE;
  compileFlags.JitIndex32x4 = TRUE;

  HRESULT result = online_->Vtbl->CompileTable(online_, table, &compileFlags);
  if (result == PH_E_NOT_IMPLEMENTED) {
    shim->Vtbl->Release(table);
    GTEST_SKIP() << "RawDog Index32x4 unavailable on this host.";
  }
  ASSERT_GE(result, 0);

  PPERFECT_HASH_TABLE_JIT_INTERFACE jitInterface = nullptr;
  result = shim->Vtbl->QueryInterface(
      table,
#ifdef PH_WINDOWS
      IID_PERFECT_HASH_TABLE_JIT_INTERFACE,
#else
      &IID_PERFECT_HASH_TABLE_JIT_INTERFACE,
#endif
      reinterpret_cast<void **>(&jitInterface));
  ASSERT_GE(result, 0);
  ASSERT_NE(jitInterface, nullptr);

  for (size_t i = 0; i < keys.size(); i += 4) {
    ULONG i1 = 0;
    ULONG i2 = 0;
    ULONG i3 = 0;
    ULONG i4 = 0;

    result = jitInterface->Vtbl->Index32x4(jitInterface,
                                           keys[i],
                                           keys[i + 1],
                                           keys[i + 2],
                                           keys[i + 3],
                                           &i1,
                                           &i2,
                                           &i3,
                                           &i4);
    ASSERT_GE(result, 0);

    EXPECT_EQ(expected[i], i1);
    EXPECT_EQ(expected[i + 1], i2);
    EXPECT_EQ(expected[i + 2], i3);
    EXPECT_EQ(expected[i + 3], i4);
  }

  jitInterface->Vtbl->Release(jitInterface);
  shim->Vtbl->Release(table);
}

TEST_F(PerfectHashOnlineTests,
       RawDogJitMultiplyShiftRIndex32x8MatchesIndex) {
  const std::vector<ULONG> keys = {
      1, 3, 5, 7, 11, 13, 17, 19,
      23, 29, 31, 37, 41, 43, 47, 53,
      59, 61, 67, 71, 73, 79, 83, 89,
      97, 101, 103, 107, 109, 113, 127, 131,
  };

  PPERFECT_HASH_TABLE table =
      CreateTableFromKeys(keys, PerfectHashHashMultiplyShiftRFunctionId);
  ASSERT_NE(table, nullptr);

  auto *shim = reinterpret_cast<PerfectHashTableShim *>(table);

  std::vector<ULONG> expected(keys.size());
  for (size_t i = 0; i < keys.size(); ++i) {
    HRESULT result = shim->Vtbl->Index(table, keys[i], &expected[i]);
    ASSERT_GE(result, 0);
  }

  PERFECT_HASH_TABLE_COMPILE_FLAGS compileFlags = {0};
  compileFlags.Jit = TRUE;
  compileFlags.JitBackendRawDog = TRUE;
  compileFlags.JitIndex32x8 = TRUE;

  HRESULT result = online_->Vtbl->CompileTable(online_, table, &compileFlags);
  if (result == PH_E_NOT_IMPLEMENTED) {
    shim->Vtbl->Release(table);
    GTEST_SKIP() << "RawDog Index32x8 unavailable on this host.";
  }
  ASSERT_GE(result, 0);

  PPERFECT_HASH_TABLE_JIT_INTERFACE jitInterface = nullptr;
  result = shim->Vtbl->QueryInterface(
      table,
#ifdef PH_WINDOWS
      IID_PERFECT_HASH_TABLE_JIT_INTERFACE,
#else
      &IID_PERFECT_HASH_TABLE_JIT_INTERFACE,
#endif
      reinterpret_cast<void **>(&jitInterface));
  ASSERT_GE(result, 0);
  ASSERT_NE(jitInterface, nullptr);

  for (size_t i = 0; i < keys.size(); i += 8) {
    ULONG i1 = 0;
    ULONG i2 = 0;
    ULONG i3 = 0;
    ULONG i4 = 0;
    ULONG i5 = 0;
    ULONG i6 = 0;
    ULONG i7 = 0;
    ULONG i8 = 0;

    result = jitInterface->Vtbl->Index32x8(jitInterface,
                                           keys[i],
                                           keys[i + 1],
                                           keys[i + 2],
                                           keys[i + 3],
                                           keys[i + 4],
                                           keys[i + 5],
                                           keys[i + 6],
                                           keys[i + 7],
                                           &i1,
                                           &i2,
                                           &i3,
                                           &i4,
                                           &i5,
                                           &i6,
                                           &i7,
                                           &i8);
    ASSERT_GE(result, 0);

    EXPECT_EQ(expected[i], i1);
    EXPECT_EQ(expected[i + 1], i2);
    EXPECT_EQ(expected[i + 2], i3);
    EXPECT_EQ(expected[i + 3], i4);
    EXPECT_EQ(expected[i + 4], i5);
    EXPECT_EQ(expected[i + 5], i6);
    EXPECT_EQ(expected[i + 6], i7);
    EXPECT_EQ(expected[i + 7], i8);
  }

  jitInterface->Vtbl->Release(jitInterface);
  shim->Vtbl->Release(table);
}

TEST_F(PerfectHashOnlineTests,
       RawDogJitMultiplyShiftRIndex32x16MatchesIndex) {
  const std::vector<ULONG> keys = {
      1, 3, 5, 7, 11, 13, 17, 19,
      23, 29, 31, 37, 41, 43, 47, 53,
      59, 61, 67, 71, 73, 79, 83, 89,
      97, 101, 103, 107, 109, 113, 127, 131,
  };

  PPERFECT_HASH_TABLE table =
      CreateTableFromKeys(keys, PerfectHashHashMultiplyShiftRFunctionId);
  ASSERT_NE(table, nullptr);

  auto *shim = reinterpret_cast<PerfectHashTableShim *>(table);

  std::vector<ULONG> expected(keys.size());
  for (size_t i = 0; i < keys.size(); ++i) {
    HRESULT result = shim->Vtbl->Index(table, keys[i], &expected[i]);
    ASSERT_GE(result, 0);
  }

  PERFECT_HASH_TABLE_COMPILE_FLAGS compileFlags = {0};
  compileFlags.Jit = TRUE;
  compileFlags.JitBackendRawDog = TRUE;
  compileFlags.JitIndex32x16 = TRUE;

  HRESULT result = online_->Vtbl->CompileTable(online_, table, &compileFlags);
  if (result == PH_E_NOT_IMPLEMENTED) {
    shim->Vtbl->Release(table);
    GTEST_SKIP() << "RawDog Index32x16 unavailable on this host.";
  }
  ASSERT_GE(result, 0);

  PPERFECT_HASH_TABLE_JIT_INTERFACE jitInterface = nullptr;
  result = shim->Vtbl->QueryInterface(
      table,
#ifdef PH_WINDOWS
      IID_PERFECT_HASH_TABLE_JIT_INTERFACE,
#else
      &IID_PERFECT_HASH_TABLE_JIT_INTERFACE,
#endif
      reinterpret_cast<void **>(&jitInterface));
  ASSERT_GE(result, 0);
  ASSERT_NE(jitInterface, nullptr);

  for (size_t i = 0; i < keys.size(); i += 16) {
    ULONG index[16] = {};

    result = jitInterface->Vtbl->Index32x16(jitInterface,
                                            keys[i],
                                            keys[i + 1],
                                            keys[i + 2],
                                            keys[i + 3],
                                            keys[i + 4],
                                            keys[i + 5],
                                            keys[i + 6],
                                            keys[i + 7],
                                            keys[i + 8],
                                            keys[i + 9],
                                            keys[i + 10],
                                            keys[i + 11],
                                            keys[i + 12],
                                            keys[i + 13],
                                            keys[i + 14],
                                            keys[i + 15],
                                            &index[0],
                                            &index[1],
                                            &index[2],
                                            &index[3],
                                            &index[4],
                                            &index[5],
                                            &index[6],
                                            &index[7],
                                            &index[8],
                                            &index[9],
                                            &index[10],
                                            &index[11],
                                            &index[12],
                                            &index[13],
                                            &index[14],
                                            &index[15]);
    ASSERT_GE(result, 0);

    for (size_t lane = 0; lane < 16; ++lane) {
      EXPECT_EQ(expected[i + lane], index[lane]);
    }
  }

  jitInterface->Vtbl->Release(jitInterface);
  shim->Vtbl->Release(table);
}

TEST_F(PerfectHashOnlineTests,
       RawDogJitMultiplyShiftRXIndex32x4MatchesIndex) {
  const std::vector<ULONG> keys = {
      1, 3, 5, 7, 11, 13, 17, 19,
      23, 29, 31, 37, 41, 43, 47, 53,
      59, 61, 67, 71, 73, 79, 83, 89,
      97, 101, 103, 107, 109, 113, 127, 131,
  };

  PPERFECT_HASH_TABLE table =
      CreateTableFromKeys(keys, PerfectHashHashMultiplyShiftRXFunctionId);
  ASSERT_NE(table, nullptr);

  auto *shim = reinterpret_cast<PerfectHashTableShim *>(table);

  std::vector<ULONG> expected(keys.size());
  for (size_t i = 0; i < keys.size(); ++i) {
    HRESULT result = shim->Vtbl->Index(table, keys[i], &expected[i]);
    ASSERT_GE(result, 0);
  }

  PERFECT_HASH_TABLE_COMPILE_FLAGS compileFlags = {0};
  compileFlags.Jit = TRUE;
  compileFlags.JitBackendRawDog = TRUE;
  compileFlags.JitIndex32x4 = TRUE;

  HRESULT result = online_->Vtbl->CompileTable(online_, table, &compileFlags);
  if (result == PH_E_NOT_IMPLEMENTED) {
    shim->Vtbl->Release(table);
    GTEST_SKIP() << "RawDog Index32x4 unavailable on this host.";
  }
  ASSERT_GE(result, 0);

  PPERFECT_HASH_TABLE_JIT_INTERFACE jitInterface = nullptr;
  result = shim->Vtbl->QueryInterface(
      table,
#ifdef PH_WINDOWS
      IID_PERFECT_HASH_TABLE_JIT_INTERFACE,
#else
      &IID_PERFECT_HASH_TABLE_JIT_INTERFACE,
#endif
      reinterpret_cast<void **>(&jitInterface));
  ASSERT_GE(result, 0);
  ASSERT_NE(jitInterface, nullptr);

  for (size_t i = 0; i < keys.size(); i += 4) {
    ULONG i1 = 0;
    ULONG i2 = 0;
    ULONG i3 = 0;
    ULONG i4 = 0;

    result = jitInterface->Vtbl->Index32x4(jitInterface,
                                           keys[i],
                                           keys[i + 1],
                                           keys[i + 2],
                                           keys[i + 3],
                                           &i1,
                                           &i2,
                                           &i3,
                                           &i4);
    ASSERT_GE(result, 0);

    EXPECT_EQ(expected[i], i1);
    EXPECT_EQ(expected[i + 1], i2);
    EXPECT_EQ(expected[i + 2], i3);
    EXPECT_EQ(expected[i + 3], i4);
  }

  jitInterface->Vtbl->Release(jitInterface);
  shim->Vtbl->Release(table);
}

TEST_F(PerfectHashOnlineTests,
       RawDogJitMultiplyShiftRXIndex32x8MatchesIndex) {
  const std::vector<ULONG> keys = {
      1, 3, 5, 7, 11, 13, 17, 19,
      23, 29, 31, 37, 41, 43, 47, 53,
      59, 61, 67, 71, 73, 79, 83, 89,
      97, 101, 103, 107, 109, 113, 127, 131,
  };

  PPERFECT_HASH_TABLE table =
      CreateTableFromKeys(keys, PerfectHashHashMultiplyShiftRXFunctionId);
  ASSERT_NE(table, nullptr);

  auto *shim = reinterpret_cast<PerfectHashTableShim *>(table);

  std::vector<ULONG> expected(keys.size());
  for (size_t i = 0; i < keys.size(); ++i) {
    HRESULT result = shim->Vtbl->Index(table, keys[i], &expected[i]);
    ASSERT_GE(result, 0);
  }

  PERFECT_HASH_TABLE_COMPILE_FLAGS compileFlags = {0};
  compileFlags.Jit = TRUE;
  compileFlags.JitBackendRawDog = TRUE;
  compileFlags.JitIndex32x8 = TRUE;

  HRESULT result = online_->Vtbl->CompileTable(online_, table, &compileFlags);
  if (result == PH_E_NOT_IMPLEMENTED) {
    shim->Vtbl->Release(table);
    GTEST_SKIP() << "RawDog Index32x8 unavailable on this host.";
  }
  ASSERT_GE(result, 0);

  PPERFECT_HASH_TABLE_JIT_INTERFACE jitInterface = nullptr;
  result = shim->Vtbl->QueryInterface(
      table,
#ifdef PH_WINDOWS
      IID_PERFECT_HASH_TABLE_JIT_INTERFACE,
#else
      &IID_PERFECT_HASH_TABLE_JIT_INTERFACE,
#endif
      reinterpret_cast<void **>(&jitInterface));
  ASSERT_GE(result, 0);
  ASSERT_NE(jitInterface, nullptr);

  for (size_t i = 0; i < keys.size(); i += 8) {
    ULONG i1 = 0;
    ULONG i2 = 0;
    ULONG i3 = 0;
    ULONG i4 = 0;
    ULONG i5 = 0;
    ULONG i6 = 0;
    ULONG i7 = 0;
    ULONG i8 = 0;

    result = jitInterface->Vtbl->Index32x8(jitInterface,
                                           keys[i],
                                           keys[i + 1],
                                           keys[i + 2],
                                           keys[i + 3],
                                           keys[i + 4],
                                           keys[i + 5],
                                           keys[i + 6],
                                           keys[i + 7],
                                           &i1,
                                           &i2,
                                           &i3,
                                           &i4,
                                           &i5,
                                           &i6,
                                           &i7,
                                           &i8);
    ASSERT_GE(result, 0);

    EXPECT_EQ(expected[i], i1);
    EXPECT_EQ(expected[i + 1], i2);
    EXPECT_EQ(expected[i + 2], i3);
    EXPECT_EQ(expected[i + 3], i4);
    EXPECT_EQ(expected[i + 4], i5);
    EXPECT_EQ(expected[i + 5], i6);
    EXPECT_EQ(expected[i + 6], i7);
    EXPECT_EQ(expected[i + 7], i8);
  }

  jitInterface->Vtbl->Release(jitInterface);
  shim->Vtbl->Release(table);
}

TEST_F(PerfectHashOnlineTests,
       RawDogJitMultiplyShiftRXIndex32x16MatchesIndex) {
  const std::vector<ULONG> keys = {
      1, 3, 5, 7, 11, 13, 17, 19,
      23, 29, 31, 37, 41, 43, 47, 53,
      59, 61, 67, 71, 73, 79, 83, 89,
      97, 101, 103, 107, 109, 113, 127, 131,
  };

  PPERFECT_HASH_TABLE table =
      CreateTableFromKeys(keys, PerfectHashHashMultiplyShiftRXFunctionId);
  ASSERT_NE(table, nullptr);

  auto *shim = reinterpret_cast<PerfectHashTableShim *>(table);

  std::vector<ULONG> expected(keys.size());
  for (size_t i = 0; i < keys.size(); ++i) {
    HRESULT result = shim->Vtbl->Index(table, keys[i], &expected[i]);
    ASSERT_GE(result, 0);
  }

  PERFECT_HASH_TABLE_COMPILE_FLAGS compileFlags = {0};
  compileFlags.Jit = TRUE;
  compileFlags.JitBackendRawDog = TRUE;
  compileFlags.JitIndex32x16 = TRUE;

  HRESULT result = online_->Vtbl->CompileTable(online_, table, &compileFlags);
  if (result == PH_E_NOT_IMPLEMENTED) {
    shim->Vtbl->Release(table);
    GTEST_SKIP() << "RawDog Index32x16 unavailable on this host.";
  }
  ASSERT_GE(result, 0);

  PPERFECT_HASH_TABLE_JIT_INTERFACE jitInterface = nullptr;
  result = shim->Vtbl->QueryInterface(
      table,
#ifdef PH_WINDOWS
      IID_PERFECT_HASH_TABLE_JIT_INTERFACE,
#else
      &IID_PERFECT_HASH_TABLE_JIT_INTERFACE,
#endif
      reinterpret_cast<void **>(&jitInterface));
  ASSERT_GE(result, 0);
  ASSERT_NE(jitInterface, nullptr);

  for (size_t i = 0; i < keys.size(); i += 16) {
    ULONG index[16] = {};

    result = jitInterface->Vtbl->Index32x16(jitInterface,
                                            keys[i],
                                            keys[i + 1],
                                            keys[i + 2],
                                            keys[i + 3],
                                            keys[i + 4],
                                            keys[i + 5],
                                            keys[i + 6],
                                            keys[i + 7],
                                            keys[i + 8],
                                            keys[i + 9],
                                            keys[i + 10],
                                            keys[i + 11],
                                            keys[i + 12],
                                            keys[i + 13],
                                            keys[i + 14],
                                            keys[i + 15],
                                            &index[0],
                                            &index[1],
                                            &index[2],
                                            &index[3],
                                            &index[4],
                                            &index[5],
                                            &index[6],
                                            &index[7],
                                            &index[8],
                                            &index[9],
                                            &index[10],
                                            &index[11],
                                            &index[12],
                                            &index[13],
                                            &index[14],
                                            &index[15]);
    ASSERT_GE(result, 0);

    for (size_t lane = 0; lane < 16; ++lane) {
      EXPECT_EQ(expected[i + lane], index[lane]);
    }
  }

  jitInterface->Vtbl->Release(jitInterface);
  shim->Vtbl->Release(table);
}

TEST_F(PerfectHashOnlineTests,
       RawDogJitMultiplyShiftRIndex32x4Assigned16MatchesIndex) {
  const std::string keysPath =
      std::string(PERFECTHASH_TEST_ROOT_DIR) +
      "/keys/HologramWorld-31016.keys";
  const std::string seedsPath =
      std::string(PERFECTHASH_TEST_ROOT_DIR) +
      "/keys/HologramWorld-31016.MultiplyShiftR.seeds";
  std::vector<ULONG> keys;
  std::vector<ULONG> seeds;
  if (!LoadKeysFromFile(keysPath, &keys)) {
    GTEST_SKIP() << "Keys file unavailable: " << keysPath;
  }
  if (!LoadSeedsFromFile(seedsPath, &seeds) || seeds.size() < 3) {
    GTEST_SKIP() << "Seeds file unavailable: " << seedsPath;
  }

  PPERFECT_HASH_TABLE table = CreateTableFromKeys(
      keys, PerfectHashHashMultiplyShiftRFunctionId, true, 3, &seeds);
  ASSERT_NE(table, nullptr);

  auto *shim = reinterpret_cast<PerfectHashTableShim *>(table);
  PERFECT_HASH_TABLE_FLAGS flags = {0};
  ASSERT_GE(shim->Vtbl->GetFlags(table, sizeof(flags), &flags), 0);
  ASSERT_EQ(flags.AssignedElementSizeInBits << 3, 16u);

  std::vector<ULONG> expected(keys.size());
  for (size_t i = 0; i < keys.size(); ++i) {
    HRESULT result = shim->Vtbl->Index(table, keys[i], &expected[i]);
    ASSERT_GE(result, 0);
  }

  PERFECT_HASH_TABLE_COMPILE_FLAGS compileFlags = {0};
  compileFlags.Jit = TRUE;
  compileFlags.JitBackendRawDog = TRUE;
  compileFlags.JitIndex32x4 = TRUE;

  HRESULT result = online_->Vtbl->CompileTable(online_, table, &compileFlags);
  if (result == PH_E_NOT_IMPLEMENTED) {
    shim->Vtbl->Release(table);
    GTEST_SKIP() << "RawDog Index32x4 unavailable on this host.";
  }
  ASSERT_GE(result, 0);

  PPERFECT_HASH_TABLE_JIT_INTERFACE jitInterface = nullptr;
  result = shim->Vtbl->QueryInterface(
      table,
#ifdef PH_WINDOWS
      IID_PERFECT_HASH_TABLE_JIT_INTERFACE,
#else
      &IID_PERFECT_HASH_TABLE_JIT_INTERFACE,
#endif
      reinterpret_cast<void **>(&jitInterface));
  ASSERT_GE(result, 0);
  ASSERT_NE(jitInterface, nullptr);

  const size_t limit = keys.size() - (keys.size() % 4);
  for (size_t i = 0; i < limit; i += 4) {
    ULONG i1 = 0;
    ULONG i2 = 0;
    ULONG i3 = 0;
    ULONG i4 = 0;

    result = jitInterface->Vtbl->Index32x4(jitInterface,
                                           keys[i],
                                           keys[i + 1],
                                           keys[i + 2],
                                           keys[i + 3],
                                           &i1,
                                           &i2,
                                           &i3,
                                           &i4);
    ASSERT_GE(result, 0);

    EXPECT_EQ(expected[i], i1);
    EXPECT_EQ(expected[i + 1], i2);
    EXPECT_EQ(expected[i + 2], i3);
    EXPECT_EQ(expected[i + 3], i4);
  }

  jitInterface->Vtbl->Release(jitInterface);
  shim->Vtbl->Release(table);
}

TEST_F(PerfectHashOnlineTests,
       RawDogJitMultiplyShiftRIndex32x8Assigned16MatchesIndex) {
  const std::string keysPath =
      std::string(PERFECTHASH_TEST_ROOT_DIR) +
      "/keys/HologramWorld-31016.keys";
  const std::string seedsPath =
      std::string(PERFECTHASH_TEST_ROOT_DIR) +
      "/keys/HologramWorld-31016.MultiplyShiftR.seeds";
  std::vector<ULONG> keys;
  std::vector<ULONG> seeds;
  if (!LoadKeysFromFile(keysPath, &keys)) {
    GTEST_SKIP() << "Keys file unavailable: " << keysPath;
  }
  if (!LoadSeedsFromFile(seedsPath, &seeds) || seeds.size() < 3) {
    GTEST_SKIP() << "Seeds file unavailable: " << seedsPath;
  }

  PPERFECT_HASH_TABLE table = CreateTableFromKeys(
      keys, PerfectHashHashMultiplyShiftRFunctionId, true, 3, &seeds);
  ASSERT_NE(table, nullptr);

  auto *shim = reinterpret_cast<PerfectHashTableShim *>(table);
  PERFECT_HASH_TABLE_FLAGS flags = {0};
  ASSERT_GE(shim->Vtbl->GetFlags(table, sizeof(flags), &flags), 0);
  ASSERT_EQ(flags.AssignedElementSizeInBits << 3, 16u);

  std::vector<ULONG> expected(keys.size());
  for (size_t i = 0; i < keys.size(); ++i) {
    HRESULT result = shim->Vtbl->Index(table, keys[i], &expected[i]);
    ASSERT_GE(result, 0);
  }

  PERFECT_HASH_TABLE_COMPILE_FLAGS compileFlags = {0};
  compileFlags.Jit = TRUE;
  compileFlags.JitBackendRawDog = TRUE;
  compileFlags.JitIndex32x8 = TRUE;

  HRESULT result = online_->Vtbl->CompileTable(online_, table, &compileFlags);
  if (result == PH_E_NOT_IMPLEMENTED) {
    shim->Vtbl->Release(table);
    GTEST_SKIP() << "RawDog Index32x8 unavailable on this host.";
  }
  ASSERT_GE(result, 0);

  PPERFECT_HASH_TABLE_JIT_INTERFACE jitInterface = nullptr;
  result = shim->Vtbl->QueryInterface(
      table,
#ifdef PH_WINDOWS
      IID_PERFECT_HASH_TABLE_JIT_INTERFACE,
#else
      &IID_PERFECT_HASH_TABLE_JIT_INTERFACE,
#endif
      reinterpret_cast<void **>(&jitInterface));
  ASSERT_GE(result, 0);
  ASSERT_NE(jitInterface, nullptr);

  const size_t limit = keys.size() - (keys.size() % 8);
  for (size_t i = 0; i < limit; i += 8) {
    ULONG i1 = 0;
    ULONG i2 = 0;
    ULONG i3 = 0;
    ULONG i4 = 0;
    ULONG i5 = 0;
    ULONG i6 = 0;
    ULONG i7 = 0;
    ULONG i8 = 0;

    result = jitInterface->Vtbl->Index32x8(jitInterface,
                                           keys[i],
                                           keys[i + 1],
                                           keys[i + 2],
                                           keys[i + 3],
                                           keys[i + 4],
                                           keys[i + 5],
                                           keys[i + 6],
                                           keys[i + 7],
                                           &i1,
                                           &i2,
                                           &i3,
                                           &i4,
                                           &i5,
                                           &i6,
                                           &i7,
                                           &i8);
    ASSERT_GE(result, 0);

    EXPECT_EQ(expected[i], i1);
    EXPECT_EQ(expected[i + 1], i2);
    EXPECT_EQ(expected[i + 2], i3);
    EXPECT_EQ(expected[i + 3], i4);
    EXPECT_EQ(expected[i + 4], i5);
    EXPECT_EQ(expected[i + 5], i6);
    EXPECT_EQ(expected[i + 6], i7);
    EXPECT_EQ(expected[i + 7], i8);
  }

  jitInterface->Vtbl->Release(jitInterface);
  shim->Vtbl->Release(table);
}

TEST_F(PerfectHashOnlineTests,
       RawDogJitMultiplyShiftRIndex32x16Assigned16MatchesIndex) {
  const std::string keysPath =
      std::string(PERFECTHASH_TEST_ROOT_DIR) +
      "/keys/HologramWorld-31016.keys";
  const std::string seedsPath =
      std::string(PERFECTHASH_TEST_ROOT_DIR) +
      "/keys/HologramWorld-31016.MultiplyShiftR.seeds";
  std::vector<ULONG> keys;
  std::vector<ULONG> seeds;
  if (!LoadKeysFromFile(keysPath, &keys)) {
    GTEST_SKIP() << "Keys file unavailable: " << keysPath;
  }
  if (!LoadSeedsFromFile(seedsPath, &seeds) || seeds.size() < 3) {
    GTEST_SKIP() << "Seeds file unavailable: " << seedsPath;
  }

  PPERFECT_HASH_TABLE table = CreateTableFromKeys(
      keys, PerfectHashHashMultiplyShiftRFunctionId, true, 3, &seeds);
  ASSERT_NE(table, nullptr);

  auto *shim = reinterpret_cast<PerfectHashTableShim *>(table);
  PERFECT_HASH_TABLE_FLAGS flags = {0};
  ASSERT_GE(shim->Vtbl->GetFlags(table, sizeof(flags), &flags), 0);
  ASSERT_EQ(flags.AssignedElementSizeInBits << 3, 16u);

  std::vector<ULONG> expected(keys.size());
  for (size_t i = 0; i < keys.size(); ++i) {
    HRESULT result = shim->Vtbl->Index(table, keys[i], &expected[i]);
    ASSERT_GE(result, 0);
  }

  PERFECT_HASH_TABLE_COMPILE_FLAGS compileFlags = {0};
  compileFlags.Jit = TRUE;
  compileFlags.JitBackendRawDog = TRUE;
  compileFlags.JitIndex32x16 = TRUE;

  HRESULT result = online_->Vtbl->CompileTable(online_, table, &compileFlags);
  if (result == PH_E_NOT_IMPLEMENTED) {
    shim->Vtbl->Release(table);
    GTEST_SKIP() << "RawDog Index32x16 unavailable on this host.";
  }
  ASSERT_GE(result, 0);

  PPERFECT_HASH_TABLE_JIT_INTERFACE jitInterface = nullptr;
  result = shim->Vtbl->QueryInterface(
      table,
#ifdef PH_WINDOWS
      IID_PERFECT_HASH_TABLE_JIT_INTERFACE,
#else
      &IID_PERFECT_HASH_TABLE_JIT_INTERFACE,
#endif
      reinterpret_cast<void **>(&jitInterface));
  ASSERT_GE(result, 0);
  ASSERT_NE(jitInterface, nullptr);

  const size_t limit = keys.size() - (keys.size() % 16);
  for (size_t i = 0; i < limit; i += 16) {
    ULONG index[16] = {};

    result = jitInterface->Vtbl->Index32x16(jitInterface,
                                            keys[i],
                                            keys[i + 1],
                                            keys[i + 2],
                                            keys[i + 3],
                                            keys[i + 4],
                                            keys[i + 5],
                                            keys[i + 6],
                                            keys[i + 7],
                                            keys[i + 8],
                                            keys[i + 9],
                                            keys[i + 10],
                                            keys[i + 11],
                                            keys[i + 12],
                                            keys[i + 13],
                                            keys[i + 14],
                                            keys[i + 15],
                                            &index[0],
                                            &index[1],
                                            &index[2],
                                            &index[3],
                                            &index[4],
                                            &index[5],
                                            &index[6],
                                            &index[7],
                                            &index[8],
                                            &index[9],
                                            &index[10],
                                            &index[11],
                                            &index[12],
                                            &index[13],
                                            &index[14],
                                            &index[15]);
    ASSERT_GE(result, 0);

    for (size_t lane = 0; lane < 16; ++lane) {
      EXPECT_EQ(expected[i + lane], index[lane]);
    }
  }

  jitInterface->Vtbl->Release(jitInterface);
  shim->Vtbl->Release(table);
}

TEST_F(PerfectHashOnlineTests,
       RawDogJitMultiplyShiftRXIndex32x4Assigned16MatchesIndex) {
  const std::string keysPath =
      std::string(PERFECTHASH_TEST_ROOT_DIR) +
      "/keys/HologramWorld-31016.keys";
  const std::string seedsPath =
      std::string(PERFECTHASH_TEST_ROOT_DIR) +
      "/keys/HologramWorld-31016.MultiplyShiftRX.seeds";
  std::vector<ULONG> keys;
  std::vector<ULONG> seeds;
  if (!LoadKeysFromFile(keysPath, &keys)) {
    GTEST_SKIP() << "Keys file unavailable: " << keysPath;
  }
  if (!LoadSeedsFromFile(seedsPath, &seeds) || seeds.size() < 3) {
    GTEST_SKIP() << "Seeds file unavailable: " << seedsPath;
  }

  PPERFECT_HASH_TABLE table = CreateTableFromKeys(
      keys, PerfectHashHashMultiplyShiftRXFunctionId, true, 3, &seeds);
  ASSERT_NE(table, nullptr);

  auto *shim = reinterpret_cast<PerfectHashTableShim *>(table);
  PERFECT_HASH_TABLE_FLAGS flags = {0};
  ASSERT_GE(shim->Vtbl->GetFlags(table, sizeof(flags), &flags), 0);
  ASSERT_EQ(flags.AssignedElementSizeInBits << 3, 16u);

  std::vector<ULONG> expected(keys.size());
  for (size_t i = 0; i < keys.size(); ++i) {
    HRESULT result = shim->Vtbl->Index(table, keys[i], &expected[i]);
    ASSERT_GE(result, 0);
  }

  PERFECT_HASH_TABLE_COMPILE_FLAGS compileFlags = {0};
  compileFlags.Jit = TRUE;
  compileFlags.JitBackendRawDog = TRUE;
  compileFlags.JitIndex32x4 = TRUE;

  HRESULT result = online_->Vtbl->CompileTable(online_, table, &compileFlags);
  if (result == PH_E_NOT_IMPLEMENTED) {
    shim->Vtbl->Release(table);
    GTEST_SKIP() << "RawDog Index32x4 unavailable on this host.";
  }
  ASSERT_GE(result, 0);

  PPERFECT_HASH_TABLE_JIT_INTERFACE jitInterface = nullptr;
  result = shim->Vtbl->QueryInterface(
      table,
#ifdef PH_WINDOWS
      IID_PERFECT_HASH_TABLE_JIT_INTERFACE,
#else
      &IID_PERFECT_HASH_TABLE_JIT_INTERFACE,
#endif
      reinterpret_cast<void **>(&jitInterface));
  ASSERT_GE(result, 0);
  ASSERT_NE(jitInterface, nullptr);

  const size_t limit = keys.size() - (keys.size() % 4);
  for (size_t i = 0; i < limit; i += 4) {
    ULONG i1 = 0;
    ULONG i2 = 0;
    ULONG i3 = 0;
    ULONG i4 = 0;

    result = jitInterface->Vtbl->Index32x4(jitInterface,
                                           keys[i],
                                           keys[i + 1],
                                           keys[i + 2],
                                           keys[i + 3],
                                           &i1,
                                           &i2,
                                           &i3,
                                           &i4);
    ASSERT_GE(result, 0);

    EXPECT_EQ(expected[i], i1);
    EXPECT_EQ(expected[i + 1], i2);
    EXPECT_EQ(expected[i + 2], i3);
    EXPECT_EQ(expected[i + 3], i4);
  }

  jitInterface->Vtbl->Release(jitInterface);
  shim->Vtbl->Release(table);
}

TEST_F(PerfectHashOnlineTests,
       RawDogJitMultiplyShiftRXIndex32x8Assigned16MatchesIndex) {
  const std::string keysPath =
      std::string(PERFECTHASH_TEST_ROOT_DIR) +
      "/keys/HologramWorld-31016.keys";
  const std::string seedsPath =
      std::string(PERFECTHASH_TEST_ROOT_DIR) +
      "/keys/HologramWorld-31016.MultiplyShiftRX.seeds";
  std::vector<ULONG> keys;
  std::vector<ULONG> seeds;
  if (!LoadKeysFromFile(keysPath, &keys)) {
    GTEST_SKIP() << "Keys file unavailable: " << keysPath;
  }
  if (!LoadSeedsFromFile(seedsPath, &seeds) || seeds.size() < 3) {
    GTEST_SKIP() << "Seeds file unavailable: " << seedsPath;
  }

  PPERFECT_HASH_TABLE table = CreateTableFromKeys(
      keys, PerfectHashHashMultiplyShiftRXFunctionId, true, 3, &seeds);
  ASSERT_NE(table, nullptr);

  auto *shim = reinterpret_cast<PerfectHashTableShim *>(table);
  PERFECT_HASH_TABLE_FLAGS flags = {0};
  ASSERT_GE(shim->Vtbl->GetFlags(table, sizeof(flags), &flags), 0);
  ASSERT_EQ(flags.AssignedElementSizeInBits << 3, 16u);

  std::vector<ULONG> expected(keys.size());
  for (size_t i = 0; i < keys.size(); ++i) {
    HRESULT result = shim->Vtbl->Index(table, keys[i], &expected[i]);
    ASSERT_GE(result, 0);
  }

  PERFECT_HASH_TABLE_COMPILE_FLAGS compileFlags = {0};
  compileFlags.Jit = TRUE;
  compileFlags.JitBackendRawDog = TRUE;
  compileFlags.JitIndex32x8 = TRUE;

  HRESULT result = online_->Vtbl->CompileTable(online_, table, &compileFlags);
  if (result == PH_E_NOT_IMPLEMENTED) {
    shim->Vtbl->Release(table);
    GTEST_SKIP() << "RawDog Index32x8 unavailable on this host.";
  }
  ASSERT_GE(result, 0);

  PPERFECT_HASH_TABLE_JIT_INTERFACE jitInterface = nullptr;
  result = shim->Vtbl->QueryInterface(
      table,
#ifdef PH_WINDOWS
      IID_PERFECT_HASH_TABLE_JIT_INTERFACE,
#else
      &IID_PERFECT_HASH_TABLE_JIT_INTERFACE,
#endif
      reinterpret_cast<void **>(&jitInterface));
  ASSERT_GE(result, 0);
  ASSERT_NE(jitInterface, nullptr);

  const size_t limit = keys.size() - (keys.size() % 8);
  for (size_t i = 0; i < limit; i += 8) {
    ULONG i1 = 0;
    ULONG i2 = 0;
    ULONG i3 = 0;
    ULONG i4 = 0;
    ULONG i5 = 0;
    ULONG i6 = 0;
    ULONG i7 = 0;
    ULONG i8 = 0;

    result = jitInterface->Vtbl->Index32x8(jitInterface,
                                           keys[i],
                                           keys[i + 1],
                                           keys[i + 2],
                                           keys[i + 3],
                                           keys[i + 4],
                                           keys[i + 5],
                                           keys[i + 6],
                                           keys[i + 7],
                                           &i1,
                                           &i2,
                                           &i3,
                                           &i4,
                                           &i5,
                                           &i6,
                                           &i7,
                                           &i8);
    ASSERT_GE(result, 0);

    EXPECT_EQ(expected[i], i1);
    EXPECT_EQ(expected[i + 1], i2);
    EXPECT_EQ(expected[i + 2], i3);
    EXPECT_EQ(expected[i + 3], i4);
    EXPECT_EQ(expected[i + 4], i5);
    EXPECT_EQ(expected[i + 5], i6);
    EXPECT_EQ(expected[i + 6], i7);
    EXPECT_EQ(expected[i + 7], i8);
  }

  jitInterface->Vtbl->Release(jitInterface);
  shim->Vtbl->Release(table);
}

TEST_F(PerfectHashOnlineTests,
       RawDogJitMultiplyShiftRXIndex32x16Assigned16MatchesIndex) {
  const std::string keysPath =
      std::string(PERFECTHASH_TEST_ROOT_DIR) +
      "/keys/HologramWorld-31016.keys";
  const std::string seedsPath =
      std::string(PERFECTHASH_TEST_ROOT_DIR) +
      "/keys/HologramWorld-31016.MultiplyShiftRX.seeds";
  std::vector<ULONG> keys;
  std::vector<ULONG> seeds;
  if (!LoadKeysFromFile(keysPath, &keys)) {
    GTEST_SKIP() << "Keys file unavailable: " << keysPath;
  }
  if (!LoadSeedsFromFile(seedsPath, &seeds) || seeds.size() < 3) {
    GTEST_SKIP() << "Seeds file unavailable: " << seedsPath;
  }

  PPERFECT_HASH_TABLE table = CreateTableFromKeys(
      keys, PerfectHashHashMultiplyShiftRXFunctionId, true, 3, &seeds);
  ASSERT_NE(table, nullptr);

  auto *shim = reinterpret_cast<PerfectHashTableShim *>(table);
  PERFECT_HASH_TABLE_FLAGS flags = {0};
  ASSERT_GE(shim->Vtbl->GetFlags(table, sizeof(flags), &flags), 0);
  ASSERT_EQ(flags.AssignedElementSizeInBits << 3, 16u);

  std::vector<ULONG> expected(keys.size());
  for (size_t i = 0; i < keys.size(); ++i) {
    HRESULT result = shim->Vtbl->Index(table, keys[i], &expected[i]);
    ASSERT_GE(result, 0);
  }

  PERFECT_HASH_TABLE_COMPILE_FLAGS compileFlags = {0};
  compileFlags.Jit = TRUE;
  compileFlags.JitBackendRawDog = TRUE;
  compileFlags.JitIndex32x16 = TRUE;

  HRESULT result = online_->Vtbl->CompileTable(online_, table, &compileFlags);
  if (result == PH_E_NOT_IMPLEMENTED) {
    shim->Vtbl->Release(table);
    GTEST_SKIP() << "RawDog Index32x16 unavailable on this host.";
  }
  ASSERT_GE(result, 0);

  PPERFECT_HASH_TABLE_JIT_INTERFACE jitInterface = nullptr;
  result = shim->Vtbl->QueryInterface(
      table,
#ifdef PH_WINDOWS
      IID_PERFECT_HASH_TABLE_JIT_INTERFACE,
#else
      &IID_PERFECT_HASH_TABLE_JIT_INTERFACE,
#endif
      reinterpret_cast<void **>(&jitInterface));
  ASSERT_GE(result, 0);
  ASSERT_NE(jitInterface, nullptr);

  const size_t limit = keys.size() - (keys.size() % 16);
  for (size_t i = 0; i < limit; i += 16) {
    ULONG index[16] = {};

    result = jitInterface->Vtbl->Index32x16(jitInterface,
                                            keys[i],
                                            keys[i + 1],
                                            keys[i + 2],
                                            keys[i + 3],
                                            keys[i + 4],
                                            keys[i + 5],
                                            keys[i + 6],
                                            keys[i + 7],
                                            keys[i + 8],
                                            keys[i + 9],
                                            keys[i + 10],
                                            keys[i + 11],
                                            keys[i + 12],
                                            keys[i + 13],
                                            keys[i + 14],
                                            keys[i + 15],
                                            &index[0],
                                            &index[1],
                                            &index[2],
                                            &index[3],
                                            &index[4],
                                            &index[5],
                                            &index[6],
                                            &index[7],
                                            &index[8],
                                            &index[9],
                                            &index[10],
                                            &index[11],
                                            &index[12],
                                            &index[13],
                                            &index[14],
                                            &index[15]);
    ASSERT_GE(result, 0);

    for (size_t lane = 0; lane < 16; ++lane) {
      EXPECT_EQ(expected[i + lane], index[lane]);
    }
  }

  jitInterface->Vtbl->Release(jitInterface);
  shim->Vtbl->Release(table);
}

TEST_F(PerfectHashOnlineTests,
       RawDogJitMulshrolate2RXIndex32x8Assigned16MatchesIndex) {
  const std::string keysPath =
      std::string(PERFECTHASH_TEST_ROOT_DIR) +
      "/keys/HologramWorld-31016.keys";
  const std::string seedsPath =
      std::string(PERFECTHASH_TEST_ROOT_DIR) +
      "/keys/HologramWorld-31016.Mulshrolate2RX.seeds";
  std::vector<ULONG> keys;
  std::vector<ULONG> seeds;
  if (!LoadKeysFromFile(keysPath, &keys)) {
    GTEST_SKIP() << "Keys file unavailable: " << keysPath;
  }
  if (!LoadSeedsFromFile(seedsPath, &seeds) || seeds.size() < 3) {
    GTEST_SKIP() << "Seeds file unavailable: " << seedsPath;
  }

  ScopedEnvVar rawdogVectorVersion("PH_RAWDOG_VECTOR_VERSION", "4");
  PPERFECT_HASH_TABLE table = CreateTableFromKeys(
      keys, PerfectHashHashMulshrolate2RXFunctionId, true, 3, &seeds);
  ASSERT_NE(table, nullptr);

  auto *shim = reinterpret_cast<PerfectHashTableShim *>(table);
  PERFECT_HASH_TABLE_FLAGS flags = {0};
  ASSERT_GE(shim->Vtbl->GetFlags(table, sizeof(flags), &flags), 0);
  ASSERT_EQ(flags.AssignedElementSizeInBits << 3, 16u);

  std::vector<ULONG> expected(keys.size());
  for (size_t i = 0; i < keys.size(); ++i) {
    HRESULT result = shim->Vtbl->Index(table, keys[i], &expected[i]);
    ASSERT_GE(result, 0);
  }

  PERFECT_HASH_TABLE_COMPILE_FLAGS compileFlags = {0};
  compileFlags.Jit = TRUE;
  compileFlags.JitBackendRawDog = TRUE;
  compileFlags.JitIndex32x8 = TRUE;

  HRESULT result = online_->Vtbl->CompileTable(online_, table, &compileFlags);
  if (result == PH_E_NOT_IMPLEMENTED) {
    shim->Vtbl->Release(table);
    GTEST_SKIP() << "RawDog Index32x8 unavailable on this host.";
  }
  ASSERT_GE(result, 0);

  PPERFECT_HASH_TABLE_JIT_INTERFACE jitInterface = nullptr;
  result = shim->Vtbl->QueryInterface(
      table,
#ifdef PH_WINDOWS
      IID_PERFECT_HASH_TABLE_JIT_INTERFACE,
#else
      &IID_PERFECT_HASH_TABLE_JIT_INTERFACE,
#endif
      reinterpret_cast<void **>(&jitInterface));
  ASSERT_GE(result, 0);
  ASSERT_NE(jitInterface, nullptr);

  const size_t limit = keys.size() - (keys.size() % 8);
  for (size_t i = 0; i < limit; i += 8) {
    ULONG i1 = 0;
    ULONG i2 = 0;
    ULONG i3 = 0;
    ULONG i4 = 0;
    ULONG i5 = 0;
    ULONG i6 = 0;
    ULONG i7 = 0;
    ULONG i8 = 0;

    result = jitInterface->Vtbl->Index32x8(jitInterface,
                                           keys[i],
                                           keys[i + 1],
                                           keys[i + 2],
                                           keys[i + 3],
                                           keys[i + 4],
                                           keys[i + 5],
                                           keys[i + 6],
                                           keys[i + 7],
                                           &i1,
                                           &i2,
                                           &i3,
                                           &i4,
                                           &i5,
                                           &i6,
                                           &i7,
                                           &i8);
    ASSERT_GE(result, 0);

    EXPECT_EQ(expected[i], i1);
    EXPECT_EQ(expected[i + 1], i2);
    EXPECT_EQ(expected[i + 2], i3);
    EXPECT_EQ(expected[i + 3], i4);
    EXPECT_EQ(expected[i + 4], i5);
    EXPECT_EQ(expected[i + 5], i6);
    EXPECT_EQ(expected[i + 6], i7);
    EXPECT_EQ(expected[i + 7], i8);
  }

  jitInterface->Vtbl->Release(jitInterface);
  shim->Vtbl->Release(table);
}

TEST_F(PerfectHashOnlineTests,
       RawDogJitMulshrolate2RXIndex32x16Assigned16MatchesIndex) {
  const std::string keysPath =
      std::string(PERFECTHASH_TEST_ROOT_DIR) +
      "/keys/HologramWorld-31016.keys";
  const std::string seedsPath =
      std::string(PERFECTHASH_TEST_ROOT_DIR) +
      "/keys/HologramWorld-31016.Mulshrolate2RX.seeds";
  std::vector<ULONG> keys;
  std::vector<ULONG> seeds;
  if (!LoadKeysFromFile(keysPath, &keys)) {
    GTEST_SKIP() << "Keys file unavailable: " << keysPath;
  }
  if (!LoadSeedsFromFile(seedsPath, &seeds) || seeds.size() < 3) {
    GTEST_SKIP() << "Seeds file unavailable: " << seedsPath;
  }

  ScopedEnvVar rawdogVectorVersion("PH_RAWDOG_VECTOR_VERSION", "4");
  PPERFECT_HASH_TABLE table = CreateTableFromKeys(
      keys, PerfectHashHashMulshrolate2RXFunctionId, true, 3, &seeds);
  ASSERT_NE(table, nullptr);

  auto *shim = reinterpret_cast<PerfectHashTableShim *>(table);
  PERFECT_HASH_TABLE_FLAGS flags = {0};
  ASSERT_GE(shim->Vtbl->GetFlags(table, sizeof(flags), &flags), 0);
  ASSERT_EQ(flags.AssignedElementSizeInBits << 3, 16u);

  std::vector<ULONG> expected(keys.size());
  for (size_t i = 0; i < keys.size(); ++i) {
    HRESULT result = shim->Vtbl->Index(table, keys[i], &expected[i]);
    ASSERT_GE(result, 0);
  }

  PERFECT_HASH_TABLE_COMPILE_FLAGS compileFlags = {0};
  compileFlags.Jit = TRUE;
  compileFlags.JitBackendRawDog = TRUE;
  compileFlags.JitIndex32x16 = TRUE;

  HRESULT result = online_->Vtbl->CompileTable(online_, table, &compileFlags);
  if (result == PH_E_NOT_IMPLEMENTED) {
    shim->Vtbl->Release(table);
    GTEST_SKIP() << "RawDog Index32x16 unavailable on this host.";
  }
  ASSERT_GE(result, 0);

  PPERFECT_HASH_TABLE_JIT_INTERFACE jitInterface = nullptr;
  result = shim->Vtbl->QueryInterface(
      table,
#ifdef PH_WINDOWS
      IID_PERFECT_HASH_TABLE_JIT_INTERFACE,
#else
      &IID_PERFECT_HASH_TABLE_JIT_INTERFACE,
#endif
      reinterpret_cast<void **>(&jitInterface));
  ASSERT_GE(result, 0);
  ASSERT_NE(jitInterface, nullptr);

  const size_t limit = keys.size() - (keys.size() % 16);
  for (size_t i = 0; i < limit; i += 16) {
    ULONG index[16] = {};

    result = jitInterface->Vtbl->Index32x16(jitInterface,
                                            keys[i],
                                            keys[i + 1],
                                            keys[i + 2],
                                            keys[i + 3],
                                            keys[i + 4],
                                            keys[i + 5],
                                            keys[i + 6],
                                            keys[i + 7],
                                            keys[i + 8],
                                            keys[i + 9],
                                            keys[i + 10],
                                            keys[i + 11],
                                            keys[i + 12],
                                            keys[i + 13],
                                            keys[i + 14],
                                            keys[i + 15],
                                            &index[0],
                                            &index[1],
                                            &index[2],
                                            &index[3],
                                            &index[4],
                                            &index[5],
                                            &index[6],
                                            &index[7],
                                            &index[8],
                                            &index[9],
                                            &index[10],
                                            &index[11],
                                            &index[12],
                                            &index[13],
                                            &index[14],
                                            &index[15]);
    ASSERT_GE(result, 0);

    for (size_t lane = 0; lane < 16; ++lane) {
      EXPECT_EQ(expected[i + lane], index[lane]);
    }
  }

  jitInterface->Vtbl->Release(jitInterface);
  shim->Vtbl->Release(table);
}

TEST_F(PerfectHashOnlineTests,
       RawDogJitMulshrolate3RXIndex32x8Assigned16MatchesIndex) {
  const std::string keysPath =
      std::string(PERFECTHASH_TEST_ROOT_DIR) +
      "/keys/HologramWorld-31016.keys";
  const std::string seedsPath =
      std::string(PERFECTHASH_TEST_ROOT_DIR) +
      "/keys/HologramWorld-31016.Mulshrolate3RX.seeds";
  std::vector<ULONG> keys;
  std::vector<ULONG> seeds;
  if (!LoadKeysFromFile(keysPath, &keys)) {
    GTEST_SKIP() << "Keys file unavailable: " << keysPath;
  }
  if (!LoadSeedsFromFile(seedsPath, &seeds) || seeds.size() < 4) {
    GTEST_SKIP() << "Seeds file unavailable: " << seedsPath;
  }

  ScopedEnvVar rawdogVectorVersion("PH_RAWDOG_VECTOR_VERSION", "4");
  PPERFECT_HASH_TABLE table = CreateTableFromKeys(
      keys, PerfectHashHashMulshrolate3RXFunctionId, true, 3, &seeds);
  ASSERT_NE(table, nullptr);

  auto *shim = reinterpret_cast<PerfectHashTableShim *>(table);
  PERFECT_HASH_TABLE_FLAGS flags = {0};
  ASSERT_GE(shim->Vtbl->GetFlags(table, sizeof(flags), &flags), 0);
  ASSERT_EQ(flags.AssignedElementSizeInBits << 3, 16u);

  std::vector<ULONG> expected(keys.size());
  for (size_t i = 0; i < keys.size(); ++i) {
    HRESULT result = shim->Vtbl->Index(table, keys[i], &expected[i]);
    ASSERT_GE(result, 0);
  }

  PERFECT_HASH_TABLE_COMPILE_FLAGS compileFlags = {0};
  compileFlags.Jit = TRUE;
  compileFlags.JitBackendRawDog = TRUE;
  compileFlags.JitIndex32x8 = TRUE;

  HRESULT result = online_->Vtbl->CompileTable(online_, table, &compileFlags);
  if (result == PH_E_NOT_IMPLEMENTED) {
    shim->Vtbl->Release(table);
    GTEST_SKIP() << "RawDog Index32x8 unavailable on this host.";
  }
  ASSERT_GE(result, 0);

  PPERFECT_HASH_TABLE_JIT_INTERFACE jitInterface = nullptr;
  result = shim->Vtbl->QueryInterface(
      table,
#ifdef PH_WINDOWS
      IID_PERFECT_HASH_TABLE_JIT_INTERFACE,
#else
      &IID_PERFECT_HASH_TABLE_JIT_INTERFACE,
#endif
      reinterpret_cast<void **>(&jitInterface));
  ASSERT_GE(result, 0);
  ASSERT_NE(jitInterface, nullptr);

  const size_t limit = keys.size() - (keys.size() % 8);
  for (size_t i = 0; i < limit; i += 8) {
    ULONG i1 = 0;
    ULONG i2 = 0;
    ULONG i3 = 0;
    ULONG i4 = 0;
    ULONG i5 = 0;
    ULONG i6 = 0;
    ULONG i7 = 0;
    ULONG i8 = 0;

    result = jitInterface->Vtbl->Index32x8(jitInterface,
                                           keys[i],
                                           keys[i + 1],
                                           keys[i + 2],
                                           keys[i + 3],
                                           keys[i + 4],
                                           keys[i + 5],
                                           keys[i + 6],
                                           keys[i + 7],
                                           &i1,
                                           &i2,
                                           &i3,
                                           &i4,
                                           &i5,
                                           &i6,
                                           &i7,
                                           &i8);
    ASSERT_GE(result, 0);

    EXPECT_EQ(expected[i], i1);
    EXPECT_EQ(expected[i + 1], i2);
    EXPECT_EQ(expected[i + 2], i3);
    EXPECT_EQ(expected[i + 3], i4);
    EXPECT_EQ(expected[i + 4], i5);
    EXPECT_EQ(expected[i + 5], i6);
    EXPECT_EQ(expected[i + 6], i7);
    EXPECT_EQ(expected[i + 7], i8);
  }

  jitInterface->Vtbl->Release(jitInterface);
  shim->Vtbl->Release(table);
}

TEST_F(PerfectHashOnlineTests,
       RawDogJitMulshrolate3RXIndex32x16Assigned16MatchesIndex) {
  const std::string keysPath =
      std::string(PERFECTHASH_TEST_ROOT_DIR) +
      "/keys/HologramWorld-31016.keys";
  const std::string seedsPath =
      std::string(PERFECTHASH_TEST_ROOT_DIR) +
      "/keys/HologramWorld-31016.Mulshrolate3RX.seeds";
  std::vector<ULONG> keys;
  std::vector<ULONG> seeds;
  if (!LoadKeysFromFile(keysPath, &keys)) {
    GTEST_SKIP() << "Keys file unavailable: " << keysPath;
  }
  if (!LoadSeedsFromFile(seedsPath, &seeds) || seeds.size() < 4) {
    GTEST_SKIP() << "Seeds file unavailable: " << seedsPath;
  }

  ScopedEnvVar rawdogVectorVersion("PH_RAWDOG_VECTOR_VERSION", "4");
  PPERFECT_HASH_TABLE table = CreateTableFromKeys(
      keys, PerfectHashHashMulshrolate3RXFunctionId, true, 3, &seeds);
  ASSERT_NE(table, nullptr);

  auto *shim = reinterpret_cast<PerfectHashTableShim *>(table);
  PERFECT_HASH_TABLE_FLAGS flags = {0};
  ASSERT_GE(shim->Vtbl->GetFlags(table, sizeof(flags), &flags), 0);
  ASSERT_EQ(flags.AssignedElementSizeInBits << 3, 16u);

  std::vector<ULONG> expected(keys.size());
  for (size_t i = 0; i < keys.size(); ++i) {
    HRESULT result = shim->Vtbl->Index(table, keys[i], &expected[i]);
    ASSERT_GE(result, 0);
  }

  PERFECT_HASH_TABLE_COMPILE_FLAGS compileFlags = {0};
  compileFlags.Jit = TRUE;
  compileFlags.JitBackendRawDog = TRUE;
  compileFlags.JitIndex32x16 = TRUE;

  HRESULT result = online_->Vtbl->CompileTable(online_, table, &compileFlags);
  if (result == PH_E_NOT_IMPLEMENTED) {
    shim->Vtbl->Release(table);
    GTEST_SKIP() << "RawDog Index32x16 unavailable on this host.";
  }
  ASSERT_GE(result, 0);

  PPERFECT_HASH_TABLE_JIT_INTERFACE jitInterface = nullptr;
  result = shim->Vtbl->QueryInterface(
      table,
#ifdef PH_WINDOWS
      IID_PERFECT_HASH_TABLE_JIT_INTERFACE,
#else
      &IID_PERFECT_HASH_TABLE_JIT_INTERFACE,
#endif
      reinterpret_cast<void **>(&jitInterface));
  ASSERT_GE(result, 0);
  ASSERT_NE(jitInterface, nullptr);

  const size_t limit = keys.size() - (keys.size() % 16);
  for (size_t i = 0; i < limit; i += 16) {
    ULONG index[16] = {};

    result = jitInterface->Vtbl->Index32x16(jitInterface,
                                            keys[i],
                                            keys[i + 1],
                                            keys[i + 2],
                                            keys[i + 3],
                                            keys[i + 4],
                                            keys[i + 5],
                                            keys[i + 6],
                                            keys[i + 7],
                                            keys[i + 8],
                                            keys[i + 9],
                                            keys[i + 10],
                                            keys[i + 11],
                                            keys[i + 12],
                                            keys[i + 13],
                                            keys[i + 14],
                                            keys[i + 15],
                                            &index[0],
                                            &index[1],
                                            &index[2],
                                            &index[3],
                                            &index[4],
                                            &index[5],
                                            &index[6],
                                            &index[7],
                                            &index[8],
                                            &index[9],
                                            &index[10],
                                            &index[11],
                                            &index[12],
                                            &index[13],
                                            &index[14],
                                            &index[15]);
    ASSERT_GE(result, 0);

    for (size_t lane = 0; lane < 16; ++lane) {
      EXPECT_EQ(expected[i + lane], index[lane]);
    }
  }

  jitInterface->Vtbl->Release(jitInterface);
  shim->Vtbl->Release(table);
}

TEST_F(PerfectHashOnlineTests,
       RawDogJitMulshrolate4RXIndex32x8Assigned16MatchesIndex) {
  const std::string keysPath =
      std::string(PERFECTHASH_TEST_ROOT_DIR) +
      "/keys/HologramWorld-31016.keys";
  const std::string seedsPath =
      std::string(PERFECTHASH_TEST_ROOT_DIR) +
      "/keys/HologramWorld-31016.Mulshrolate4RX.seeds";
  std::vector<ULONG> keys;
  std::vector<ULONG> seeds;
  if (!LoadKeysFromFile(keysPath, &keys)) {
    GTEST_SKIP() << "Keys file unavailable: " << keysPath;
  }
  if (!LoadSeedsFromFile(seedsPath, &seeds) || seeds.size() < 5) {
    GTEST_SKIP() << "Seeds file unavailable: " << seedsPath;
  }

  ScopedEnvVar rawdogVectorVersion("PH_RAWDOG_VECTOR_VERSION", "4");
  PPERFECT_HASH_TABLE table = CreateTableFromKeys(
      keys, PerfectHashHashMulshrolate4RXFunctionId, true, 3, &seeds);
  ASSERT_NE(table, nullptr);

  auto *shim = reinterpret_cast<PerfectHashTableShim *>(table);
  PERFECT_HASH_TABLE_FLAGS flags = {0};
  ASSERT_GE(shim->Vtbl->GetFlags(table, sizeof(flags), &flags), 0);
  ASSERT_EQ(flags.AssignedElementSizeInBits << 3, 16u);

  std::vector<ULONG> expected(keys.size());
  for (size_t i = 0; i < keys.size(); ++i) {
    HRESULT result = shim->Vtbl->Index(table, keys[i], &expected[i]);
    ASSERT_GE(result, 0);
  }

  PERFECT_HASH_TABLE_COMPILE_FLAGS compileFlags = {0};
  compileFlags.Jit = TRUE;
  compileFlags.JitBackendRawDog = TRUE;
  compileFlags.JitIndex32x8 = TRUE;

  HRESULT result = online_->Vtbl->CompileTable(online_, table, &compileFlags);
  if (result == PH_E_NOT_IMPLEMENTED) {
    shim->Vtbl->Release(table);
    GTEST_SKIP() << "RawDog Index32x8 unavailable on this host.";
  }
  ASSERT_GE(result, 0);

  PPERFECT_HASH_TABLE_JIT_INTERFACE jitInterface = nullptr;
  result = shim->Vtbl->QueryInterface(
      table,
#ifdef PH_WINDOWS
      IID_PERFECT_HASH_TABLE_JIT_INTERFACE,
#else
      &IID_PERFECT_HASH_TABLE_JIT_INTERFACE,
#endif
      reinterpret_cast<void **>(&jitInterface));
  ASSERT_GE(result, 0);
  ASSERT_NE(jitInterface, nullptr);

  const size_t limit = keys.size() - (keys.size() % 8);
  for (size_t i = 0; i < limit; i += 8) {
    ULONG i1 = 0;
    ULONG i2 = 0;
    ULONG i3 = 0;
    ULONG i4 = 0;
    ULONG i5 = 0;
    ULONG i6 = 0;
    ULONG i7 = 0;
    ULONG i8 = 0;

    result = jitInterface->Vtbl->Index32x8(jitInterface,
                                           keys[i],
                                           keys[i + 1],
                                           keys[i + 2],
                                           keys[i + 3],
                                           keys[i + 4],
                                           keys[i + 5],
                                           keys[i + 6],
                                           keys[i + 7],
                                           &i1,
                                           &i2,
                                           &i3,
                                           &i4,
                                           &i5,
                                           &i6,
                                           &i7,
                                           &i8);
    ASSERT_GE(result, 0);

    EXPECT_EQ(expected[i], i1);
    EXPECT_EQ(expected[i + 1], i2);
    EXPECT_EQ(expected[i + 2], i3);
    EXPECT_EQ(expected[i + 3], i4);
    EXPECT_EQ(expected[i + 4], i5);
    EXPECT_EQ(expected[i + 5], i6);
    EXPECT_EQ(expected[i + 6], i7);
    EXPECT_EQ(expected[i + 7], i8);
  }

  jitInterface->Vtbl->Release(jitInterface);
  shim->Vtbl->Release(table);
}

TEST_F(PerfectHashOnlineTests,
       RawDogJitMulshrolate4RXIndex32x16Assigned16MatchesIndex) {
  const std::string keysPath =
      std::string(PERFECTHASH_TEST_ROOT_DIR) +
      "/keys/HologramWorld-31016.keys";
  const std::string seedsPath =
      std::string(PERFECTHASH_TEST_ROOT_DIR) +
      "/keys/HologramWorld-31016.Mulshrolate4RX.seeds";
  std::vector<ULONG> keys;
  std::vector<ULONG> seeds;
  if (!LoadKeysFromFile(keysPath, &keys)) {
    GTEST_SKIP() << "Keys file unavailable: " << keysPath;
  }
  if (!LoadSeedsFromFile(seedsPath, &seeds) || seeds.size() < 5) {
    GTEST_SKIP() << "Seeds file unavailable: " << seedsPath;
  }

  ScopedEnvVar rawdogVectorVersion("PH_RAWDOG_VECTOR_VERSION", "4");
  PPERFECT_HASH_TABLE table = CreateTableFromKeys(
      keys, PerfectHashHashMulshrolate4RXFunctionId, true, 3, &seeds);
  ASSERT_NE(table, nullptr);

  auto *shim = reinterpret_cast<PerfectHashTableShim *>(table);
  PERFECT_HASH_TABLE_FLAGS flags = {0};
  ASSERT_GE(shim->Vtbl->GetFlags(table, sizeof(flags), &flags), 0);
  ASSERT_EQ(flags.AssignedElementSizeInBits << 3, 16u);

  std::vector<ULONG> expected(keys.size());
  for (size_t i = 0; i < keys.size(); ++i) {
    HRESULT result = shim->Vtbl->Index(table, keys[i], &expected[i]);
    ASSERT_GE(result, 0);
  }

  PERFECT_HASH_TABLE_COMPILE_FLAGS compileFlags = {0};
  compileFlags.Jit = TRUE;
  compileFlags.JitBackendRawDog = TRUE;
  compileFlags.JitIndex32x16 = TRUE;

  HRESULT result = online_->Vtbl->CompileTable(online_, table, &compileFlags);
  if (result == PH_E_NOT_IMPLEMENTED) {
    shim->Vtbl->Release(table);
    GTEST_SKIP() << "RawDog Index32x16 unavailable on this host.";
  }
  ASSERT_GE(result, 0);

  PPERFECT_HASH_TABLE_JIT_INTERFACE jitInterface = nullptr;
  result = shim->Vtbl->QueryInterface(
      table,
#ifdef PH_WINDOWS
      IID_PERFECT_HASH_TABLE_JIT_INTERFACE,
#else
      &IID_PERFECT_HASH_TABLE_JIT_INTERFACE,
#endif
      reinterpret_cast<void **>(&jitInterface));
  ASSERT_GE(result, 0);
  ASSERT_NE(jitInterface, nullptr);

  const size_t limit = keys.size() - (keys.size() % 16);
  for (size_t i = 0; i < limit; i += 16) {
    ULONG index[16] = {};

    result = jitInterface->Vtbl->Index32x16(jitInterface,
                                            keys[i],
                                            keys[i + 1],
                                            keys[i + 2],
                                            keys[i + 3],
                                            keys[i + 4],
                                            keys[i + 5],
                                            keys[i + 6],
                                            keys[i + 7],
                                            keys[i + 8],
                                            keys[i + 9],
                                            keys[i + 10],
                                            keys[i + 11],
                                            keys[i + 12],
                                            keys[i + 13],
                                            keys[i + 14],
                                            keys[i + 15],
                                            &index[0],
                                            &index[1],
                                            &index[2],
                                            &index[3],
                                            &index[4],
                                            &index[5],
                                            &index[6],
                                            &index[7],
                                            &index[8],
                                            &index[9],
                                            &index[10],
                                            &index[11],
                                            &index[12],
                                            &index[13],
                                            &index[14],
                                            &index[15]);
    ASSERT_GE(result, 0);

    for (size_t lane = 0; lane < 16; ++lane) {
      EXPECT_EQ(expected[i + lane], index[lane]);
    }
  }

  jitInterface->Vtbl->Release(jitInterface);
  shim->Vtbl->Release(table);
}

TEST_F(PerfectHashOnlineTests, RawDogJitMulshrolate1RXAssigned16MatchesIndex) {
  const std::string keysPath =
      std::string(PERFECTHASH_TEST_ROOT_DIR) +
      "/keys/HologramWorld-31016.keys";
  const std::string seedsPath =
      std::string(PERFECTHASH_TEST_ROOT_DIR) +
      "/keys/HologramWorld-31016.Mulshrolate1RX.seeds";
  std::vector<ULONG> keys;
  std::vector<ULONG> seeds;
  if (!LoadKeysFromFile(keysPath, &keys)) {
    GTEST_SKIP() << "Keys file unavailable: " << keysPath;
  }
  if (!LoadSeedsFromFile(seedsPath, &seeds) || seeds.size() < 3) {
    GTEST_SKIP() << "Seeds file unavailable: " << seedsPath;
  }

  PPERFECT_HASH_TABLE table = CreateTableFromKeys(
      keys, PerfectHashHashMulshrolate1RXFunctionId, true, 3, &seeds);
  ASSERT_NE(table, nullptr);

  auto *shim = reinterpret_cast<PerfectHashTableShim *>(table);
  PERFECT_HASH_TABLE_FLAGS flags = {0};
  ASSERT_GE(shim->Vtbl->GetFlags(table, sizeof(flags), &flags), 0);
  ASSERT_EQ(flags.AssignedElementSizeInBits << 3, 16u);

  std::vector<ULONG> expected(keys.size());
  for (size_t i = 0; i < keys.size(); ++i) {
    HRESULT result = shim->Vtbl->Index(table, keys[i], &expected[i]);
    ASSERT_GE(result, 0);
  }

  PERFECT_HASH_TABLE_COMPILE_FLAGS compileFlags = {0};
  compileFlags.Jit = TRUE;
  compileFlags.JitBackendRawDog = TRUE;

  HRESULT result = online_->Vtbl->CompileTable(online_, table, &compileFlags);
  if (result == PH_E_NOT_IMPLEMENTED) {
    shim->Vtbl->Release(table);
    GTEST_SKIP() << "RawDog Mulshrolate1RX unavailable on this host.";
  }
  ASSERT_GE(result, 0);

  for (size_t i = 0; i < keys.size(); ++i) {
    ULONG index = 0;
    result = shim->Vtbl->Index(table, keys[i], &index);
    ASSERT_GE(result, 0);
    EXPECT_EQ(expected[i], index);
  }

  shim->Vtbl->Release(table);
}

TEST_F(PerfectHashOnlineTests,
       RawDogJitMulshrolate1RXIndex32x4Assigned16MatchesIndex) {
  const std::string keysPath =
      std::string(PERFECTHASH_TEST_ROOT_DIR) +
      "/keys/HologramWorld-31016.keys";
  const std::string seedsPath =
      std::string(PERFECTHASH_TEST_ROOT_DIR) +
      "/keys/HologramWorld-31016.Mulshrolate1RX.seeds";
  std::vector<ULONG> keys;
  std::vector<ULONG> seeds;
  if (!LoadKeysFromFile(keysPath, &keys)) {
    GTEST_SKIP() << "Keys file unavailable: " << keysPath;
  }
  if (!LoadSeedsFromFile(seedsPath, &seeds) || seeds.size() < 3) {
    GTEST_SKIP() << "Seeds file unavailable: " << seedsPath;
  }

  PPERFECT_HASH_TABLE table = CreateTableFromKeys(
      keys, PerfectHashHashMulshrolate1RXFunctionId, true, 3, &seeds);
  ASSERT_NE(table, nullptr);

  auto *shim = reinterpret_cast<PerfectHashTableShim *>(table);
  PERFECT_HASH_TABLE_FLAGS flags = {0};
  ASSERT_GE(shim->Vtbl->GetFlags(table, sizeof(flags), &flags), 0);
  ASSERT_EQ(flags.AssignedElementSizeInBits << 3, 16u);

  std::vector<ULONG> expected(keys.size());
  for (size_t i = 0; i < keys.size(); ++i) {
    HRESULT result = shim->Vtbl->Index(table, keys[i], &expected[i]);
    ASSERT_GE(result, 0);
  }

  PERFECT_HASH_TABLE_COMPILE_FLAGS compileFlags = {0};
  compileFlags.Jit = TRUE;
  compileFlags.JitBackendRawDog = TRUE;
  compileFlags.JitIndex32x4 = TRUE;

  HRESULT result = online_->Vtbl->CompileTable(online_, table, &compileFlags);
  if (result == PH_E_NOT_IMPLEMENTED) {
    shim->Vtbl->Release(table);
    GTEST_SKIP() << "RawDog Index32x4 unavailable on this host.";
  }
  ASSERT_GE(result, 0);

  PPERFECT_HASH_TABLE_JIT_INTERFACE jitInterface = nullptr;
  result = shim->Vtbl->QueryInterface(
      table,
#ifdef PH_WINDOWS
      IID_PERFECT_HASH_TABLE_JIT_INTERFACE,
#else
      &IID_PERFECT_HASH_TABLE_JIT_INTERFACE,
#endif
      reinterpret_cast<void **>(&jitInterface));
  ASSERT_GE(result, 0);
  ASSERT_NE(jitInterface, nullptr);

  const size_t limit = keys.size() - (keys.size() % 4);
  for (size_t i = 0; i < limit; i += 4) {
    ULONG i1 = 0;
    ULONG i2 = 0;
    ULONG i3 = 0;
    ULONG i4 = 0;

    result = jitInterface->Vtbl->Index32x4(jitInterface,
                                           keys[i],
                                           keys[i + 1],
                                           keys[i + 2],
                                           keys[i + 3],
                                           &i1,
                                           &i2,
                                           &i3,
                                           &i4);
    ASSERT_GE(result, 0);

    EXPECT_EQ(expected[i], i1);
    EXPECT_EQ(expected[i + 1], i2);
    EXPECT_EQ(expected[i + 2], i3);
    EXPECT_EQ(expected[i + 3], i4);
  }

  jitInterface->Vtbl->Release(jitInterface);
  shim->Vtbl->Release(table);
}

TEST_F(PerfectHashOnlineTests,
       RawDogJitMulshrolate1RXIndex32x8Assigned16MatchesIndex) {
  const std::string keysPath =
      std::string(PERFECTHASH_TEST_ROOT_DIR) +
      "/keys/HologramWorld-31016.keys";
  const std::string seedsPath =
      std::string(PERFECTHASH_TEST_ROOT_DIR) +
      "/keys/HologramWorld-31016.Mulshrolate1RX.seeds";
  std::vector<ULONG> keys;
  std::vector<ULONG> seeds;
  if (!LoadKeysFromFile(keysPath, &keys)) {
    GTEST_SKIP() << "Keys file unavailable: " << keysPath;
  }
  if (!LoadSeedsFromFile(seedsPath, &seeds) || seeds.size() < 3) {
    GTEST_SKIP() << "Seeds file unavailable: " << seedsPath;
  }

  PPERFECT_HASH_TABLE table = CreateTableFromKeys(
      keys, PerfectHashHashMulshrolate1RXFunctionId, true, 3, &seeds);
  ASSERT_NE(table, nullptr);

  auto *shim = reinterpret_cast<PerfectHashTableShim *>(table);
  PERFECT_HASH_TABLE_FLAGS flags = {0};
  ASSERT_GE(shim->Vtbl->GetFlags(table, sizeof(flags), &flags), 0);
  ASSERT_EQ(flags.AssignedElementSizeInBits << 3, 16u);

  std::vector<ULONG> expected(keys.size());
  for (size_t i = 0; i < keys.size(); ++i) {
    HRESULT result = shim->Vtbl->Index(table, keys[i], &expected[i]);
    ASSERT_GE(result, 0);
  }

  PERFECT_HASH_TABLE_COMPILE_FLAGS compileFlags = {0};
  compileFlags.Jit = TRUE;
  compileFlags.JitBackendRawDog = TRUE;
  compileFlags.JitIndex32x8 = TRUE;

  HRESULT result = online_->Vtbl->CompileTable(online_, table, &compileFlags);
  if (result == PH_E_NOT_IMPLEMENTED) {
    shim->Vtbl->Release(table);
    GTEST_SKIP() << "RawDog Index32x8 unavailable on this host.";
  }
  ASSERT_GE(result, 0);

  PPERFECT_HASH_TABLE_JIT_INTERFACE jitInterface = nullptr;
  result = shim->Vtbl->QueryInterface(
      table,
#ifdef PH_WINDOWS
      IID_PERFECT_HASH_TABLE_JIT_INTERFACE,
#else
      &IID_PERFECT_HASH_TABLE_JIT_INTERFACE,
#endif
      reinterpret_cast<void **>(&jitInterface));
  ASSERT_GE(result, 0);
  ASSERT_NE(jitInterface, nullptr);

  const size_t limit = keys.size() - (keys.size() % 8);
  for (size_t i = 0; i < limit; i += 8) {
    ULONG i1 = 0;
    ULONG i2 = 0;
    ULONG i3 = 0;
    ULONG i4 = 0;
    ULONG i5 = 0;
    ULONG i6 = 0;
    ULONG i7 = 0;
    ULONG i8 = 0;

    result = jitInterface->Vtbl->Index32x8(jitInterface,
                                           keys[i],
                                           keys[i + 1],
                                           keys[i + 2],
                                           keys[i + 3],
                                           keys[i + 4],
                                           keys[i + 5],
                                           keys[i + 6],
                                           keys[i + 7],
                                           &i1,
                                           &i2,
                                           &i3,
                                           &i4,
                                           &i5,
                                           &i6,
                                           &i7,
                                           &i8);
    ASSERT_GE(result, 0);

    EXPECT_EQ(expected[i], i1);
    EXPECT_EQ(expected[i + 1], i2);
    EXPECT_EQ(expected[i + 2], i3);
    EXPECT_EQ(expected[i + 3], i4);
    EXPECT_EQ(expected[i + 4], i5);
    EXPECT_EQ(expected[i + 5], i6);
    EXPECT_EQ(expected[i + 6], i7);
    EXPECT_EQ(expected[i + 7], i8);
  }

  jitInterface->Vtbl->Release(jitInterface);
  shim->Vtbl->Release(table);
}

TEST_F(PerfectHashOnlineTests,
       RawDogJitMulshrolate1RXIndex32x16Assigned16MatchesIndex) {
  const std::string keysPath =
      std::string(PERFECTHASH_TEST_ROOT_DIR) +
      "/keys/HologramWorld-31016.keys";
  const std::string seedsPath =
      std::string(PERFECTHASH_TEST_ROOT_DIR) +
      "/keys/HologramWorld-31016.Mulshrolate1RX.seeds";
  std::vector<ULONG> keys;
  std::vector<ULONG> seeds;
  if (!LoadKeysFromFile(keysPath, &keys)) {
    GTEST_SKIP() << "Keys file unavailable: " << keysPath;
  }
  if (!LoadSeedsFromFile(seedsPath, &seeds) || seeds.size() < 3) {
    GTEST_SKIP() << "Seeds file unavailable: " << seedsPath;
  }

  PPERFECT_HASH_TABLE table = CreateTableFromKeys(
      keys, PerfectHashHashMulshrolate1RXFunctionId, true, 3, &seeds);
  ASSERT_NE(table, nullptr);

  auto *shim = reinterpret_cast<PerfectHashTableShim *>(table);
  PERFECT_HASH_TABLE_FLAGS flags = {0};
  ASSERT_GE(shim->Vtbl->GetFlags(table, sizeof(flags), &flags), 0);
  ASSERT_EQ(flags.AssignedElementSizeInBits << 3, 16u);

  std::vector<ULONG> expected(keys.size());
  for (size_t i = 0; i < keys.size(); ++i) {
    HRESULT result = shim->Vtbl->Index(table, keys[i], &expected[i]);
    ASSERT_GE(result, 0);
  }

  PERFECT_HASH_TABLE_COMPILE_FLAGS compileFlags = {0};
  compileFlags.Jit = TRUE;
  compileFlags.JitBackendRawDog = TRUE;
  compileFlags.JitIndex32x16 = TRUE;

  HRESULT result = online_->Vtbl->CompileTable(online_, table, &compileFlags);
  if (result == PH_E_NOT_IMPLEMENTED) {
    shim->Vtbl->Release(table);
    GTEST_SKIP() << "RawDog Index32x16 unavailable on this host.";
  }
  ASSERT_GE(result, 0);

  PPERFECT_HASH_TABLE_JIT_INTERFACE jitInterface = nullptr;
  result = shim->Vtbl->QueryInterface(
      table,
#ifdef PH_WINDOWS
      IID_PERFECT_HASH_TABLE_JIT_INTERFACE,
#else
      &IID_PERFECT_HASH_TABLE_JIT_INTERFACE,
#endif
      reinterpret_cast<void **>(&jitInterface));
  ASSERT_GE(result, 0);
  ASSERT_NE(jitInterface, nullptr);

  const size_t limit = keys.size() - (keys.size() % 16);
  for (size_t i = 0; i < limit; i += 16) {
    ULONG index[16] = {};

    result = jitInterface->Vtbl->Index32x16(jitInterface,
                                            keys[i],
                                            keys[i + 1],
                                            keys[i + 2],
                                            keys[i + 3],
                                            keys[i + 4],
                                            keys[i + 5],
                                            keys[i + 6],
                                            keys[i + 7],
                                            keys[i + 8],
                                            keys[i + 9],
                                            keys[i + 10],
                                            keys[i + 11],
                                            keys[i + 12],
                                            keys[i + 13],
                                            keys[i + 14],
                                            keys[i + 15],
                                            &index[0],
                                            &index[1],
                                            &index[2],
                                            &index[3],
                                            &index[4],
                                            &index[5],
                                            &index[6],
                                            &index[7],
                                            &index[8],
                                            &index[9],
                                            &index[10],
                                            &index[11],
                                            &index[12],
                                            &index[13],
                                            &index[14],
                                            &index[15]);
    ASSERT_GE(result, 0);

    for (size_t lane = 0; lane < 16; ++lane) {
      EXPECT_EQ(expected[i + lane], index[lane]);
    }
  }

  jitInterface->Vtbl->Release(jitInterface);
  shim->Vtbl->Release(table);
}

TEST_F(PerfectHashOnlineTests,
       RawDogJitMulshrolate1RXIndex32x16MatchesIndex) {
  const std::vector<ULONG> keys = {
      1, 3, 5, 7, 11, 13, 17, 19,
      23, 29, 31, 37, 41, 43, 47, 53,
      59, 61, 67, 71, 73, 79, 83, 89,
      97, 101, 103, 107, 109, 113, 127, 131,
  };

  PPERFECT_HASH_TABLE table =
      CreateTableFromKeys(keys, PerfectHashHashMulshrolate1RXFunctionId);
  ASSERT_NE(table, nullptr);

  auto *shim = reinterpret_cast<PerfectHashTableShim *>(table);

  std::vector<ULONG> expected(keys.size());
  for (size_t i = 0; i < keys.size(); ++i) {
    HRESULT result = shim->Vtbl->Index(table, keys[i], &expected[i]);
    ASSERT_GE(result, 0);
  }

  PERFECT_HASH_TABLE_COMPILE_FLAGS compileFlags = {0};
  compileFlags.Jit = TRUE;
  compileFlags.JitBackendRawDog = TRUE;
  compileFlags.JitIndex32x16 = TRUE;

  HRESULT result = online_->Vtbl->CompileTable(online_, table, &compileFlags);
  if (result == PH_E_NOT_IMPLEMENTED) {
    shim->Vtbl->Release(table);
    GTEST_SKIP() << "RawDog Index32x16 unavailable on this host.";
  }
  ASSERT_GE(result, 0);

  PPERFECT_HASH_TABLE_JIT_INTERFACE jitInterface = nullptr;
  result = shim->Vtbl->QueryInterface(
      table,
#ifdef PH_WINDOWS
      IID_PERFECT_HASH_TABLE_JIT_INTERFACE,
#else
      &IID_PERFECT_HASH_TABLE_JIT_INTERFACE,
#endif
      reinterpret_cast<void **>(&jitInterface));
  ASSERT_GE(result, 0);
  ASSERT_NE(jitInterface, nullptr);

  for (size_t i = 0; i < keys.size(); i += 16) {
    ULONG i1 = 0;
    ULONG i2 = 0;
    ULONG i3 = 0;
    ULONG i4 = 0;
    ULONG i5 = 0;
    ULONG i6 = 0;
    ULONG i7 = 0;
    ULONG i8 = 0;
    ULONG i9 = 0;
    ULONG i10 = 0;
    ULONG i11 = 0;
    ULONG i12 = 0;
    ULONG i13 = 0;
    ULONG i14 = 0;
    ULONG i15 = 0;
    ULONG i16 = 0;

    result = jitInterface->Vtbl->Index32x16(jitInterface,
                                            keys[i],
                                            keys[i + 1],
                                            keys[i + 2],
                                            keys[i + 3],
                                            keys[i + 4],
                                            keys[i + 5],
                                            keys[i + 6],
                                            keys[i + 7],
                                            keys[i + 8],
                                            keys[i + 9],
                                            keys[i + 10],
                                            keys[i + 11],
                                            keys[i + 12],
                                            keys[i + 13],
                                            keys[i + 14],
                                            keys[i + 15],
                                            &i1,
                                            &i2,
                                            &i3,
                                            &i4,
                                            &i5,
                                            &i6,
                                            &i7,
                                            &i8,
                                            &i9,
                                            &i10,
                                            &i11,
                                            &i12,
                                            &i13,
                                            &i14,
                                            &i15,
                                            &i16);
    ASSERT_GE(result, 0);

    EXPECT_EQ(expected[i], i1);
    EXPECT_EQ(expected[i + 1], i2);
    EXPECT_EQ(expected[i + 2], i3);
    EXPECT_EQ(expected[i + 3], i4);
    EXPECT_EQ(expected[i + 4], i5);
    EXPECT_EQ(expected[i + 5], i6);
    EXPECT_EQ(expected[i + 6], i7);
    EXPECT_EQ(expected[i + 7], i8);
    EXPECT_EQ(expected[i + 8], i9);
    EXPECT_EQ(expected[i + 9], i10);
    EXPECT_EQ(expected[i + 10], i11);
    EXPECT_EQ(expected[i + 11], i12);
    EXPECT_EQ(expected[i + 12], i13);
    EXPECT_EQ(expected[i + 13], i14);
    EXPECT_EQ(expected[i + 14], i15);
    EXPECT_EQ(expected[i + 15], i16);
  }

  jitInterface->Vtbl->Release(jitInterface);
  shim->Vtbl->Release(table);
}

TEST_F(PerfectHashOnlineTests,
       RawDogJitMulshrolate2RXIndex32x4MatchesIndex) {
  const std::vector<ULONG> keys = {
      1, 3, 5, 7, 11, 13, 17, 19,
      23, 29, 31, 37, 41, 43, 47, 53,
      59, 61, 67, 71, 73, 79, 83, 89,
      97, 101, 103, 107, 109, 113, 127, 131,
  };

  PPERFECT_HASH_TABLE table =
      CreateTableFromKeys(keys, PerfectHashHashMulshrolate2RXFunctionId);
  ASSERT_NE(table, nullptr);

  auto *shim = reinterpret_cast<PerfectHashTableShim *>(table);

  std::vector<ULONG> expected(keys.size());
  for (size_t i = 0; i < keys.size(); ++i) {
    HRESULT result = shim->Vtbl->Index(table, keys[i], &expected[i]);
    ASSERT_GE(result, 0);
  }

  PERFECT_HASH_TABLE_COMPILE_FLAGS compileFlags = {0};
  compileFlags.Jit = TRUE;
  compileFlags.JitBackendRawDog = TRUE;
  compileFlags.JitIndex32x4 = TRUE;

  HRESULT result = online_->Vtbl->CompileTable(online_, table, &compileFlags);
  if (result == PH_E_NOT_IMPLEMENTED) {
    shim->Vtbl->Release(table);
    GTEST_SKIP() << "RawDog Index32x4 unavailable on this host.";
  }
  ASSERT_GE(result, 0);

  PPERFECT_HASH_TABLE_JIT_INTERFACE jitInterface = nullptr;
  result = shim->Vtbl->QueryInterface(
      table,
#ifdef PH_WINDOWS
      IID_PERFECT_HASH_TABLE_JIT_INTERFACE,
#else
      &IID_PERFECT_HASH_TABLE_JIT_INTERFACE,
#endif
      reinterpret_cast<void **>(&jitInterface));
  ASSERT_GE(result, 0);
  ASSERT_NE(jitInterface, nullptr);

  for (size_t i = 0; i < keys.size(); i += 4) {
    ULONG i1 = 0;
    ULONG i2 = 0;
    ULONG i3 = 0;
    ULONG i4 = 0;

    result = jitInterface->Vtbl->Index32x4(jitInterface,
                                           keys[i],
                                           keys[i + 1],
                                           keys[i + 2],
                                           keys[i + 3],
                                           &i1,
                                           &i2,
                                           &i3,
                                           &i4);
    ASSERT_GE(result, 0);

    EXPECT_EQ(expected[i], i1);
    EXPECT_EQ(expected[i + 1], i2);
    EXPECT_EQ(expected[i + 2], i3);
    EXPECT_EQ(expected[i + 3], i4);
  }

  jitInterface->Vtbl->Release(jitInterface);
  shim->Vtbl->Release(table);
}

TEST_F(PerfectHashOnlineTests,
       RawDogJitMulshrolate2RXIndex32x8MatchesIndex) {
  const std::vector<ULONG> keys = {
      1, 3, 5, 7, 11, 13, 17, 19,
      23, 29, 31, 37, 41, 43, 47, 53,
      59, 61, 67, 71, 73, 79, 83, 89,
      97, 101, 103, 107, 109, 113, 127, 131,
  };

  PPERFECT_HASH_TABLE table =
      CreateTableFromKeys(keys, PerfectHashHashMulshrolate2RXFunctionId);
  ASSERT_NE(table, nullptr);

  auto *shim = reinterpret_cast<PerfectHashTableShim *>(table);

  std::vector<ULONG> expected(keys.size());
  for (size_t i = 0; i < keys.size(); ++i) {
    HRESULT result = shim->Vtbl->Index(table, keys[i], &expected[i]);
    ASSERT_GE(result, 0);
  }

  PERFECT_HASH_TABLE_COMPILE_FLAGS compileFlags = {0};
  compileFlags.Jit = TRUE;
  compileFlags.JitBackendRawDog = TRUE;
  compileFlags.JitIndex32x8 = TRUE;

  HRESULT result = online_->Vtbl->CompileTable(online_, table, &compileFlags);
  if (result == PH_E_NOT_IMPLEMENTED) {
    shim->Vtbl->Release(table);
    GTEST_SKIP() << "RawDog AVX2 Index32x8 unavailable on this host.";
  }
  ASSERT_GE(result, 0);

  PPERFECT_HASH_TABLE_JIT_INTERFACE jitInterface = nullptr;
  result = shim->Vtbl->QueryInterface(
      table,
#ifdef PH_WINDOWS
      IID_PERFECT_HASH_TABLE_JIT_INTERFACE,
#else
      &IID_PERFECT_HASH_TABLE_JIT_INTERFACE,
#endif
      reinterpret_cast<void **>(&jitInterface));
  ASSERT_GE(result, 0);
  ASSERT_NE(jitInterface, nullptr);

  for (size_t i = 0; i < keys.size(); i += 8) {
    ULONG i1 = 0;
    ULONG i2 = 0;
    ULONG i3 = 0;
    ULONG i4 = 0;
    ULONG i5 = 0;
    ULONG i6 = 0;
    ULONG i7 = 0;
    ULONG i8 = 0;

    result = jitInterface->Vtbl->Index32x8(jitInterface,
                                           keys[i],
                                           keys[i + 1],
                                           keys[i + 2],
                                           keys[i + 3],
                                           keys[i + 4],
                                           keys[i + 5],
                                           keys[i + 6],
                                           keys[i + 7],
                                           &i1,
                                           &i2,
                                           &i3,
                                           &i4,
                                           &i5,
                                           &i6,
                                           &i7,
                                           &i8);
    ASSERT_GE(result, 0);

    EXPECT_EQ(expected[i], i1);
    EXPECT_EQ(expected[i + 1], i2);
    EXPECT_EQ(expected[i + 2], i3);
    EXPECT_EQ(expected[i + 3], i4);
    EXPECT_EQ(expected[i + 4], i5);
    EXPECT_EQ(expected[i + 5], i6);
    EXPECT_EQ(expected[i + 6], i7);
    EXPECT_EQ(expected[i + 7], i8);
  }

  jitInterface->Vtbl->Release(jitInterface);
  shim->Vtbl->Release(table);
}

TEST_F(PerfectHashOnlineTests,
       RawDogJitMulshrolate2RXIndex32x16MatchesIndex) {
  const std::vector<ULONG> keys = {
      1, 3, 5, 7, 11, 13, 17, 19,
      23, 29, 31, 37, 41, 43, 47, 53,
      59, 61, 67, 71, 73, 79, 83, 89,
      97, 101, 103, 107, 109, 113, 127, 131,
  };

  PPERFECT_HASH_TABLE table =
      CreateTableFromKeys(keys, PerfectHashHashMulshrolate2RXFunctionId);
  ASSERT_NE(table, nullptr);

  auto *shim = reinterpret_cast<PerfectHashTableShim *>(table);

  std::vector<ULONG> expected(keys.size());
  for (size_t i = 0; i < keys.size(); ++i) {
    HRESULT result = shim->Vtbl->Index(table, keys[i], &expected[i]);
    ASSERT_GE(result, 0);
  }

  PERFECT_HASH_TABLE_COMPILE_FLAGS compileFlags = {0};
  compileFlags.Jit = TRUE;
  compileFlags.JitBackendRawDog = TRUE;
  compileFlags.JitIndex32x16 = TRUE;

  HRESULT result = online_->Vtbl->CompileTable(online_, table, &compileFlags);
  if (result == PH_E_NOT_IMPLEMENTED) {
    shim->Vtbl->Release(table);
    GTEST_SKIP() << "RawDog AVX-512 Index32x16 unavailable on this host.";
  }
  ASSERT_GE(result, 0);

  PPERFECT_HASH_TABLE_JIT_INTERFACE jitInterface = nullptr;
  result = shim->Vtbl->QueryInterface(
      table,
#ifdef PH_WINDOWS
      IID_PERFECT_HASH_TABLE_JIT_INTERFACE,
#else
      &IID_PERFECT_HASH_TABLE_JIT_INTERFACE,
#endif
      reinterpret_cast<void **>(&jitInterface));
  ASSERT_GE(result, 0);
  ASSERT_NE(jitInterface, nullptr);

  for (size_t i = 0; i < keys.size(); i += 16) {
    ULONG i1 = 0;
    ULONG i2 = 0;
    ULONG i3 = 0;
    ULONG i4 = 0;
    ULONG i5 = 0;
    ULONG i6 = 0;
    ULONG i7 = 0;
    ULONG i8 = 0;
    ULONG i9 = 0;
    ULONG i10 = 0;
    ULONG i11 = 0;
    ULONG i12 = 0;
    ULONG i13 = 0;
    ULONG i14 = 0;
    ULONG i15 = 0;
    ULONG i16 = 0;

    result = jitInterface->Vtbl->Index32x16(jitInterface,
                                            keys[i],
                                            keys[i + 1],
                                            keys[i + 2],
                                            keys[i + 3],
                                            keys[i + 4],
                                            keys[i + 5],
                                            keys[i + 6],
                                            keys[i + 7],
                                            keys[i + 8],
                                            keys[i + 9],
                                            keys[i + 10],
                                            keys[i + 11],
                                            keys[i + 12],
                                            keys[i + 13],
                                            keys[i + 14],
                                            keys[i + 15],
                                            &i1,
                                            &i2,
                                            &i3,
                                            &i4,
                                            &i5,
                                            &i6,
                                            &i7,
                                            &i8,
                                            &i9,
                                            &i10,
                                            &i11,
                                            &i12,
                                            &i13,
                                            &i14,
                                            &i15,
                                            &i16);
    ASSERT_GE(result, 0);

    EXPECT_EQ(expected[i], i1);
    EXPECT_EQ(expected[i + 1], i2);
    EXPECT_EQ(expected[i + 2], i3);
    EXPECT_EQ(expected[i + 3], i4);
    EXPECT_EQ(expected[i + 4], i5);
    EXPECT_EQ(expected[i + 5], i6);
    EXPECT_EQ(expected[i + 6], i7);
    EXPECT_EQ(expected[i + 7], i8);
    EXPECT_EQ(expected[i + 8], i9);
    EXPECT_EQ(expected[i + 9], i10);
    EXPECT_EQ(expected[i + 10], i11);
    EXPECT_EQ(expected[i + 11], i12);
    EXPECT_EQ(expected[i + 12], i13);
    EXPECT_EQ(expected[i + 13], i14);
    EXPECT_EQ(expected[i + 14], i15);
    EXPECT_EQ(expected[i + 15], i16);
  }

  jitInterface->Vtbl->Release(jitInterface);
  shim->Vtbl->Release(table);
}

TEST_F(PerfectHashOnlineTests, RawDogJitMulshrolate3RXMatchesSlowIndex) {
  const std::vector<ULONG> keys = {
      1, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53,
      59, 61, 67, 71, 73, 79, 83, 89, 97, 101, 103, 107, 109, 113,
  };

  PPERFECT_HASH_TABLE table =
      CreateTableFromKeys(keys, PerfectHashHashMulshrolate3RXFunctionId);
  ASSERT_NE(table, nullptr);

  auto *shim = reinterpret_cast<PerfectHashTableShim *>(table);
  ASSERT_NE(shim->Vtbl->SlowIndex, nullptr);

  PERFECT_HASH_TABLE_COMPILE_FLAGS compileFlags = {0};
  compileFlags.Jit = TRUE;
  compileFlags.JitBackendRawDog = TRUE;

  HRESULT result = online_->Vtbl->CompileTable(online_, table, &compileFlags);
  if (result == PH_E_NOT_IMPLEMENTED) {
    shim->Vtbl->Release(table);
    GTEST_SKIP() << "RawDog Mulshrolate3RX unavailable on this host.";
  }
  ASSERT_GE(result, 0);

  for (ULONG key : keys) {
    ULONG slowIndex = 0;
    ULONG jitIndex = 0;

    result = shim->Vtbl->SlowIndex(table, key, &slowIndex);
    ASSERT_GE(result, 0);

    result = shim->Vtbl->Index(table, key, &jitIndex);
    ASSERT_GE(result, 0);

    EXPECT_EQ(slowIndex, jitIndex);
  }

  shim->Vtbl->Release(table);
}

TEST_F(PerfectHashOnlineTests,
       RawDogJitMulshrolate3RXIndex32x8MatchesIndex) {
  const std::vector<ULONG> keys = {
      1, 3, 5, 7, 11, 13, 17, 19,
      23, 29, 31, 37, 41, 43, 47, 53,
      59, 61, 67, 71, 73, 79, 83, 89,
      97, 101, 103, 107, 109, 113, 127, 131,
  };

  PPERFECT_HASH_TABLE table =
      CreateTableFromKeys(keys, PerfectHashHashMulshrolate3RXFunctionId);
  ASSERT_NE(table, nullptr);

  auto *shim = reinterpret_cast<PerfectHashTableShim *>(table);

  std::vector<ULONG> expected(keys.size());
  for (size_t i = 0; i < keys.size(); ++i) {
    HRESULT result = shim->Vtbl->Index(table, keys[i], &expected[i]);
    ASSERT_GE(result, 0);
  }

  PERFECT_HASH_TABLE_COMPILE_FLAGS compileFlags = {0};
  compileFlags.Jit = TRUE;
  compileFlags.JitBackendRawDog = TRUE;
  compileFlags.JitIndex32x8 = TRUE;

  HRESULT result = online_->Vtbl->CompileTable(online_, table, &compileFlags);
  if (result == PH_E_NOT_IMPLEMENTED) {
    shim->Vtbl->Release(table);
    GTEST_SKIP() << "RawDog AVX2 Index32x8 unavailable on this host.";
  }
  ASSERT_GE(result, 0);

  PPERFECT_HASH_TABLE_JIT_INTERFACE jitInterface = nullptr;
  result = shim->Vtbl->QueryInterface(
      table,
#ifdef PH_WINDOWS
      IID_PERFECT_HASH_TABLE_JIT_INTERFACE,
#else
      &IID_PERFECT_HASH_TABLE_JIT_INTERFACE,
#endif
      reinterpret_cast<void **>(&jitInterface));
  ASSERT_GE(result, 0);
  ASSERT_NE(jitInterface, nullptr);

  for (size_t i = 0; i < keys.size(); i += 8) {
    ULONG i1 = 0;
    ULONG i2 = 0;
    ULONG i3 = 0;
    ULONG i4 = 0;
    ULONG i5 = 0;
    ULONG i6 = 0;
    ULONG i7 = 0;
    ULONG i8 = 0;

    result = jitInterface->Vtbl->Index32x8(jitInterface,
                                           keys[i],
                                           keys[i + 1],
                                           keys[i + 2],
                                           keys[i + 3],
                                           keys[i + 4],
                                           keys[i + 5],
                                           keys[i + 6],
                                           keys[i + 7],
                                           &i1,
                                           &i2,
                                           &i3,
                                           &i4,
                                           &i5,
                                           &i6,
                                           &i7,
                                           &i8);
    ASSERT_GE(result, 0);

    EXPECT_EQ(expected[i], i1);
    EXPECT_EQ(expected[i + 1], i2);
    EXPECT_EQ(expected[i + 2], i3);
    EXPECT_EQ(expected[i + 3], i4);
    EXPECT_EQ(expected[i + 4], i5);
    EXPECT_EQ(expected[i + 5], i6);
    EXPECT_EQ(expected[i + 6], i7);
    EXPECT_EQ(expected[i + 7], i8);
  }

  jitInterface->Vtbl->Release(jitInterface);
  shim->Vtbl->Release(table);
}

TEST_F(PerfectHashOnlineTests,
       RawDogJitMulshrolate3RXIndex32x16MatchesIndex) {
  const std::vector<ULONG> keys = {
      1, 3, 5, 7, 11, 13, 17, 19,
      23, 29, 31, 37, 41, 43, 47, 53,
      59, 61, 67, 71, 73, 79, 83, 89,
      97, 101, 103, 107, 109, 113, 127, 131,
  };

  PPERFECT_HASH_TABLE table =
      CreateTableFromKeys(keys, PerfectHashHashMulshrolate3RXFunctionId);
  ASSERT_NE(table, nullptr);

  auto *shim = reinterpret_cast<PerfectHashTableShim *>(table);

  std::vector<ULONG> expected(keys.size());
  for (size_t i = 0; i < keys.size(); ++i) {
    HRESULT result = shim->Vtbl->Index(table, keys[i], &expected[i]);
    ASSERT_GE(result, 0);
  }

  PERFECT_HASH_TABLE_COMPILE_FLAGS compileFlags = {0};
  compileFlags.Jit = TRUE;
  compileFlags.JitBackendRawDog = TRUE;
  compileFlags.JitIndex32x16 = TRUE;

  HRESULT result = online_->Vtbl->CompileTable(online_, table, &compileFlags);
  if (result == PH_E_NOT_IMPLEMENTED) {
    shim->Vtbl->Release(table);
    GTEST_SKIP() << "RawDog AVX-512 Index32x16 unavailable on this host.";
  }
  ASSERT_GE(result, 0);

  PPERFECT_HASH_TABLE_JIT_INTERFACE jitInterface = nullptr;
  result = shim->Vtbl->QueryInterface(
      table,
#ifdef PH_WINDOWS
      IID_PERFECT_HASH_TABLE_JIT_INTERFACE,
#else
      &IID_PERFECT_HASH_TABLE_JIT_INTERFACE,
#endif
      reinterpret_cast<void **>(&jitInterface));
  ASSERT_GE(result, 0);
  ASSERT_NE(jitInterface, nullptr);

  for (size_t i = 0; i < keys.size(); i += 16) {
    ULONG i1 = 0;
    ULONG i2 = 0;
    ULONG i3 = 0;
    ULONG i4 = 0;
    ULONG i5 = 0;
    ULONG i6 = 0;
    ULONG i7 = 0;
    ULONG i8 = 0;
    ULONG i9 = 0;
    ULONG i10 = 0;
    ULONG i11 = 0;
    ULONG i12 = 0;
    ULONG i13 = 0;
    ULONG i14 = 0;
    ULONG i15 = 0;
    ULONG i16 = 0;

    result = jitInterface->Vtbl->Index32x16(jitInterface,
                                            keys[i],
                                            keys[i + 1],
                                            keys[i + 2],
                                            keys[i + 3],
                                            keys[i + 4],
                                            keys[i + 5],
                                            keys[i + 6],
                                            keys[i + 7],
                                            keys[i + 8],
                                            keys[i + 9],
                                            keys[i + 10],
                                            keys[i + 11],
                                            keys[i + 12],
                                            keys[i + 13],
                                            keys[i + 14],
                                            keys[i + 15],
                                            &i1,
                                            &i2,
                                            &i3,
                                            &i4,
                                            &i5,
                                            &i6,
                                            &i7,
                                            &i8,
                                            &i9,
                                            &i10,
                                            &i11,
                                            &i12,
                                            &i13,
                                            &i14,
                                            &i15,
                                            &i16);
    ASSERT_GE(result, 0);

    EXPECT_EQ(expected[i], i1);
    EXPECT_EQ(expected[i + 1], i2);
    EXPECT_EQ(expected[i + 2], i3);
    EXPECT_EQ(expected[i + 3], i4);
    EXPECT_EQ(expected[i + 4], i5);
    EXPECT_EQ(expected[i + 5], i6);
    EXPECT_EQ(expected[i + 6], i7);
    EXPECT_EQ(expected[i + 7], i8);
    EXPECT_EQ(expected[i + 8], i9);
    EXPECT_EQ(expected[i + 9], i10);
    EXPECT_EQ(expected[i + 10], i11);
    EXPECT_EQ(expected[i + 11], i12);
    EXPECT_EQ(expected[i + 12], i13);
    EXPECT_EQ(expected[i + 13], i14);
    EXPECT_EQ(expected[i + 14], i15);
    EXPECT_EQ(expected[i + 15], i16);
  }

  jitInterface->Vtbl->Release(jitInterface);
  shim->Vtbl->Release(table);
}

TEST_F(PerfectHashOnlineTests, RawDogJitMulshrolate4RXStrictWidthsMatchIndex) {
  const auto keys = MakePseudoRandomKeys(64);
  const std::vector<uint32_t> queries(keys.begin(), keys.end());
  PH_ONLINE_JIT_CONTEXT context = {classFactory_, online_};
  struct Request {
    uint32_t width;
    PH_ONLINE_JIT_MAX_ISA max_isa;
    bool avx_only = false;
  };
  const auto compile = [&](PH_ONLINE_JIT_TABLE *table, const Request &request) {
#if defined(_M_X64) || defined(__x86_64__)
    if (request.avx_only) {
      return CompileRawDogX4WithoutAvx2(&context, table, request.max_isa);
    }
#endif
    return PhOnlineJitCompileTableEx(
        &context, table, PhOnlineJitBackendRawDogJit, request.width, request.max_isa,
        PH_ONLINE_JIT_COMPILE_FLAG_STRICT_VECTOR_WIDTH, nullptr, nullptr);
  };

#if defined(_M_X64) || defined(__x86_64__)
  // MultiplyShiftRX has the same ISA requirements for x4, x8 and x16.
  std::vector<Request> requests = {{1, PhOnlineJitMaxIsaAuto}};
  PH_ONLINE_JIT_TABLE *probe = nullptr;
  ASSERT_GE(PhOnlineJitCreateTable32(&context, PhOnlineJitHashMultiplyShiftRX,
                                     queries.data(), queries.size(), &probe), 0);
  std::unique_ptr<PH_ONLINE_JIT_TABLE, decltype(&PhOnlineJitReleaseTable)>
      probe_owner(probe, &PhOnlineJitReleaseTable);
  for (uint32_t width : {4u, 8u, 16u}) {
    const auto result = compile(probe, {width, PhOnlineJitMaxIsaAuto});
    if (result == PH_E_NOT_IMPLEMENTED) {
      continue;
    }
    ASSERT_GE(result, 0);
    if (width == 4) {
      for (const auto cap : {PhOnlineJitMaxIsaAuto, PhOnlineJitMaxIsaAvx,
                             PhOnlineJitMaxIsaAvx2, PhOnlineJitMaxIsaAvx512}) {
        requests.push_back({4, cap, false});
        requests.push_back({4, cap, true});
      }
    } else {
      requests.push_back({width, PhOnlineJitMaxIsaAuto});
    }
  }
#else
  const Request requests[] = {{1, PhOnlineJitMaxIsaAuto}, {4, PhOnlineJitMaxIsaAuto},
                              {8, PhOnlineJitMaxIsaAuto}, {16, PhOnlineJitMaxIsaAuto}};
#endif

  for (bool assigned16 : {false, true}) {
    SCOPED_TRACE(assigned16);
    PPERFECT_HASH_TABLE table = CreateTableFromKeys(
        keys, PerfectHashHashMulshrolate4RXFunctionId, assigned16, 3);
    ASSERT_NE(table, nullptr);
    const auto release = [](PPERFECT_HASH_TABLE t) {
      reinterpret_cast<PerfectHashTableShim *>(t)->Vtbl->Release(t);
    };
    std::unique_ptr<PERFECT_HASH_TABLE, decltype(release)> owner(table, release);
    PH_ONLINE_JIT_TABLE wrapper = {table};
    PH_ONLINE_JIT_TABLE_INFO info = {};
    ASSERT_GE(PhOnlineJitGetTableInfo(&wrapper, &info), 0);
    ASSERT_EQ(info.AssignedElementSizeInBytes, assigned16 ? 2u : 4u);

    std::vector<uint32_t> expected(keys.size());
    for (size_t i = 0; i < keys.size(); ++i) {
      ASSERT_GE(PhOnlineJitIndex32(&wrapper, queries[i], &expected[i]), 0);
    }

    for (const auto &request : requests) {
      const auto width = request.width;
      SCOPED_TRACE(width);
      SCOPED_TRACE(request.max_isa);
      SCOPED_TRACE(request.avx_only);
      ASSERT_GE(compile(&wrapper, request), 0);
#if defined(_M_X64) || defined(__x86_64__)
      if (width == 4) {
        auto *shim = reinterpret_cast<PerfectHashTableShim *>(table);
        PPERFECT_HASH_TABLE_JIT_INTERFACE jit = nullptr;
        HRESULT result = shim->Vtbl->QueryInterface(
            table,
#ifdef PH_WINDOWS
            IID_PERFECT_HASH_TABLE_JIT_INTERFACE,
#else
            &IID_PERFECT_HASH_TABLE_JIT_INTERFACE,
#endif
            reinterpret_cast<void **>(&jit));
        ASSERT_GE(result, 0);
        ASSERT_NE(jit, nullptr);
        PERFECT_HASH_TABLE_JIT_INFO jit_info = {};
        result = jit->Vtbl->GetInfo(jit, &jit_info);
        jit->Vtbl->Release(jit);
        ASSERT_GE(result, 0);
        EXPECT_EQ(jit_info.JitMaxIsa, PerfectHashJitMaxIsaAvx);
      }
#endif
      std::vector<uint32_t> actual(keys.size());
      for (size_t i = 0; i < keys.size(); i += width) {
        if (width == 1) {
          ASSERT_GE(PhOnlineJitIndex32(&wrapper, queries[i], &actual[i]), 0);
        } else {
          const auto index = width == 4 ? PhOnlineJitIndex32x4 :
                             width == 8 ? PhOnlineJitIndex32x8 : PhOnlineJitIndex32x16;
          ASSERT_GE(index(&wrapper, &queries[i], &actual[i]), 0);
        }
      }
      EXPECT_EQ(actual, expected);
    }
  }
}

TEST_F(PerfectHashOnlineTests,
       RawDogJitMulshrolate4RXIndex32x8MatchesIndex) {
  const std::vector<ULONG> keys = {
      1, 3, 5, 7, 11, 13, 17, 19,
      23, 29, 31, 37, 41, 43, 47, 53,
      59, 61, 67, 71, 73, 79, 83, 89,
      97, 101, 103, 107, 109, 113, 127, 131,
  };

  ScopedEnvVar rawdogVectorVersion("PH_RAWDOG_VECTOR_VERSION", "4");
  PPERFECT_HASH_TABLE table =
      CreateTableFromKeys(keys, PerfectHashHashMulshrolate4RXFunctionId);
  ASSERT_NE(table, nullptr);

  auto *shim = reinterpret_cast<PerfectHashTableShim *>(table);

  std::vector<ULONG> expected(keys.size());
  for (size_t i = 0; i < keys.size(); ++i) {
    HRESULT result = shim->Vtbl->Index(table, keys[i], &expected[i]);
    ASSERT_GE(result, 0);
  }

  PERFECT_HASH_TABLE_COMPILE_FLAGS compileFlags = {0};
  compileFlags.Jit = TRUE;
  compileFlags.JitBackendRawDog = TRUE;
  compileFlags.JitIndex32x8 = TRUE;

  HRESULT result = online_->Vtbl->CompileTable(online_, table, &compileFlags);
  if (result == PH_E_NOT_IMPLEMENTED) {
    shim->Vtbl->Release(table);
    GTEST_SKIP() << "RawDog AVX2 Index32x8 unavailable on this host.";
  }
  ASSERT_GE(result, 0);

  PPERFECT_HASH_TABLE_JIT_INTERFACE jitInterface = nullptr;
  result = shim->Vtbl->QueryInterface(
      table,
#ifdef PH_WINDOWS
      IID_PERFECT_HASH_TABLE_JIT_INTERFACE,
#else
      &IID_PERFECT_HASH_TABLE_JIT_INTERFACE,
#endif
      reinterpret_cast<void **>(&jitInterface));
  ASSERT_GE(result, 0);
  ASSERT_NE(jitInterface, nullptr);

  for (size_t i = 0; i < keys.size(); i += 8) {
    ULONG i1 = 0;
    ULONG i2 = 0;
    ULONG i3 = 0;
    ULONG i4 = 0;
    ULONG i5 = 0;
    ULONG i6 = 0;
    ULONG i7 = 0;
    ULONG i8 = 0;

    result = jitInterface->Vtbl->Index32x8(jitInterface,
                                           keys[i],
                                           keys[i + 1],
                                           keys[i + 2],
                                           keys[i + 3],
                                           keys[i + 4],
                                           keys[i + 5],
                                           keys[i + 6],
                                           keys[i + 7],
                                           &i1,
                                           &i2,
                                           &i3,
                                           &i4,
                                           &i5,
                                           &i6,
                                           &i7,
                                           &i8);
    ASSERT_GE(result, 0);

    EXPECT_EQ(expected[i], i1);
    EXPECT_EQ(expected[i + 1], i2);
    EXPECT_EQ(expected[i + 2], i3);
    EXPECT_EQ(expected[i + 3], i4);
    EXPECT_EQ(expected[i + 4], i5);
    EXPECT_EQ(expected[i + 5], i6);
    EXPECT_EQ(expected[i + 6], i7);
    EXPECT_EQ(expected[i + 7], i8);
  }

  jitInterface->Vtbl->Release(jitInterface);
  shim->Vtbl->Release(table);
}

TEST_F(PerfectHashOnlineTests,
       RawDogJitMulshrolate4RXIndex32x16MatchesIndex) {
  const std::vector<ULONG> keys = {
      1, 3, 5, 7, 11, 13, 17, 19,
      23, 29, 31, 37, 41, 43, 47, 53,
      59, 61, 67, 71, 73, 79, 83, 89,
      97, 101, 103, 107, 109, 113, 127, 131,
  };

  ScopedEnvVar rawdogVectorVersion("PH_RAWDOG_VECTOR_VERSION", "4");
  PPERFECT_HASH_TABLE table =
      CreateTableFromKeys(keys, PerfectHashHashMulshrolate4RXFunctionId);
  ASSERT_NE(table, nullptr);

  auto *shim = reinterpret_cast<PerfectHashTableShim *>(table);

  std::vector<ULONG> expected(keys.size());
  for (size_t i = 0; i < keys.size(); ++i) {
    HRESULT result = shim->Vtbl->Index(table, keys[i], &expected[i]);
    ASSERT_GE(result, 0);
  }

  PERFECT_HASH_TABLE_COMPILE_FLAGS compileFlags = {0};
  compileFlags.Jit = TRUE;
  compileFlags.JitBackendRawDog = TRUE;
  compileFlags.JitIndex32x16 = TRUE;

  HRESULT result = online_->Vtbl->CompileTable(online_, table, &compileFlags);
  if (result == PH_E_NOT_IMPLEMENTED) {
    shim->Vtbl->Release(table);
    GTEST_SKIP() << "RawDog AVX-512 Index32x16 unavailable on this host.";
  }
  ASSERT_GE(result, 0);

  PPERFECT_HASH_TABLE_JIT_INTERFACE jitInterface = nullptr;
  result = shim->Vtbl->QueryInterface(
      table,
#ifdef PH_WINDOWS
      IID_PERFECT_HASH_TABLE_JIT_INTERFACE,
#else
      &IID_PERFECT_HASH_TABLE_JIT_INTERFACE,
#endif
      reinterpret_cast<void **>(&jitInterface));
  ASSERT_GE(result, 0);
  ASSERT_NE(jitInterface, nullptr);

  for (size_t i = 0; i < keys.size(); i += 16) {
    ULONG i1 = 0;
    ULONG i2 = 0;
    ULONG i3 = 0;
    ULONG i4 = 0;
    ULONG i5 = 0;
    ULONG i6 = 0;
    ULONG i7 = 0;
    ULONG i8 = 0;
    ULONG i9 = 0;
    ULONG i10 = 0;
    ULONG i11 = 0;
    ULONG i12 = 0;
    ULONG i13 = 0;
    ULONG i14 = 0;
    ULONG i15 = 0;
    ULONG i16 = 0;

    result = jitInterface->Vtbl->Index32x16(jitInterface,
                                            keys[i],
                                            keys[i + 1],
                                            keys[i + 2],
                                            keys[i + 3],
                                            keys[i + 4],
                                            keys[i + 5],
                                            keys[i + 6],
                                            keys[i + 7],
                                            keys[i + 8],
                                            keys[i + 9],
                                            keys[i + 10],
                                            keys[i + 11],
                                            keys[i + 12],
                                            keys[i + 13],
                                            keys[i + 14],
                                            keys[i + 15],
                                            &i1,
                                            &i2,
                                            &i3,
                                            &i4,
                                            &i5,
                                            &i6,
                                            &i7,
                                            &i8,
                                            &i9,
                                            &i10,
                                            &i11,
                                            &i12,
                                            &i13,
                                            &i14,
                                            &i15,
                                            &i16);
    ASSERT_GE(result, 0);

    EXPECT_EQ(expected[i], i1);
    EXPECT_EQ(expected[i + 1], i2);
    EXPECT_EQ(expected[i + 2], i3);
    EXPECT_EQ(expected[i + 3], i4);
    EXPECT_EQ(expected[i + 4], i5);
    EXPECT_EQ(expected[i + 5], i6);
    EXPECT_EQ(expected[i + 6], i7);
    EXPECT_EQ(expected[i + 7], i8);
    EXPECT_EQ(expected[i + 8], i9);
    EXPECT_EQ(expected[i + 9], i10);
    EXPECT_EQ(expected[i + 10], i11);
    EXPECT_EQ(expected[i + 11], i12);
    EXPECT_EQ(expected[i + 12], i13);
    EXPECT_EQ(expected[i + 13], i14);
    EXPECT_EQ(expected[i + 14], i15);
    EXPECT_EQ(expected[i + 15], i16);
  }

  jitInterface->Vtbl->Release(jitInterface);
  shim->Vtbl->Release(table);
}
#endif

#if defined(PH_HAS_LLVM)
TEST_F(PerfectHashOnlineTests, JitIndexMatchesSlowIndex) {
  const std::vector<ULONG> keys = {
      1, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53,
      59, 61, 67, 71, 73, 79, 83, 89, 97, 101, 103, 107, 109, 113,
  };

  const PERFECT_HASH_HASH_FUNCTION_ID hashFunctions[] = {
      PerfectHashHashMultiplyShiftRFunctionId,
      PerfectHashHashMultiplyShiftRXFunctionId,
      PerfectHashHashMulshrolate1RXFunctionId,
      PerfectHashHashMulshrolate2RXFunctionId,
      PerfectHashHashMulshrolate3RXFunctionId,
      PerfectHashHashMulshrolate4RXFunctionId,
  };

  for (auto hashFunctionId : hashFunctions) {
    PPERFECT_HASH_TABLE table = CreateTableFromKeys(keys, hashFunctionId);
    ASSERT_NE(table, nullptr);

    auto *shim = reinterpret_cast<PerfectHashTableShim *>(table);
    ASSERT_NE(shim->Vtbl->SlowIndex, nullptr);

    PERFECT_HASH_TABLE_COMPILE_FLAGS compileFlags = {0};
    HRESULT result = online_->Vtbl->CompileTable(online_, table, &compileFlags);
    if (result < 0) {
      shim->Vtbl->Release(table);
      GTEST_SKIP() << "LLVM JIT unavailable on this host.";
    }
    ASSERT_GE(result, 0);

    for (ULONG key : keys) {
      ULONG slowIndex = 0;
      ULONG jitIndex = 0;

      result = shim->Vtbl->SlowIndex(table, key, &slowIndex);
      ASSERT_GE(result, 0);

      result = shim->Vtbl->Index(table, key, &jitIndex);
      ASSERT_GE(result, 0);

      EXPECT_EQ(slowIndex, jitIndex);
    }

    shim->Vtbl->Release(table);
  }
}

TEST_F(PerfectHashOnlineTests, JitAssigned16MatchesOriginalIndex) {
  const std::vector<ULONG> keys = {
      1, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53,
  };

  PPERFECT_HASH_TABLE table = CreateTableFromKeys(
      keys, PerfectHashHashMultiplyShiftRFunctionId, true);
  ASSERT_NE(table, nullptr);

  auto *shim = reinterpret_cast<PerfectHashTableShim *>(table);
  ASSERT_EQ(shim->Vtbl->SlowIndex, nullptr);

  std::vector<ULONG> expected;
  expected.reserve(keys.size());

  for (ULONG key : keys) {
    ULONG index = 0;
    HRESULT result = shim->Vtbl->Index(table, key, &index);
    ASSERT_GE(result, 0);
    expected.push_back(index);
  }

  PERFECT_HASH_TABLE_COMPILE_FLAGS compileFlags = {0};
  HRESULT result = online_->Vtbl->CompileTable(online_, table, &compileFlags);
  if (result < 0) {
    shim->Vtbl->Release(table);
    GTEST_SKIP() << "LLVM JIT unavailable on this host.";
  }
  ASSERT_GE(result, 0);

  for (size_t i = 0; i < keys.size(); ++i) {
    ULONG index = 0;
    result = shim->Vtbl->Index(table, keys[i], &index);
    ASSERT_GE(result, 0);
    EXPECT_EQ(expected[i], index);
  }

  shim->Vtbl->Release(table);
}

TEST_F(PerfectHashOnlineTests,
       JitInterfaceIndex32x2Index32x4Index32x8MatchSlowIndex) {
  const std::vector<ULONG> keys = {
      1, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53,
      59, 61, 67, 71, 73, 79, 83, 89, 97, 101, 103, 107, 109, 113,
      127, 131,
  };

  PPERFECT_HASH_TABLE table =
      CreateTableFromKeys(keys, PerfectHashHashMultiplyShiftRFunctionId);
  ASSERT_NE(table, nullptr);

  auto *shim = reinterpret_cast<PerfectHashTableShim *>(table);
  ASSERT_NE(shim->Vtbl->SlowIndex, nullptr);

  PERFECT_HASH_TABLE_COMPILE_FLAGS compileFlags = {0};
  compileFlags.JitIndex32x2 = TRUE;
  compileFlags.JitIndex32x4 = TRUE;
  compileFlags.JitIndex32x8 = TRUE;

  HRESULT result = online_->Vtbl->CompileTable(online_, table, &compileFlags);
  if (result < 0) {
    shim->Vtbl->Release(table);
    GTEST_SKIP() << "LLVM JIT unavailable on this host.";
  }
  ASSERT_GE(result, 0);

  PPERFECT_HASH_TABLE_JIT_INTERFACE jit = nullptr;
  result = shim->Vtbl->QueryInterface(
      table,
#ifdef PH_WINDOWS
      IID_PERFECT_HASH_TABLE_JIT_INTERFACE,
#else
      &IID_PERFECT_HASH_TABLE_JIT_INTERFACE,
#endif
      reinterpret_cast<void **>(&jit));
  ASSERT_GE(result, 0);
  ASSERT_NE(jit, nullptr);

  for (size_t i = 0; i < keys.size(); i += 2) {
    ULONG expected1 = 0;
    ULONG expected2 = 0;
    ULONG index1 = 0;
    ULONG index2 = 0;

    result = shim->Vtbl->SlowIndex(table, keys[i], &expected1);
    ASSERT_GE(result, 0);
    result = shim->Vtbl->SlowIndex(table, keys[i + 1], &expected2);
    ASSERT_GE(result, 0);

    result = jit->Vtbl->Index32x2(jit, keys[i], keys[i + 1], &index1, &index2);
    ASSERT_GE(result, 0);

    EXPECT_EQ(expected1, index1);
    EXPECT_EQ(expected2, index2);
  }

  for (size_t i = 0; i < keys.size(); i += 4) {
    ULONG expected1 = 0;
    ULONG expected2 = 0;
    ULONG expected3 = 0;
    ULONG expected4 = 0;
    ULONG index1 = 0;
    ULONG index2 = 0;
    ULONG index3 = 0;
    ULONG index4 = 0;

    result = shim->Vtbl->SlowIndex(table, keys[i], &expected1);
    ASSERT_GE(result, 0);
    result = shim->Vtbl->SlowIndex(table, keys[i + 1], &expected2);
    ASSERT_GE(result, 0);
    result = shim->Vtbl->SlowIndex(table, keys[i + 2], &expected3);
    ASSERT_GE(result, 0);
    result = shim->Vtbl->SlowIndex(table, keys[i + 3], &expected4);
    ASSERT_GE(result, 0);

    result = jit->Vtbl->Index32x4(jit,
                               keys[i],
                               keys[i + 1],
                               keys[i + 2],
                               keys[i + 3],
                               &index1,
                               &index2,
                               &index3,
                               &index4);
    ASSERT_GE(result, 0);

    EXPECT_EQ(expected1, index1);
    EXPECT_EQ(expected2, index2);
    EXPECT_EQ(expected3, index3);
    EXPECT_EQ(expected4, index4);
  }

  for (size_t i = 0; i < keys.size(); i += 8) {
    ULONG expected1 = 0;
    ULONG expected2 = 0;
    ULONG expected3 = 0;
    ULONG expected4 = 0;
    ULONG expected5 = 0;
    ULONG expected6 = 0;
    ULONG expected7 = 0;
    ULONG expected8 = 0;
    ULONG index1 = 0;
    ULONG index2 = 0;
    ULONG index3 = 0;
    ULONG index4 = 0;
    ULONG index5 = 0;
    ULONG index6 = 0;
    ULONG index7 = 0;
    ULONG index8 = 0;

    result = shim->Vtbl->SlowIndex(table, keys[i], &expected1);
    ASSERT_GE(result, 0);
    result = shim->Vtbl->SlowIndex(table, keys[i + 1], &expected2);
    ASSERT_GE(result, 0);
    result = shim->Vtbl->SlowIndex(table, keys[i + 2], &expected3);
    ASSERT_GE(result, 0);
    result = shim->Vtbl->SlowIndex(table, keys[i + 3], &expected4);
    ASSERT_GE(result, 0);
    result = shim->Vtbl->SlowIndex(table, keys[i + 4], &expected5);
    ASSERT_GE(result, 0);
    result = shim->Vtbl->SlowIndex(table, keys[i + 5], &expected6);
    ASSERT_GE(result, 0);
    result = shim->Vtbl->SlowIndex(table, keys[i + 6], &expected7);
    ASSERT_GE(result, 0);
    result = shim->Vtbl->SlowIndex(table, keys[i + 7], &expected8);
    ASSERT_GE(result, 0);

    result = jit->Vtbl->Index32x8(jit,
                               keys[i],
                               keys[i + 1],
                               keys[i + 2],
                               keys[i + 3],
                               keys[i + 4],
                               keys[i + 5],
                               keys[i + 6],
                               keys[i + 7],
                               &index1,
                               &index2,
                               &index3,
                               &index4,
                               &index5,
                               &index6,
                               &index7,
                               &index8);
    ASSERT_GE(result, 0);

    EXPECT_EQ(expected1, index1);
    EXPECT_EQ(expected2, index2);
    EXPECT_EQ(expected3, index3);
    EXPECT_EQ(expected4, index4);
    EXPECT_EQ(expected5, index5);
    EXPECT_EQ(expected6, index6);
    EXPECT_EQ(expected7, index7);
    EXPECT_EQ(expected8, index8);
  }

  jit->Vtbl->Release(jit);
  shim->Vtbl->Release(table);
}

TEST_F(PerfectHashOnlineTests,
       JitVectorInterfaceIndex32x2Index32x4Index32x8MatchSlowIndex) {
  const std::vector<ULONG> keys = {
      1, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53,
      59, 61, 67, 71, 73, 79, 83, 89, 97, 101, 103, 107, 109, 113,
      127, 131,
  };

  PPERFECT_HASH_TABLE table =
      CreateTableFromKeys(keys, PerfectHashHashMultiplyShiftRFunctionId);
  ASSERT_NE(table, nullptr);

  auto *shim = reinterpret_cast<PerfectHashTableShim *>(table);
  ASSERT_NE(shim->Vtbl->SlowIndex, nullptr);

  PERFECT_HASH_TABLE_COMPILE_FLAGS compileFlags = {0};
  compileFlags.JitVectorIndex32x2 = TRUE;
  compileFlags.JitVectorIndex32x4 = TRUE;
  compileFlags.JitVectorIndex32x8 = TRUE;

  HRESULT result = online_->Vtbl->CompileTable(online_, table, &compileFlags);
  if (result < 0) {
    shim->Vtbl->Release(table);
    GTEST_SKIP() << "LLVM JIT unavailable on this host.";
  }
  ASSERT_GE(result, 0);

  PPERFECT_HASH_TABLE_JIT_INTERFACE jit = nullptr;
  result = shim->Vtbl->QueryInterface(
      table,
#ifdef PH_WINDOWS
      IID_PERFECT_HASH_TABLE_JIT_INTERFACE,
#else
      &IID_PERFECT_HASH_TABLE_JIT_INTERFACE,
#endif
      reinterpret_cast<void **>(&jit));
  ASSERT_GE(result, 0);
  ASSERT_NE(jit, nullptr);

  PERFECT_HASH_TABLE_JIT_INFO info = {0};
  result = jit->Vtbl->GetInfo(jit, &info);
  ASSERT_GE(result, 0);
  EXPECT_TRUE(info.Flags.Index32x2Vector);
  EXPECT_TRUE(info.Flags.Index32x4Vector);
  EXPECT_TRUE(info.Flags.Index32x8Vector);

  for (size_t i = 0; i < keys.size(); i += 2) {
    ULONG expected1 = 0;
    ULONG expected2 = 0;
    ULONG index1 = 0;
    ULONG index2 = 0;

    result = shim->Vtbl->SlowIndex(table, keys[i], &expected1);
    ASSERT_GE(result, 0);
    result = shim->Vtbl->SlowIndex(table, keys[i + 1], &expected2);
    ASSERT_GE(result, 0);

    result = jit->Vtbl->Index32x2(jit, keys[i], keys[i + 1], &index1, &index2);
    ASSERT_GE(result, 0);

    EXPECT_EQ(expected1, index1);
    EXPECT_EQ(expected2, index2);
  }

  for (size_t i = 0; i < keys.size(); i += 4) {
    ULONG expected1 = 0;
    ULONG expected2 = 0;
    ULONG expected3 = 0;
    ULONG expected4 = 0;
    ULONG index1 = 0;
    ULONG index2 = 0;
    ULONG index3 = 0;
    ULONG index4 = 0;

    result = shim->Vtbl->SlowIndex(table, keys[i], &expected1);
    ASSERT_GE(result, 0);
    result = shim->Vtbl->SlowIndex(table, keys[i + 1], &expected2);
    ASSERT_GE(result, 0);
    result = shim->Vtbl->SlowIndex(table, keys[i + 2], &expected3);
    ASSERT_GE(result, 0);
    result = shim->Vtbl->SlowIndex(table, keys[i + 3], &expected4);
    ASSERT_GE(result, 0);

    result = jit->Vtbl->Index32x4(jit,
                               keys[i],
                               keys[i + 1],
                               keys[i + 2],
                               keys[i + 3],
                               &index1,
                               &index2,
                               &index3,
                               &index4);
    ASSERT_GE(result, 0);

    EXPECT_EQ(expected1, index1);
    EXPECT_EQ(expected2, index2);
    EXPECT_EQ(expected3, index3);
    EXPECT_EQ(expected4, index4);
  }

  for (size_t i = 0; i < keys.size(); i += 8) {
    ULONG expected1 = 0;
    ULONG expected2 = 0;
    ULONG expected3 = 0;
    ULONG expected4 = 0;
    ULONG expected5 = 0;
    ULONG expected6 = 0;
    ULONG expected7 = 0;
    ULONG expected8 = 0;
    ULONG index1 = 0;
    ULONG index2 = 0;
    ULONG index3 = 0;
    ULONG index4 = 0;
    ULONG index5 = 0;
    ULONG index6 = 0;
    ULONG index7 = 0;
    ULONG index8 = 0;

    result = shim->Vtbl->SlowIndex(table, keys[i], &expected1);
    ASSERT_GE(result, 0);
    result = shim->Vtbl->SlowIndex(table, keys[i + 1], &expected2);
    ASSERT_GE(result, 0);
    result = shim->Vtbl->SlowIndex(table, keys[i + 2], &expected3);
    ASSERT_GE(result, 0);
    result = shim->Vtbl->SlowIndex(table, keys[i + 3], &expected4);
    ASSERT_GE(result, 0);
    result = shim->Vtbl->SlowIndex(table, keys[i + 4], &expected5);
    ASSERT_GE(result, 0);
    result = shim->Vtbl->SlowIndex(table, keys[i + 5], &expected6);
    ASSERT_GE(result, 0);
    result = shim->Vtbl->SlowIndex(table, keys[i + 6], &expected7);
    ASSERT_GE(result, 0);
    result = shim->Vtbl->SlowIndex(table, keys[i + 7], &expected8);
    ASSERT_GE(result, 0);

    result = jit->Vtbl->Index32x8(jit,
                               keys[i],
                               keys[i + 1],
                               keys[i + 2],
                               keys[i + 3],
                               keys[i + 4],
                               keys[i + 5],
                               keys[i + 6],
                               keys[i + 7],
                               &index1,
                               &index2,
                               &index3,
                               &index4,
                               &index5,
                               &index6,
                               &index7,
                               &index8);
    ASSERT_GE(result, 0);

    EXPECT_EQ(expected1, index1);
    EXPECT_EQ(expected2, index2);
    EXPECT_EQ(expected3, index3);
    EXPECT_EQ(expected4, index4);
    EXPECT_EQ(expected5, index5);
    EXPECT_EQ(expected6, index6);
    EXPECT_EQ(expected7, index7);
    EXPECT_EQ(expected8, index8);
  }

  jit->Vtbl->Release(jit);
  shim->Vtbl->Release(table);
}

TEST_F(PerfectHashOnlineTests, JitInterfaceIndex32x16MatchesSlowIndex) {
  const std::vector<ULONG> keys = {
      1, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53,
      59, 61, 67, 71, 73, 79, 83, 89, 97, 101, 103, 107, 109, 113,
      127, 131,
  };

  PPERFECT_HASH_TABLE table =
      CreateTableFromKeys(keys, PerfectHashHashMultiplyShiftRFunctionId);
  ASSERT_NE(table, nullptr);

  auto *shim = reinterpret_cast<PerfectHashTableShim *>(table);
  ASSERT_NE(shim->Vtbl->SlowIndex, nullptr);

  PERFECT_HASH_TABLE_COMPILE_FLAGS compileFlags = {0};
  compileFlags.JitBackendLlvm = TRUE;
  compileFlags.JitIndex32x16 = TRUE;
  const bool arm64 = PerfectHashGetCurrentCpuArch() == PerfectHashArm64CpuArchId;
  if (arm64) {
    compileFlags.JitMaxIsa = PerfectHashJitMaxIsaNeon;
  }

  HRESULT result = online_->Vtbl->CompileTable(online_, table, &compileFlags);
  if (result < 0) {
    shim->Vtbl->Release(table);
#ifdef PH_HAS_LLVM
    if (arm64) {
      FAIL() << "LLVM could not compile the NEON x16 index: " << result;
    }
#endif
    GTEST_SKIP() << "LLVM JIT unavailable on this host.";
  }
  ASSERT_GE(result, 0);

  PPERFECT_HASH_TABLE_JIT_INTERFACE jit = nullptr;
  result = shim->Vtbl->QueryInterface(
      table,
#ifdef PH_WINDOWS
      IID_PERFECT_HASH_TABLE_JIT_INTERFACE,
#else
      &IID_PERFECT_HASH_TABLE_JIT_INTERFACE,
#endif
      reinterpret_cast<void **>(&jit));
  ASSERT_GE(result, 0);
  ASSERT_NE(jit, nullptr);

  PERFECT_HASH_TABLE_JIT_INFO info = {0};
  result = jit->Vtbl->GetInfo(jit, &info);
  ASSERT_GE(result, 0);

  if (!info.Flags.Index32x16Compiled) {
    jit->Vtbl->Release(jit);
    shim->Vtbl->Release(table);
    if (arm64) {
      FAIL() << "LLVM did not compile the requested ARM64 x16 index.";
    }
    GTEST_SKIP() << "JIT Index32x16 not supported on this host.";
  }

  EXPECT_TRUE(info.Flags.Index32x16Vector);

  for (size_t i = 0; i < keys.size(); i += 16) {
    ULONG expected[16] = {};
    ULONG index[16] = {};

    for (size_t lane = 0; lane < 16; ++lane) {
      result = shim->Vtbl->SlowIndex(table, keys[i + lane], &expected[lane]);
      ASSERT_GE(result, 0);
    }

    result = jit->Vtbl->Index32x16(jit,
                                   keys[i],
                                   keys[i + 1],
                                   keys[i + 2],
                                   keys[i + 3],
                                   keys[i + 4],
                                   keys[i + 5],
                                   keys[i + 6],
                                   keys[i + 7],
                                   keys[i + 8],
                                   keys[i + 9],
                                   keys[i + 10],
                                   keys[i + 11],
                                   keys[i + 12],
                                   keys[i + 13],
                                   keys[i + 14],
                                   keys[i + 15],
                                   &index[0],
                                   &index[1],
                                   &index[2],
                                   &index[3],
                                   &index[4],
                                   &index[5],
                                   &index[6],
                                   &index[7],
                                   &index[8],
                                   &index[9],
                                   &index[10],
                                   &index[11],
                                   &index[12],
                                   &index[13],
                                   &index[14],
                                   &index[15]);
    ASSERT_GE(result, 0);

    for (size_t lane = 0; lane < 16; ++lane) {
      EXPECT_EQ(expected[lane], index[lane]);
    }
  }

  jit->Vtbl->Release(jit);
  shim->Vtbl->Release(table);
}

TEST_F(PerfectHashOnlineTests, JitInterfaceIndex64x2x4x8MatchesDownsizedIndex) {
  const std::vector<ULONGLONG> keys = {
      1ull, 3ull, 5ull, 7ull, 11ull, 13ull, 17ull, 19ull,
      23ull, 29ull, 31ull, 37ull, 41ull, 43ull, 47ull, 53ull,
      59ull, 61ull, 67ull, 71ull, 73ull, 79ull, 83ull, 89ull,
      97ull, 101ull, 103ull, 107ull, 109ull, 113ull, 127ull, 131ull,
  };

  PPERFECT_HASH_TABLE table =
      CreateTableFromKeys64(keys, PerfectHashHashMultiplyShiftRFunctionId);
  ASSERT_NE(table, nullptr);

  auto *shim = reinterpret_cast<PerfectHashTableShim *>(table);
  ASSERT_NE(shim->Vtbl->SlowIndex, nullptr);

  const ULONGLONG mask = BuildDownsizeMask(keys);
  std::vector<ULONG> expected;
  expected.reserve(keys.size());

  for (auto key : keys) {
    ULONG downsized = DownsizeKey(key, mask);
    ULONG index = 0;
    HRESULT result = shim->Vtbl->SlowIndex(table, downsized, &index);
    ASSERT_GE(result, 0);
    expected.push_back(index);
  }

  PERFECT_HASH_TABLE_COMPILE_FLAGS compileFlags = {0};
  compileFlags.JitIndex64 = TRUE;
  compileFlags.JitIndex32x2 = TRUE;
  compileFlags.JitIndex32x4 = TRUE;
  compileFlags.JitIndex32x8 = TRUE;

  HRESULT result = online_->Vtbl->CompileTable(online_, table, &compileFlags);
  if (result < 0) {
    shim->Vtbl->Release(table);
    GTEST_SKIP() << "LLVM JIT unavailable on this host.";
  }
  ASSERT_GE(result, 0);

  PPERFECT_HASH_TABLE_JIT_INTERFACE jit = nullptr;
  result = shim->Vtbl->QueryInterface(
      table,
#ifdef PH_WINDOWS
      IID_PERFECT_HASH_TABLE_JIT_INTERFACE,
#else
      &IID_PERFECT_HASH_TABLE_JIT_INTERFACE,
#endif
      reinterpret_cast<void **>(&jit));
  ASSERT_GE(result, 0);
  ASSERT_NE(jit, nullptr);

  for (size_t i = 0; i < keys.size(); ++i) {
    ULONG index = 0;
    result = jit->Vtbl->Index64(jit, keys[i], &index);
    ASSERT_GE(result, 0);
    EXPECT_EQ(expected[i], index);
  }

  for (size_t i = 0; i < keys.size(); i += 2) {
    ULONG index1 = 0;
    ULONG index2 = 0;

    result = jit->Vtbl->Index64x2(jit,
                                  keys[i],
                                  keys[i + 1],
                                  &index1,
                                  &index2);
    ASSERT_GE(result, 0);

    EXPECT_EQ(expected[i], index1);
    EXPECT_EQ(expected[i + 1], index2);
  }

  for (size_t i = 0; i < keys.size(); i += 4) {
    ULONG index1 = 0;
    ULONG index2 = 0;
    ULONG index3 = 0;
    ULONG index4 = 0;

    result = jit->Vtbl->Index64x4(jit,
                                  keys[i],
                                  keys[i + 1],
                                  keys[i + 2],
                                  keys[i + 3],
                                  &index1,
                                  &index2,
                                  &index3,
                                  &index4);
    ASSERT_GE(result, 0);

    EXPECT_EQ(expected[i], index1);
    EXPECT_EQ(expected[i + 1], index2);
    EXPECT_EQ(expected[i + 2], index3);
    EXPECT_EQ(expected[i + 3], index4);
  }

  for (size_t i = 0; i < keys.size(); i += 8) {
    ULONG index1 = 0;
    ULONG index2 = 0;
    ULONG index3 = 0;
    ULONG index4 = 0;
    ULONG index5 = 0;
    ULONG index6 = 0;
    ULONG index7 = 0;
    ULONG index8 = 0;

    result = jit->Vtbl->Index64x8(jit,
                                  keys[i],
                                  keys[i + 1],
                                  keys[i + 2],
                                  keys[i + 3],
                                  keys[i + 4],
                                  keys[i + 5],
                                  keys[i + 6],
                                  keys[i + 7],
                                  &index1,
                                  &index2,
                                  &index3,
                                  &index4,
                                  &index5,
                                  &index6,
                                  &index7,
                                  &index8);
    ASSERT_GE(result, 0);

    EXPECT_EQ(expected[i], index1);
    EXPECT_EQ(expected[i + 1], index2);
    EXPECT_EQ(expected[i + 2], index3);
    EXPECT_EQ(expected[i + 3], index4);
    EXPECT_EQ(expected[i + 4], index5);
    EXPECT_EQ(expected[i + 5], index6);
    EXPECT_EQ(expected[i + 6], index7);
    EXPECT_EQ(expected[i + 7], index8);
  }

  jit->Vtbl->Release(jit);
  shim->Vtbl->Release(table);
}

TEST_F(PerfectHashOnlineTests,
       JitVectorInterfaceIndex64x2x4x8MatchesDownsizedIndex) {
  const std::vector<ULONGLONG> keys = {
      1ull, 3ull, 5ull, 7ull, 11ull, 13ull, 17ull, 19ull,
      23ull, 29ull, 31ull, 37ull, 41ull, 43ull, 47ull, 53ull,
      59ull, 61ull, 67ull, 71ull, 73ull, 79ull, 83ull, 89ull,
      97ull, 101ull, 103ull, 107ull, 109ull, 113ull, 127ull, 131ull,
  };

  PPERFECT_HASH_TABLE table =
      CreateTableFromKeys64(keys, PerfectHashHashMultiplyShiftRFunctionId);
  ASSERT_NE(table, nullptr);

  auto *shim = reinterpret_cast<PerfectHashTableShim *>(table);
  ASSERT_NE(shim->Vtbl->SlowIndex, nullptr);

  const ULONGLONG mask = BuildDownsizeMask(keys);
  std::vector<ULONG> expected;
  expected.reserve(keys.size());

  for (auto key : keys) {
    ULONG downsized = DownsizeKey(key, mask);
    ULONG index = 0;
    HRESULT result = shim->Vtbl->SlowIndex(table, downsized, &index);
    ASSERT_GE(result, 0);
    expected.push_back(index);
  }

  PERFECT_HASH_TABLE_COMPILE_FLAGS compileFlags = {0};
  compileFlags.JitIndex64 = TRUE;
  compileFlags.JitVectorIndex32x2 = TRUE;
  compileFlags.JitVectorIndex32x4 = TRUE;
  compileFlags.JitVectorIndex32x8 = TRUE;

  HRESULT result = online_->Vtbl->CompileTable(online_, table, &compileFlags);
  if (result < 0) {
    shim->Vtbl->Release(table);
    GTEST_SKIP() << "LLVM JIT unavailable on this host.";
  }
  ASSERT_GE(result, 0);

  PPERFECT_HASH_TABLE_JIT_INTERFACE jit = nullptr;
  result = shim->Vtbl->QueryInterface(
      table,
#ifdef PH_WINDOWS
      IID_PERFECT_HASH_TABLE_JIT_INTERFACE,
#else
      &IID_PERFECT_HASH_TABLE_JIT_INTERFACE,
#endif
      reinterpret_cast<void **>(&jit));
  ASSERT_GE(result, 0);
  ASSERT_NE(jit, nullptr);

  PERFECT_HASH_TABLE_JIT_INFO info = {0};
  result = jit->Vtbl->GetInfo(jit, &info);
  ASSERT_GE(result, 0);
  EXPECT_TRUE(info.Flags.Index64x2Vector);
  EXPECT_TRUE(info.Flags.Index64x4Vector);
  EXPECT_TRUE(info.Flags.Index64x8Vector);

  for (size_t i = 0; i < keys.size(); ++i) {
    ULONG index = 0;
    result = jit->Vtbl->Index64(jit, keys[i], &index);
    ASSERT_GE(result, 0);
    EXPECT_EQ(expected[i], index);
  }

  for (size_t i = 0; i < keys.size(); i += 2) {
    ULONG index1 = 0;
    ULONG index2 = 0;

    result = jit->Vtbl->Index64x2(jit,
                                  keys[i],
                                  keys[i + 1],
                                  &index1,
                                  &index2);
    ASSERT_GE(result, 0);

    EXPECT_EQ(expected[i], index1);
    EXPECT_EQ(expected[i + 1], index2);
  }

  for (size_t i = 0; i < keys.size(); i += 4) {
    ULONG index1 = 0;
    ULONG index2 = 0;
    ULONG index3 = 0;
    ULONG index4 = 0;

    result = jit->Vtbl->Index64x4(jit,
                                  keys[i],
                                  keys[i + 1],
                                  keys[i + 2],
                                  keys[i + 3],
                                  &index1,
                                  &index2,
                                  &index3,
                                  &index4);
    ASSERT_GE(result, 0);

    EXPECT_EQ(expected[i], index1);
    EXPECT_EQ(expected[i + 1], index2);
    EXPECT_EQ(expected[i + 2], index3);
    EXPECT_EQ(expected[i + 3], index4);
  }

  for (size_t i = 0; i < keys.size(); i += 8) {
    ULONG index1 = 0;
    ULONG index2 = 0;
    ULONG index3 = 0;
    ULONG index4 = 0;
    ULONG index5 = 0;
    ULONG index6 = 0;
    ULONG index7 = 0;
    ULONG index8 = 0;

    result = jit->Vtbl->Index64x8(jit,
                                  keys[i],
                                  keys[i + 1],
                                  keys[i + 2],
                                  keys[i + 3],
                                  keys[i + 4],
                                  keys[i + 5],
                                  keys[i + 6],
                                  keys[i + 7],
                                  &index1,
                                  &index2,
                                  &index3,
                                  &index4,
                                  &index5,
                                  &index6,
                                  &index7,
                                  &index8);
    ASSERT_GE(result, 0);

    EXPECT_EQ(expected[i], index1);
    EXPECT_EQ(expected[i + 1], index2);
    EXPECT_EQ(expected[i + 2], index3);
    EXPECT_EQ(expected[i + 3], index4);
    EXPECT_EQ(expected[i + 4], index5);
    EXPECT_EQ(expected[i + 5], index6);
    EXPECT_EQ(expected[i + 6], index7);
    EXPECT_EQ(expected[i + 7], index8);
  }

  jit->Vtbl->Release(jit);
  shim->Vtbl->Release(table);
}
#else
TEST_F(PerfectHashOnlineTests, JitIndexMatchesSlowIndex) {
  GTEST_SKIP() << "LLVM support is disabled.";
}

TEST_F(PerfectHashOnlineTests, JitAssigned16MatchesOriginalIndex) {
  GTEST_SKIP() << "LLVM support is disabled.";
}

TEST_F(PerfectHashOnlineTests,
       JitInterfaceIndex32x2Index32x4Index32x8MatchSlowIndex) {
  GTEST_SKIP() << "LLVM support is disabled.";
}

TEST_F(PerfectHashOnlineTests,
       JitVectorInterfaceIndex32x2Index32x4Index32x8MatchSlowIndex) {
  GTEST_SKIP() << "LLVM support is disabled.";
}

TEST_F(PerfectHashOnlineTests, JitInterfaceIndex32x16MatchesSlowIndex) {
  GTEST_SKIP() << "LLVM support is disabled.";
}

TEST_F(PerfectHashOnlineTests, JitInterfaceIndex64x2x4x8MatchesDownsizedIndex) {
  GTEST_SKIP() << "LLVM support is disabled.";
}

TEST_F(PerfectHashOnlineTests,
       JitVectorInterfaceIndex64x2x4x8MatchesDownsizedIndex) {
  GTEST_SKIP() << "LLVM support is disabled.";
}
#endif

} // namespace
