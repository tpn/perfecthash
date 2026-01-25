#include <gtest/gtest.h>

#include <PerfectHash.h>

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <fstream>
#include <string>
#include <unordered_set>
#include <vector>

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
      const std::vector<ULONG> *seeds = nullptr) {
    PERFECT_HASH_KEYS_LOAD_FLAGS keysFlags = {0};
    PERFECT_HASH_TABLE_CREATE_FLAGS tableFlags = {0};
    std::vector<PERFECT_HASH_TABLE_CREATE_PARAMETER> table_params;
    PERFECT_HASH_TABLE_CREATE_PARAMETERS tableCreateParams = {};
    PPERFECT_HASH_TABLE_CREATE_PARAMETERS tableCreateParamsPtr = nullptr;
    PPERFECT_HASH_TABLE table = nullptr;

    tableFlags.NoFileIo = TRUE;
    tableFlags.Quiet = TRUE;
    tableFlags.DoNotTryUseHash16Impl = allowAssigned16 ? FALSE : TRUE;

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

#if defined(PH_HAS_RAWDOG_JIT)
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
#endif

#if defined(PH_HAS_LLVM)
TEST_F(PerfectHashOnlineTests, JitIndexMatchesSlowIndex) {
  const std::vector<ULONG> keys = {
      1, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53,
      59, 61, 67, 71, 73, 79, 83, 89, 97, 101, 103, 107, 109, 113,
  };

  const PERFECT_HASH_HASH_FUNCTION_ID hashFunctions[] = {
      PerfectHashHashMultiplyShiftRFunctionId,
      PerfectHashHashMultiplyShiftLRFunctionId,
      PerfectHashHashMultiplyShiftRMultiplyFunctionId,
      PerfectHashHashMultiplyShiftR2FunctionId,
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
  compileFlags.JitIndex32x16 = TRUE;

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

  if (!info.Flags.Index32x16Compiled) {
    jit->Vtbl->Release(jit);
    shim->Vtbl->Release(table);
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
