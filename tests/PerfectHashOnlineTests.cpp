#include <gtest/gtest.h>

#include <PerfectHash/PerfectHash.h>

#include "PerfectHashGraphImpl4TestConfig.h"

#include <algorithm>
#include <chrono>
#include <cctype>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <fstream>
#include <optional>
#include <string>
#include <system_error>
#include <unordered_set>
#include <vector>

#ifdef PH_WINDOWS
#include <process.h>
#else
#include <unistd.h>
#endif

namespace {

namespace fs = std::filesystem;

#ifdef PH_WINDOWS
#define PH_TEST_IID(Name) IID_##Name
#else
#define PH_TEST_IID(Name) &IID_##Name
#endif

constexpr const char *kGraphImpl4Sparse32ReloadSeedsArg =
    "--Seeds=0x8A696C6B,0x5CC6B6A7,0x608";
constexpr const char *kGraphImpl4Downsized64ReloadSeedsArg =
    "--Seeds=0xDDE48099,0x20C4AF8D,0x608";
constexpr const char *kGraphImpl3Downsized64ReloadSeedsArg =
    "--Seeds=0x9F5ABF20,0x0F97985A,0x608";
// Deterministic MultiplyShiftR/And seeds for the exact reload fixtures below.
// Each fixture has its own pinned seed triple so failures identify the key
// shape that needs regeneration.  Regenerate them if the solver path, key
// generators, or hash schedule changes and the guards fire.
// Regenerate by scanning i=1.. with Seed1 = 0x01234567 + i*0x9E3779B9,
// Seed2 = 0x89ABCDEF + i*0x85EBCA6B, and Seed3 = 0x608, then use the first
// successful PerfectHashCreate candidate for the matching key generator and
// GraphImpl.

#define ASSERT_RELOAD_FIXTURE_CREATE_SUCCEEDED(Result, SeedsArg)           \
  do {                                                                     \
    if ((Result) == PH_I_SOLVE_TIMEOUT_EXPIRED) {                          \
      GTEST_SKIP() << "reload fixture seeds timed out on this host; "      \
                   << "regenerate with the documented seed scan if this "  \
                   << "is persistent: " << (SeedsArg);                    \
    }                                                                      \
    ASSERT_NE((Result), PH_I_CREATE_TABLE_ROUTINE_FAILED_TO_FIND_SOLUTION) \
        << "reload fixture seeds did not solve; regenerate using the "     \
        << "documented seed scan: " << (SeedsArg);                        \
    ASSERT_GE((Result), 0)                                                 \
        << "reload fixture create failed for seeds: " << (SeedsArg);      \
  } while (false)

std::vector<ULONG> SeedValuesFromSeedsArg(const char *seeds_arg) {
  constexpr const char *prefix = "--Seeds=";
  std::string text(seeds_arg);
  std::vector<ULONG> seeds;

  if (text.rfind(prefix, 0) != 0) {
    ADD_FAILURE() << "Invalid seeds arg: " << text;
    return seeds;
  }

  size_t start = std::string(prefix).size();
  while (start < text.size()) {
    size_t comma = text.find(',', start);
    std::string token = text.substr(
        start,
        comma == std::string::npos ? std::string::npos : comma - start);
    char *end = nullptr;
    unsigned long value = std::strtoul(token.c_str(), &end, 0);
    if (token.empty() || end == token.c_str() || *end != '\0') {
      ADD_FAILURE() << "Invalid seed token in " << text << ": " << token;
      return {};
    }
    seeds.push_back(static_cast<ULONG>(value));
    if (comma == std::string::npos) {
      break;
    }
    start = comma + 1;
  }

  return seeds;
}

struct PerfectHashTableShim {
  PPERFECT_HASH_TABLE_VTBL Vtbl;
};

extern "C" {
ULONGLONG PerfectHashComposeGraphImpl4DownsizeBitmap(ULONGLONG outer_bitmap,
                                                     ULONG inner_bitmap);
VOID PerfectHashComputeDownsizeMetadataFromBitmap(ULONGLONG bitmap,
                                                  PBOOLEAN metadata_valid,
                                                  PULONGLONG shifted_mask,
                                                  PBYTE trailing_zeros,
                                                  PBOOLEAN contiguous);
}

template <typename T>
ULONGLONG BuildDownsizeMask(const std::vector<T> &keys) {
  ULONGLONG mask = 0;
  for (auto key : keys) {
    mask |= static_cast<ULONGLONG>(key);
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

template <typename T>
ULONGLONG BuildDownsizedKeyMask(const std::vector<T> &keys,
                                ULONGLONG downsize_mask) {
  ULONGLONG mask = 0;
  for (auto key : keys) {
    mask |= DownsizeKey(static_cast<ULONGLONG>(key), downsize_mask);
  }
  return mask;
}

bool IsContiguousBitmap(ULONGLONG mask) {
  if (mask == 0) {
    return false;
  }

  while ((mask & 1ull) == 0) {
    mask >>= 1;
  }

  while ((mask & 1ull) != 0) {
    mask >>= 1;
  }

  return mask == 0;
}

void AssertCollisionFreeIndexes(const std::vector<ULONG> &indexes,
                                size_t number_of_keys,
                                const char *label) {
  SCOPED_TRACE(label);
  ASSERT_EQ(indexes.size(), number_of_keys);

  std::unordered_set<ULONG> seen;
  seen.reserve(number_of_keys);
  for (ULONG index : indexes) {
    EXPECT_LT(index, number_of_keys);
    EXPECT_TRUE(seen.insert(index).second);
  }
  EXPECT_EQ(seen.size(), number_of_keys);
}

PERFECT_HASH_TABLE_COMPILE_FLAGS MakeLlvmJitCompileFlags() {
  PERFECT_HASH_TABLE_COMPILE_FLAGS flags = {0};

  //
  // PerfectHashOnlineCompileTable also forces Jit before table dispatch.  Set
  // both flags explicitly so these tests document the loaded-table LLVM
  // precondition and remain equivalent if routed through Table::Compile.
  //
  flags.Jit = TRUE;
  flags.JitBackendLlvm = TRUE;

  return flags;
}

template <typename T>
ULONGLONG ExpandSparseBits(T value, const std::vector<int> &bit_positions) {
  ULONGLONG result = 0;
  for (size_t index = 0; index < bit_positions.size(); ++index) {
    if ((static_cast<ULONGLONG>(value) & (1ull << index)) != 0) {
      result |= (1ull << bit_positions[index]);
    }
  }
  return result;
}

std::vector<ULONG> MakeSparse32Keys(size_t count) {
  const std::vector<int> bit_positions = {0, 3, 7, 12, 16, 21, 25, 29, 31};
  std::vector<ULONG> keys;
  keys.reserve(count);

  for (ULONG value = 1; keys.size() < count; ++value) {
    keys.push_back(static_cast<ULONG>(ExpandSparseBits(value, bit_positions)));
  }

  return keys;
}

std::vector<ULONGLONG> MakeSparseDownsized64KeysWithPositions(
    size_t count,
    const std::vector<int> &bit_positions) {
  std::vector<ULONGLONG> keys;
  keys.reserve(count);

  for (ULONG value = 1; keys.size() < count; ++value) {
    keys.push_back(ExpandSparseBits(value, bit_positions));
  }

  return keys;
}

std::vector<ULONGLONG> MakeSparseDownsized64Keys(size_t count) {
  const std::vector<int> bit_positions = {
      PERFECTHASH_GRAPHIMPL4_DOWNSIZED64_BIT_POSITIONS
  };
  return MakeSparseDownsized64KeysWithPositions(count, bit_positions);
}

template <typename T>
void WriteBinaryKeys(const fs::path &path, const std::vector<T> &keys) {
  std::ofstream file(path, std::ios::binary | std::ios::trunc);
  ASSERT_TRUE(file) << path;
  file.write(reinterpret_cast<const char *>(keys.data()),
             static_cast<std::streamsize>(keys.size() * sizeof(T)));
  ASSERT_TRUE(file.good()) << path;
}

fs::path MakeTestDirectory(const char *name) {
  auto ticks = std::chrono::steady_clock::now().time_since_epoch().count();
#ifdef PH_WINDOWS
  auto process_id = _getpid();
#else
  auto process_id = getpid();
#endif
  fs::path dir = fs::temp_directory_path() /
                 (std::string("perfecthash_") + name + "_" +
                  std::to_string(static_cast<long long>(process_id)) + "_" +
                  std::to_string(static_cast<long long>(ticks)));
  fs::remove_all(dir);
  fs::create_directories(dir);
  return dir;
}

class ScopedTestDirectory {
 public:
  explicit ScopedTestDirectory(const char *name) : path_(MakeTestDirectory(name)) {}

  ~ScopedTestDirectory() {
    std::error_code error;
    fs::remove_all(path_, error);
  }

  const fs::path &path() const { return path_; }

 private:
  fs::path path_;
};

class ScopedPhKeys {
 public:
  explicit ScopedPhKeys(PPERFECT_HASH_KEYS value = nullptr) : value_(value) {}
  ScopedPhKeys(const ScopedPhKeys &) = delete;
  ScopedPhKeys &operator=(const ScopedPhKeys &) = delete;

  ~ScopedPhKeys() {
    if (value_ != nullptr) {
      value_->Vtbl->Release(value_);
    }
  }

  PPERFECT_HASH_KEYS get() const { return value_; }

 private:
  PPERFECT_HASH_KEYS value_;
};

class ScopedPhTable {
 public:
  explicit ScopedPhTable(PPERFECT_HASH_TABLE value = nullptr) : value_(value) {}
  ScopedPhTable(const ScopedPhTable &) = delete;
  ScopedPhTable &operator=(const ScopedPhTable &) = delete;

  ~ScopedPhTable() {
    if (value_ != nullptr) {
      // PPERFECT_HASH_TABLE is opaque in the public C API, so table tests use
      // the local vtbl-prefix shim to release table instances.
      reinterpret_cast<PerfectHashTableShim *>(value_)->Vtbl->Release(value_);
    }
  }

  PPERFECT_HASH_TABLE get() const { return value_; }

 private:
  PPERFECT_HASH_TABLE value_;
};

class ScopedPhJit {
 public:
  explicit ScopedPhJit(PPERFECT_HASH_TABLE_JIT_INTERFACE value = nullptr) :
      value_(value) {}
  ScopedPhJit(const ScopedPhJit &) = delete;
  ScopedPhJit &operator=(const ScopedPhJit &) = delete;

  ~ScopedPhJit() {
    if (value_ != nullptr) {
      value_->Vtbl->Release(value_);
    }
  }

  PPERFECT_HASH_TABLE_JIT_INTERFACE get() const { return value_; }

 private:
  PPERFECT_HASH_TABLE_JIT_INTERFACE value_;
};

UNICODE_STRING MakeUnicodeString(std::wstring *value) {
  UNICODE_STRING string = {};
  string.Length = static_cast<USHORT>(value->size() * sizeof(WCHAR));
  string.MaximumLength = static_cast<USHORT>((value->size() + 1) * sizeof(WCHAR));
  string.Buffer = const_cast<PWSTR>(value->c_str());
  return string;
}

fs::path FindSingleTableFile(const fs::path &output_dir) {
  std::vector<fs::path> table_paths;
  for (const auto &entry : fs::recursive_directory_iterator(output_dir)) {
    if (!entry.is_regular_file()) {
      continue;
    }
    if (entry.path().extension() == ".pht1") {
      table_paths.push_back(entry.path());
    }
  }

  if (table_paths.size() != 1u) {
    ADD_FAILURE() << "Expected exactly one .pht1 file under " << output_dir
                  << ", found " << table_paths.size();
    return fs::path();
  }
  return table_paths[0];
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

TEST(CompiledPerfectHashPortableHelpers, ZeroRotateCountIsIdentity) {
  constexpr ULONG value32 = 0x98000101ul;
  constexpr ULONGLONG value64 = 0x9800010102040810ull;

  EXPECT_EQ(RotateLeft32_C(value32, 0), value32);
  EXPECT_EQ(RotateRight32_C(value32, 0), value32);
  EXPECT_EQ(RotateLeft32_C(value32, 32), value32);
  EXPECT_EQ(RotateRight32_C(value32, 32), value32);

  EXPECT_EQ(RotateLeft64_C(value64, 0), value64);
  EXPECT_EQ(RotateRight64_C(value64, 0), value64);
  EXPECT_EQ(RotateLeft64_C(value64, 64), value64);
  EXPECT_EQ(RotateRight64_C(value64, 64), value64);
}

TEST(GraphImpl4BitUtils, ContiguousBitmapDetection) {
  const ULONGLONG contiguous_composed_mask = 0x380ull;
  const ULONGLONG high_contiguous_mask = 0xffffffffffffff00ull;
  const ULONGLONG contiguous_samples[] = {
      0x080ull,
      0x100ull,
      0x180ull,
      0x380ull,
      0xffffull,
  };

  ASSERT_TRUE(IsContiguousBitmap(contiguous_composed_mask));
  ASSERT_FALSE(IsContiguousBitmap(0x81ull));
  ASSERT_TRUE(IsContiguousBitmap(high_contiguous_mask));
  ASSERT_FALSE(IsContiguousBitmap(0x8000000000000001ull));
  for (ULONGLONG sample : contiguous_samples) {
    ULONG via_shift = static_cast<ULONG>((sample >> 7) & 0x7ull);
    ULONG via_extract = DownsizeKey(sample, contiguous_composed_mask);
    EXPECT_EQ(via_extract, via_shift);
  }

  BOOLEAN metadata_valid = FALSE;
  BOOLEAN contiguous = FALSE;
  BYTE trailing_zeros = 0;
  ULONGLONG shifted_mask = 0;

  PerfectHashComputeDownsizeMetadataFromBitmap(high_contiguous_mask,
                                               &metadata_valid,
                                               &shifted_mask,
                                               &trailing_zeros,
                                               &contiguous);
  EXPECT_TRUE(metadata_valid);
  EXPECT_TRUE(contiguous);
  EXPECT_EQ(trailing_zeros, 8u);
  EXPECT_EQ(shifted_mask, 0x00ffffffffffffffull);

  ULONG high_via_shift =
      static_cast<ULONG>((0xffffffffffffff00ull >> trailing_zeros) &
                         shifted_mask);
  ULONG high_via_extract =
      DownsizeKey(0xffffffffffffff00ull, high_contiguous_mask);
  EXPECT_EQ(high_via_extract, high_via_shift);

  PerfectHashComputeDownsizeMetadataFromBitmap(0x8000000000000001ull,
                                               &metadata_valid,
                                               &shifted_mask,
                                               &trailing_zeros,
                                               &contiguous);
  EXPECT_TRUE(metadata_valid);
  EXPECT_FALSE(contiguous);
  EXPECT_EQ(shifted_mask, 0ull);
}

TEST(GraphImpl4BitUtils, ComposedDownsizeMetadataUsesComposedBitmap) {
  const ULONGLONG outer_mask = (1ull << 8) |
                               (1ull << 9) |
                               (1ull << 10) |
                               (1ull << 30) |
                               (1ull << 32);
  const ULONG inner_mask = 0x7u;
  const ULONGLONG composed_mask = 0x700ull;

  ASSERT_FALSE(IsContiguousBitmap(outer_mask));
  ASSERT_TRUE(IsContiguousBitmap(inner_mask));
  ASSERT_EQ(PerfectHashComposeGraphImpl4DownsizeBitmap(outer_mask, inner_mask),
            composed_mask);

  BOOLEAN metadata_valid = FALSE;
  BOOLEAN contiguous = FALSE;
  BYTE trailing_zeros = 0;
  ULONGLONG shifted_mask = 0;

  PerfectHashComputeDownsizeMetadataFromBitmap(composed_mask,
                                               &metadata_valid,
                                               &shifted_mask,
                                               &trailing_zeros,
                                               &contiguous);
  EXPECT_TRUE(metadata_valid);
  EXPECT_TRUE(contiguous);
  EXPECT_EQ(trailing_zeros, 8u);
  EXPECT_EQ(shifted_mask, 0x7ull);
}

TEST(GraphImpl4BitUtils,
     ComposedDownsizeMetadataRejectsInnerContiguityMismatch) {
  const ULONGLONG outer_mask = (1ull << 8) |
                               (1ull << 10) |
                               (1ull << 30) |
                               (1ull << 32);
  const ULONG inner_mask = 0x7u;
  const ULONGLONG composed_mask = (1ull << 8) |
                                  (1ull << 10) |
                                  (1ull << 30);

  ASSERT_FALSE(IsContiguousBitmap(outer_mask));
  ASSERT_TRUE(IsContiguousBitmap(inner_mask));
  ASSERT_EQ(PerfectHashComposeGraphImpl4DownsizeBitmap(outer_mask, inner_mask),
            composed_mask);
  ASSERT_FALSE(IsContiguousBitmap(composed_mask));

  BOOLEAN metadata_valid = FALSE;
  BOOLEAN contiguous = TRUE;
  BYTE trailing_zeros = 0;
  ULONGLONG shifted_mask = 0;

  PerfectHashComputeDownsizeMetadataFromBitmap(composed_mask,
                                               &metadata_valid,
                                               &shifted_mask,
                                               &trailing_zeros,
                                               &contiguous);
  EXPECT_TRUE(metadata_valid);
  EXPECT_FALSE(contiguous);
  EXPECT_EQ(shifted_mask, 0ull);

  for (ULONGLONG value = 0; value < 8; ++value) {
    ULONGLONG key = 0;
    if ((value & 1ull) != 0) {
      key |= (1ull << 8);
    }
    if ((value & 2ull) != 0) {
      key |= (1ull << 10);
    }
    if ((value & 4ull) != 0) {
      key |= (1ull << 30);
    }

    ULONG outer = DownsizeKey(key, outer_mask);
    ULONG two_step = DownsizeKey(outer, inner_mask);
    ULONG direct = DownsizeKey(key, composed_mask);
    EXPECT_EQ(two_step, direct);
  }
}

TEST(GraphImpl4BitUtils, ComposedExtractionMatchesTwoStepExtraction) {
  const ULONGLONG outer_mask = 0x1f00ull;
  const ULONG inner_mask = 0x15u;
  const ULONGLONG composed_mask = 0x1500ull;

  ASSERT_TRUE(IsContiguousBitmap(outer_mask));
  ASSERT_FALSE(IsContiguousBitmap(inner_mask));
  ASSERT_EQ(PerfectHashComposeGraphImpl4DownsizeBitmap(outer_mask, inner_mask),
            composed_mask);
  ASSERT_FALSE(IsContiguousBitmap(composed_mask));

  for (ULONGLONG value = 0; value < 32; ++value) {
    ULONGLONG key = value << 8;
    ULONG outer = DownsizeKey(key, outer_mask);
    ULONG two_step = DownsizeKey(outer, inner_mask);
    ULONG direct = DownsizeKey(key, composed_mask);
    EXPECT_EQ(two_step, direct);
  }
}

TEST(GraphImpl4BitUtils, Downsized64OuterBitmapProducesIdentityInnerBitmap) {
  const std::vector<int> bit_positions = {1, 3, 6, 10, 18, 24, 32, 41, 56};
  const auto keys = MakeSparseDownsized64KeysWithPositions(256,
                                                           bit_positions);
  const ULONGLONG outer_mask = BuildDownsizeMask(keys);
  const ULONGLONG downsized_mask = BuildDownsizedKeyMask(keys, outer_mask);
  const ULONGLONG expected_mask = (1ull << bit_positions.size()) - 1ull;

  //
  // Starting the raw 64-bit positions above bit 0 does not create a non-zero
  // trailing shift after outer downsizing.  Because the outer bitmap is the OR
  // of the same key set, every packed bit has at least one source key and the
  // GraphImpl4 inner bitmap is identity in the downsized64 domain.
  //

  ASSERT_EQ(outer_mask & 1ull, 0ull);
  EXPECT_EQ(downsized_mask, expected_mask);
  EXPECT_TRUE(IsContiguousBitmap(downsized_mask));
  EXPECT_NE(downsized_mask & 1ull, 0ull);
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
      const std::vector<ULONG> *seeds = nullptr,
      ULONG maxSolveTimeInSeconds = 0,
      bool useSystemRng = false,
      bool useRandomStartSeed = false,
      bool hashAllKeysFirst = false) {
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
    tableFlags.HashAllKeysFirst = hashAllKeysFirst ? TRUE : FALSE;

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
      bool allowAssigned16 = false,
      ULONG graphImpl = 0,
      ULONG maxSolveTimeInSeconds = 0,
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
    tableFlags.HashAllKeysFirst = FALSE;  // Mirrors --DoNotHashAllKeysFirst.

    if (graphImpl != 0) {
      PERFECT_HASH_TABLE_CREATE_PARAMETER param = {};
      param.Id = TableCreateParameterGraphImplId;
      param.AsULong = graphImpl;
      table_params.push_back(param);
    }

    if (maxSolveTimeInSeconds > 0) {
      PERFECT_HASH_TABLE_CREATE_PARAMETER param = {};
      param.Id = TableCreateParameterMaxSolveTimeInSecondsId;
      param.AsULong = maxSolveTimeInSeconds;
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
        sizeof(ULONGLONG),
        static_cast<ULONGLONG>(keys.size()),
        const_cast<ULONGLONG *>(keys.data()),
        &keysFlags,
        &tableFlags,
        tableCreateParamsPtr,
        &table);
    EXPECT_GE(result, 0);
    return table;
  }

  PERFECT_HASH_TABLE_FLAGS GetTableFlags(PPERFECT_HASH_TABLE table) {
    PERFECT_HASH_TABLE_FLAGS flags = {0};
    auto *shim = reinterpret_cast<PerfectHashTableShim *>(table);
    EXPECT_GE(shim->Vtbl->GetFlags(table, sizeof(flags), &flags), 0);
    return flags;
  }

  std::vector<ULONG> CaptureIndexes(
      PPERFECT_HASH_TABLE table,
      const std::vector<ULONG> &keys) {
    std::vector<ULONG> indexes;
    auto *shim = reinterpret_cast<PerfectHashTableShim *>(table);
    indexes.reserve(keys.size());

    for (ULONG key : keys) {
      ULONG index = 0;
      EXPECT_GE(shim->Vtbl->Index(table, key, &index), 0);
      indexes.push_back(index);
    }

    return indexes;
  }

  std::vector<ULONG> CaptureIndexes64(
      PPERFECT_HASH_TABLE table,
      const std::vector<ULONGLONG> &keys) {
    std::vector<ULONG> indexes;
    auto *shim = reinterpret_cast<PerfectHashTableShim *>(table);
    const ULONGLONG mask = BuildDownsizeMask(keys);
    indexes.reserve(keys.size());

    for (ULONGLONG key : keys) {
      ULONG downsized = DownsizeKey(key, mask);
      ULONG index = 0;
      EXPECT_GE(shim->Vtbl->Index(table, downsized, &index), 0);
      indexes.push_back(index);
    }

    return indexes;
  }

  std::optional<ULONGLONG> LoadKeyDownsizeBitmapFromHeader(
      const fs::path &table_path) {
    fs::path header_path = table_path;
    header_path.replace_extension(".h");

    std::ifstream file(header_path);
    EXPECT_TRUE(file.is_open()) << "Failed to open " << header_path;
    if (!file.is_open()) {
      return std::nullopt;
    }

    std::string macro_prefix = header_path.stem().string();
    std::transform(macro_prefix.begin(),
                   macro_prefix.end(),
                   macro_prefix.begin(),
                   [](unsigned char ch) {
                     return static_cast<char>(std::toupper(ch));
                   });

    const std::string key_downsize_needle =
        "#define " + macro_prefix + "_KEY_DOWNSIZE_BITMAP ";
    const std::string graphimpl4_needle =
        "#define " + macro_prefix + "_GRAPHIMPL4_KEY_DOWNSIZE_BITMAP ";

    std::vector<std::string> lines;
    std::string line;
    while (std::getline(file, line)) {
      lines.push_back(line);
    }

    //
    // Prefer the 64-bit key-downsize macro when present.  Sparse 32-bit
    // GraphImpl4 tables do not emit that macro, so fall back to the GraphImpl4
    // inner compact-key bitmap; the return below intentionally exits both loops.
    //
    for (const auto &needle : {key_downsize_needle, graphimpl4_needle}) {
      for (const auto &candidate : lines) {
        if (candidate.find(needle) != 0) {
          continue;
        }

        const std::string value = candidate.substr(needle.size());
        try {
          //
          // Generated hex macros are emitted with a 0x prefix by
          // OUTPUT_HEX_RAW; base 0 parsing below depends on that prefix.
          //
          return static_cast<ULONGLONG>(std::stoull(value, nullptr, 0));
        } catch (const std::exception &ex) {
          ADD_FAILURE() << "Failed to parse key downsize bitmap from "
                        << header_path << ": " << candidate << " ("
                        << ex.what() << ")";
          return std::nullopt;
        }
      }
    }

    ADD_FAILURE() << "Failed to find key downsize bitmap macro in "
                  << header_path;
    return std::nullopt;
  }

  HRESULT CreateFileBackedTableFromKeys(
      const fs::path &keys_path,
      const fs::path &output_dir,
      ULONG key_size_in_bytes,
      const std::vector<std::string> &extra_args = {},
      ULONG graph_impl = 4) {
    PPERFECT_HASH_CONTEXT context = nullptr;
    auto createInstance = classFactory_->Vtbl->CreateInstance;
    HRESULT result = createInstance(
        classFactory_,
        nullptr,
        PH_TEST_IID(PERFECT_HASH_CONTEXT),
        reinterpret_cast<void **>(&context));
    EXPECT_GE(result, 0);
    EXPECT_NE(context, nullptr);
    if (FAILED(result)) {
      if (context != nullptr) {
        context->Vtbl->Release(context);
      }
      return result;
    }
    if (context == nullptr) {
      return E_POINTER;
    }

    std::vector<std::string> args = {
        "PerfectHashCreateExe",
        keys_path.string(),
        output_dir.string(),
        "Chm01",
        "MultiplyShiftR",
        "And",
        "1",
        "--Quiet",
        "--DoNotHashAllKeysFirst",
        "--MaxSolveTimeInSeconds=5",
        "--KeySizeInBytes=" + std::to_string(key_size_in_bytes),
    };

    if (graph_impl != 0) {
      args.push_back("--GraphImpl=" + std::to_string(graph_impl));
    }

    args.insert(args.end(), extra_args.begin(), extra_args.end());

    std::vector<char *> argv;
    argv.reserve(args.size());
    for (auto &arg : args) {
      argv.push_back(arg.data());
    }

    result = context->Vtbl->TableCreateArgvA(
        context,
        static_cast<int>(argv.size()),
        argv.data());

    context->Vtbl->Release(context);
    return result;
  }

  PPERFECT_HASH_KEYS LoadKeysFromPath(
      const fs::path &keys_path,
      ULONG key_size_in_bytes) {
    PPERFECT_HASH_KEYS keys = nullptr;
    auto createInstance = classFactory_->Vtbl->CreateInstance;
    HRESULT result = createInstance(
        classFactory_,
        nullptr,
        PH_TEST_IID(PERFECT_HASH_KEYS),
        reinterpret_cast<void **>(&keys));
    EXPECT_GE(result, 0);
    EXPECT_NE(keys, nullptr);
    if (FAILED(result)) {
      if (keys != nullptr) {
        keys->Vtbl->Release(keys);
      }
      return nullptr;
    }
    if (keys == nullptr) {
      return nullptr;
    }

    std::wstring keys_path_w = keys_path.wstring();
    UNICODE_STRING keys_path_string = MakeUnicodeString(&keys_path_w);
    PERFECT_HASH_KEYS_LOAD_FLAGS keys_load_flags = {0};

    result = keys->Vtbl->Load(keys,
                              &keys_load_flags,
                              &keys_path_string,
                              key_size_in_bytes);
    EXPECT_GE(result, 0);
    if (FAILED(result)) {
      keys->Vtbl->Release(keys);
      return nullptr;
    }

    return keys;
  }

  PPERFECT_HASH_TABLE LoadTableFromPath(
      const fs::path &table_path,
      PPERFECT_HASH_KEYS keys = nullptr,
      HRESULT *load_result = nullptr,
      bool allow_not_implemented = false) {
    PPERFECT_HASH_TABLE table = nullptr;
    if (load_result != nullptr) {
      *load_result = S_OK;
    }

    auto createInstance = classFactory_->Vtbl->CreateInstance;
    HRESULT result = createInstance(
        classFactory_,
        nullptr,
        PH_TEST_IID(PERFECT_HASH_TABLE),
        reinterpret_cast<void **>(&table));
    if (load_result != nullptr) {
      *load_result = result;
    }
    EXPECT_GE(result, 0);
    if (FAILED(result)) {
      if (table != nullptr) {
        auto *shim = reinterpret_cast<PerfectHashTableShim *>(table);
        shim->Vtbl->Release(table);
      }
      return nullptr;
    }
    if (table == nullptr) {
      if (load_result != nullptr) {
        *load_result = E_POINTER;
      }
      return nullptr;
    }

    std::wstring table_path_w = table_path.wstring();
    UNICODE_STRING table_path_string = MakeUnicodeString(&table_path_w);
    PERFECT_HASH_TABLE_LOAD_FLAGS load_flags = {0};
    auto *shim = reinterpret_cast<PerfectHashTableShim *>(table);
    result = shim->Vtbl->Load(table, &load_flags, &table_path_string, keys);
    if (load_result != nullptr) {
      *load_result = result;
    }
    if (!(allow_not_implemented && result == PH_E_NOT_IMPLEMENTED)) {
      EXPECT_GE(result, 0);
    }
    if (FAILED(result)) {
      shim->Vtbl->Release(table);
      return nullptr;
    }

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
  //
  // The keyed and keyless reload paths consume the same persisted metadata.
  // In addition to cross-checking the two paths agree, assert that the
  // keyless path still maps this key set into a dense, collision-free range.
  //
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

TEST_F(PerfectHashOnlineTests, GraphImpl4Assigned8RequiresOptIn) {
  const auto keys = MakePseudoRandomKeys(8, 0x2468ACE0u);

  PPERFECT_HASH_TABLE tableGraphImpl4 = CreateTableFromKeys(
      keys,
      PerfectHashHashMultiplyShiftRFunctionId,
      true,
      4,
      nullptr,
      5);
  ASSERT_NE(tableGraphImpl4, nullptr);

  auto *shimGraphImpl4 = reinterpret_cast<PerfectHashTableShim *>(tableGraphImpl4);
  PERFECT_HASH_TABLE_FLAGS flagsGraphImpl4 = {0};
  ASSERT_GE(
      shimGraphImpl4->Vtbl->GetFlags(
          tableGraphImpl4,
          sizeof(flagsGraphImpl4),
          &flagsGraphImpl4),
      0);
  EXPECT_EQ(flagsGraphImpl4.AssignedElementSizeInBits << 3, 8u);
  EXPECT_EQ(shimGraphImpl4->Vtbl->SlowIndex, nullptr);

  std::unordered_set<ULONG> seen;
  seen.reserve(keys.size());
  for (ULONG key : keys) {
    ULONG index = 0;
    ASSERT_GE(shimGraphImpl4->Vtbl->Index(tableGraphImpl4, key, &index), 0);
    EXPECT_TRUE(seen.insert(index).second);
  }

  shimGraphImpl4->Vtbl->Release(tableGraphImpl4);

  PPERFECT_HASH_TABLE tableOptOut = CreateTableFromKeys(
      keys,
      PerfectHashHashMultiplyShiftRFunctionId,
      false,
      4,
      nullptr,
      5);
  ASSERT_NE(tableOptOut, nullptr);

  auto *shimOptOut = reinterpret_cast<PerfectHashTableShim *>(tableOptOut);
  PERFECT_HASH_TABLE_FLAGS flagsOptOut = {0};
  ASSERT_GE(shimOptOut->Vtbl->GetFlags(tableOptOut, sizeof(flagsOptOut), &flagsOptOut), 0);
  EXPECT_EQ(flagsOptOut.AssignedElementSizeInBits << 3, 32u);
  shimOptOut->Vtbl->Release(tableOptOut);
}

TEST_F(PerfectHashOnlineTests, GraphImpl4SupportsDownsized64BitInputs) {
  const std::vector<ULONGLONG> keys = {
      1ull, 3ull, 5ull, 7ull, 11ull, 13ull, 17ull, 19ull,
      23ull, 29ull, 31ull, 37ull, 41ull, 43ull, 47ull, 53ull,
  };

  PPERFECT_HASH_TABLE table = CreateTableFromKeys64(
      keys,
      PerfectHashHashMulshrolate3RXFunctionId,
      true,
      4,
      5);
  ASSERT_NE(table, nullptr);

  auto *shim = reinterpret_cast<PerfectHashTableShim *>(table);
  PERFECT_HASH_TABLE_FLAGS flags = {0};
  ASSERT_GE(shim->Vtbl->GetFlags(table, sizeof(flags), &flags), 0);
  EXPECT_EQ(flags.AssignedElementSizeInBits << 3, 8u);

  const ULONGLONG mask = BuildDownsizeMask(keys);
  std::unordered_set<ULONG> seen;
  seen.reserve(keys.size());

  for (auto key : keys) {
    ULONG downsized = DownsizeKey(key, mask);
    ULONG index = 0;
    ASSERT_GE(shim->Vtbl->Index(table, downsized, &index), 0);
    EXPECT_TRUE(seen.insert(index).second);
  }

  shim->Vtbl->Release(table);
}

TEST_F(PerfectHashOnlineTests, GraphImpl4FileBackedReloadPreservesSparse32Compaction) {
  const ScopedTestDirectory root("graphimpl4_sparse32_reload");
  const auto keys_path = root.path() / "GraphImpl4Sparse32.keys";
  const auto output_dir = root.path() / "out";
  const auto keys = MakeSparse32Keys(256);

  WriteBinaryKeys(keys_path, keys);
  HRESULT result = CreateFileBackedTableFromKeys(keys_path,
                                                 output_dir,
                                                 sizeof(ULONG),
                                                 {kGraphImpl4Sparse32ReloadSeedsArg});
  ASSERT_RELOAD_FIXTURE_CREATE_SUCCEEDED(result,
                                         kGraphImpl4Sparse32ReloadSeedsArg);

  fs::path table_path = FindSingleTableFile(output_dir);
  ASSERT_FALSE(table_path.empty());

  ScopedPhKeys reference_keys(LoadKeysFromPath(keys_path, sizeof(ULONG)));
  ASSERT_NE(reference_keys.get(), nullptr);

  ScopedPhTable reference_table(LoadTableFromPath(table_path,
                                                 reference_keys.get()));
  ASSERT_NE(reference_table.get(), nullptr);

  ScopedPhTable table(LoadTableFromPath(table_path));
  ASSERT_NE(table.get(), nullptr);

  auto *reference_shim =
      reinterpret_cast<PerfectHashTableShim *>(reference_table.get());
  auto *shim = reinterpret_cast<PerfectHashTableShim *>(table.get());
  std::vector<ULONG> reference_indexes;
  std::vector<ULONG> keyless_indexes;
  reference_indexes.reserve(keys.size());
  keyless_indexes.reserve(keys.size());

  const ULONGLONG mask = BuildDownsizeMask(keys);
  const auto header_mask = LoadKeyDownsizeBitmapFromHeader(table_path);
  ASSERT_TRUE(header_mask.has_value());
  ASSERT_EQ(mask, *header_mask);
  ASSERT_FALSE(IsContiguousBitmap(mask));

  //
  // This sparse32 reload test is the file-backed coverage for non-identity
  // GraphImpl4 inner compact-key metadata.  There is no outer 64-bit downsize
  // here, so the header loader falls back to _GRAPHIMPL4_KEY_DOWNSIZE_BITMAP.
  //

  for (size_t index = 0; index < keys.size(); ++index) {
    ULONG key = keys[index];
    ULONG expected = 0;
    ULONG actual = 0;
    ASSERT_GE(reference_shim->Vtbl->Index(reference_table.get(),
                                          key,
                                          &expected), 0);
    ASSERT_GE(shim->Vtbl->Index(table.get(), key, &actual), 0);
    EXPECT_EQ(expected, actual);
    EXPECT_LT(expected, keys.size());
    EXPECT_LT(actual, keys.size());
    reference_indexes.push_back(expected);
    keyless_indexes.push_back(actual);
  }

  AssertCollisionFreeIndexes(reference_indexes,
                             keys.size(),
                             "keyed sparse32 reload");
  AssertCollisionFreeIndexes(keyless_indexes,
                             keys.size(),
                             "keyless sparse32 reload");
}

TEST_F(PerfectHashOnlineTests, GraphImpl4FileBackedReloadPreservesDownsized64Compaction) {
  const ScopedTestDirectory root("graphimpl4_downsized64_reload_nojit");
  const auto keys_path = root.path() / "GraphImpl4Downsized64.keys";
  const auto output_dir = root.path() / "out";
  const auto keys = MakeSparseDownsized64Keys(256);

  WriteBinaryKeys(keys_path, keys);
  HRESULT result = CreateFileBackedTableFromKeys(keys_path,
                                                 output_dir,
                                                 sizeof(ULONGLONG),
                                                 {kGraphImpl4Downsized64ReloadSeedsArg});
  ASSERT_RELOAD_FIXTURE_CREATE_SUCCEEDED(
      result,
      kGraphImpl4Downsized64ReloadSeedsArg);

  fs::path table_path = FindSingleTableFile(output_dir);
  ASSERT_FALSE(table_path.empty());

  ScopedPhKeys loaded_keys(LoadKeysFromPath(keys_path, sizeof(ULONGLONG)));
  ASSERT_NE(loaded_keys.get(), nullptr);

  HRESULT reference_load_result = S_OK;
  ScopedPhTable reference_table(LoadTableFromPath(table_path,
                                                 loaded_keys.get(),
                                                 &reference_load_result));
  ASSERT_EQ(reference_load_result, S_OK);
  ASSERT_NE(reference_table.get(), nullptr);

  HRESULT keyless_load_result = S_OK;
  ScopedPhTable table(LoadTableFromPath(table_path,
                                       nullptr,
                                       &keyless_load_result));
  ASSERT_EQ(keyless_load_result, S_OK);
  ASSERT_NE(table.get(), nullptr);

  auto *reference_shim =
      reinterpret_cast<PerfectHashTableShim *>(reference_table.get());
  auto *shim = reinterpret_cast<PerfectHashTableShim *>(table.get());
  const ULONGLONG mask = BuildDownsizeMask(keys);
  const auto create_seeds =
      SeedValuesFromSeedsArg(kGraphImpl4Downsized64ReloadSeedsArg);
  ASSERT_FALSE(create_seeds.empty());
  ScopedPhTable created_table(CreateTableFromKeys64(
      keys,
      PerfectHashHashMultiplyShiftRFunctionId,
      false,
      4,
      5,
      &create_seeds));
  ASSERT_NE(created_table.get(), nullptr);
  auto *created_shim =
      reinterpret_cast<PerfectHashTableShim *>(created_table.get());
  std::vector<ULONG> reference_indexes;
  std::vector<ULONG> keyless_indexes;
  std::vector<ULONG> created_indexes;
  reference_indexes.reserve(keys.size());
  keyless_indexes.reserve(keys.size());
  created_indexes.reserve(keys.size());

  const auto header_mask = LoadKeyDownsizeBitmapFromHeader(table_path);
  ASSERT_TRUE(header_mask.has_value());
  ASSERT_EQ(mask, *header_mask);
  ASSERT_FALSE(IsContiguousBitmap(mask));

  //
  // The loaded-table Index() API is 32-bit.  For downsized-64 GraphImpl4
  // tables, callers pass the outer-bitmap result rather than the raw 64-bit
  // key; the GraphImpl4 vtbl then applies the inner compact-key step.
  // The in-memory table anchors reload correctness to creation-time state so a
  // shared keyed/keyless reload bug cannot make this test pass by agreement.
  //

  for (size_t index = 0; index < keys.size(); ++index) {
    ULONGLONG key = keys[index];
    ULONG created = 0;
    ULONG expected = 0;
    ULONG actual = 0;
    ULONG downsized = DownsizeKey(key, mask);

    ASSERT_GE(created_shim->Vtbl->Index(created_table.get(),
                                        downsized,
                                        &created), 0);
    ASSERT_GE(reference_shim->Vtbl->Index(reference_table.get(),
                                          downsized,
                                          &expected), 0);
    ASSERT_GE(shim->Vtbl->Index(table.get(), downsized, &actual), 0);
    EXPECT_EQ(created, expected);
    EXPECT_EQ(created, actual);
    EXPECT_EQ(expected, actual);
    EXPECT_LT(created, keys.size());
    EXPECT_LT(expected, keys.size());
    EXPECT_LT(actual, keys.size());
    created_indexes.push_back(created);
    reference_indexes.push_back(expected);
    keyless_indexes.push_back(actual);
  }

  AssertCollisionFreeIndexes(created_indexes,
                             keys.size(),
                             "created downsized64 reference");
  AssertCollisionFreeIndexes(reference_indexes,
                             keys.size(),
                             "keyed downsized64 reload");
  AssertCollisionFreeIndexes(keyless_indexes,
                             keys.size(),
                             "keyless downsized64 reload");
}

TEST_F(PerfectHashOnlineTests, NonGraphImplFileBackedReloadPreservesDownsized64Metadata) {
  const ScopedTestDirectory root("graphimpl3_downsized64_reload_nojit");
  const auto keys_path = root.path() / "GraphImpl3Downsized64.keys";
  const auto output_dir = root.path() / "out";
  const auto keys = MakeSparseDownsized64Keys(128);

  WriteBinaryKeys(keys_path, keys);
  HRESULT result = CreateFileBackedTableFromKeys(keys_path,
                                                 output_dir,
                                                 sizeof(ULONGLONG),
                                                 {kGraphImpl3Downsized64ReloadSeedsArg},
                                                 3);
  ASSERT_RELOAD_FIXTURE_CREATE_SUCCEEDED(
      result,
      kGraphImpl3Downsized64ReloadSeedsArg);

  fs::path table_path = FindSingleTableFile(output_dir);
  ASSERT_FALSE(table_path.empty());

  ScopedPhKeys loaded_keys(LoadKeysFromPath(keys_path, sizeof(ULONGLONG)));
  ASSERT_NE(loaded_keys.get(), nullptr);

  ScopedPhTable reference_table(LoadTableFromPath(table_path,
                                                 loaded_keys.get()));
  ASSERT_NE(reference_table.get(), nullptr);

  ScopedPhTable table(LoadTableFromPath(table_path));
  ASSERT_NE(table.get(), nullptr);

  auto *reference_shim =
      reinterpret_cast<PerfectHashTableShim *>(reference_table.get());
  auto *shim = reinterpret_cast<PerfectHashTableShim *>(table.get());
  const ULONGLONG mask = BuildDownsizeMask(keys);
  const auto create_seeds =
      SeedValuesFromSeedsArg(kGraphImpl3Downsized64ReloadSeedsArg);
  ASSERT_FALSE(create_seeds.empty());
  ScopedPhTable created_table(CreateTableFromKeys64(
      keys,
      PerfectHashHashMultiplyShiftRFunctionId,
      false,
      3,
      5,
      &create_seeds));
  ASSERT_NE(created_table.get(), nullptr);
  auto *created_shim =
      reinterpret_cast<PerfectHashTableShim *>(created_table.get());
  const auto header_mask = LoadKeyDownsizeBitmapFromHeader(table_path);
  ASSERT_TRUE(header_mask.has_value());
  ASSERT_EQ(mask, *header_mask);
  ASSERT_FALSE(IsContiguousBitmap(mask));

  std::unordered_set<ULONG> created_seen;
  std::unordered_set<ULONG> seen;
  created_seen.reserve(keys.size());
  seen.reserve(keys.size());
  for (ULONGLONG key : keys) {
    ULONG created = 0;
    ULONG expected = 0;
    ULONG actual = 0;
    ULONG downsized = DownsizeKey(key, mask);

    ASSERT_GE(created_shim->Vtbl->Index(created_table.get(),
                                        downsized,
                                        &created), 0);
    ASSERT_GE(reference_shim->Vtbl->Index(reference_table.get(),
                                          downsized,
                                          &expected), 0);
    ASSERT_GE(shim->Vtbl->Index(table.get(), downsized, &actual), 0);
    EXPECT_EQ(created, expected);
    EXPECT_EQ(created, actual);
    EXPECT_EQ(expected, actual);
    EXPECT_LT(created, keys.size());
    EXPECT_LT(expected, keys.size());
    EXPECT_LT(actual, keys.size());
    EXPECT_TRUE(created_seen.insert(created).second);
    EXPECT_TRUE(seen.insert(actual).second);
  }

  EXPECT_EQ(created_seen.size(), keys.size());
  EXPECT_EQ(seen.size(), keys.size());
}

#if defined(PH_HAS_LLVM)
TEST_F(PerfectHashOnlineTests, GraphImpl4FileBackedReloadSparse32JitIndex32) {
  const ScopedTestDirectory root("graphimpl4_sparse32_reload_jit");
  const auto keys_path = root.path() / "GraphImpl4Sparse32.keys";
  const auto output_dir = root.path() / "out";
  const auto keys = MakeSparse32Keys(256);

  WriteBinaryKeys(keys_path, keys);
  HRESULT result = CreateFileBackedTableFromKeys(keys_path,
                                                 output_dir,
                                                 sizeof(ULONG),
                                                 {kGraphImpl4Sparse32ReloadSeedsArg});
  ASSERT_RELOAD_FIXTURE_CREATE_SUCCEEDED(result,
                                         kGraphImpl4Sparse32ReloadSeedsArg);

  fs::path table_path = FindSingleTableFile(output_dir);
  ASSERT_FALSE(table_path.empty());

  ScopedPhTable table(LoadTableFromPath(table_path));
  ASSERT_NE(table.get(), nullptr);

  auto *shim = reinterpret_cast<PerfectHashTableShim *>(table.get());
  const auto expected = CaptureIndexes(table.get(), keys);

  PERFECT_HASH_TABLE_COMPILE_FLAGS compile_flags = MakeLlvmJitCompileFlags();
  result = online_->Vtbl->CompileTable(online_,
                                       table.get(),
                                       &compile_flags);
  if (result == PH_E_LLVM_BACKEND_NOT_FOUND) {
    GTEST_SKIP() << "LLVM backend not found on this host.";
  }
  //
  // Backend availability is the only skip case.  Any other compile failure is
  // specific to loaded sparse32 GraphImpl4 JIT state and should fail loudly.
  //
  ASSERT_EQ(result, S_OK)
      << "LLVM compile of keyless sparse32 table failed: 0x"
      << std::hex << result;

  std::vector<ULONG> indexes;
  indexes.reserve(keys.size());
  for (size_t index = 0; index < keys.size(); ++index) {
    ULONG key = keys[index];
    ULONG actual = 0;

    ASSERT_GE(shim->Vtbl->Index(table.get(), key, &actual), 0);
    ASSERT_LT(index, expected.size());
    EXPECT_EQ(expected[index], actual);
    indexes.push_back(actual);
  }

  AssertCollisionFreeIndexes(indexes,
                             keys.size(),
                             "keyless sparse32 reload jit");
}

TEST_F(PerfectHashOnlineTests, GraphImpl4FileBackedReloadDownsized64JitIndex64) {
  const ScopedTestDirectory root("graphimpl4_downsized64_reload");
  const auto keys_path = root.path() / "GraphImpl4Downsized64.keys";
  const auto output_dir = root.path() / "out";
  const auto keys = MakeSparseDownsized64Keys(256);

  WriteBinaryKeys(keys_path, keys);
  HRESULT result = CreateFileBackedTableFromKeys(keys_path,
                                                 output_dir,
                                                 sizeof(ULONGLONG),
                                                 {kGraphImpl4Downsized64ReloadSeedsArg});
  ASSERT_RELOAD_FIXTURE_CREATE_SUCCEEDED(
      result,
      kGraphImpl4Downsized64ReloadSeedsArg);

  fs::path table_path = FindSingleTableFile(output_dir);
  ASSERT_FALSE(table_path.empty());

  ScopedPhTable table(LoadTableFromPath(table_path));
  ASSERT_NE(table.get(), nullptr);

  PERFECT_HASH_TABLE_COMPILE_FLAGS compile_flags = MakeLlvmJitCompileFlags();
  compile_flags.JitIndex64 = TRUE;

  result = online_->Vtbl->CompileTable(online_,
                                       table.get(),
                                       &compile_flags);
  if (result == PH_E_LLVM_BACKEND_NOT_FOUND) {
    GTEST_SKIP() << "LLVM backend not found on this host.";
  }
  ASSERT_EQ(result, S_OK)
      << "LLVM compile of keyless loaded table failed: 0x"
      << std::hex << result;

  PPERFECT_HASH_TABLE_JIT_INTERFACE raw_jit = nullptr;
  auto *shim = reinterpret_cast<PerfectHashTableShim *>(table.get());
  result = shim->Vtbl->QueryInterface(
      table.get(),
      PH_TEST_IID(PERFECT_HASH_TABLE_JIT_INTERFACE),
      reinterpret_cast<void **>(&raw_jit));
  ScopedPhJit jit(raw_jit);
  ASSERT_GE(result, 0);
  ASSERT_NE(jit.get(), nullptr);

  const ULONGLONG mask = BuildDownsizeMask(keys);
  const ULONGLONG downsized_mask = BuildDownsizedKeyMask(keys, mask);

  ASSERT_FALSE(IsContiguousBitmap(mask));
  ASSERT_TRUE(IsContiguousBitmap(downsized_mask));

  std::vector<ULONG> keyless_indexes;
  keyless_indexes.reserve(keys.size());
  std::unordered_set<ULONG> keyless_seen;
  keyless_seen.reserve(keys.size());

  for (size_t index = 0; index < keys.size(); ++index) {
    ULONGLONG key = keys[index];
    ULONG reloaded = 0;
    ULONG actual = 0;
    ULONG downsized = DownsizeKey(key, mask);

    //
    // Index() receives the already-outer-downsized 32-bit key here.  Comparing
    // it with Index64() validates the outer-downsize plus inner-compact
    // decomposition used by loaded-table JIT.
    //
    ASSERT_GE(shim->Vtbl->Index(table.get(), downsized, &reloaded), 0);
    ASSERT_GE(jit.get()->Vtbl->Index64(jit.get(), key, &actual), 0);
    EXPECT_EQ(reloaded, actual);
    EXPECT_LT(reloaded, keys.size());
    EXPECT_LT(actual, keys.size());
    EXPECT_TRUE(keyless_seen.insert(actual).second);
    keyless_indexes.push_back(actual);
  }
  EXPECT_EQ(keyless_seen.size(), keys.size());

  ScopedPhKeys loaded_keys(LoadKeysFromPath(keys_path, sizeof(ULONGLONG)));
  ASSERT_NE(loaded_keys.get(), nullptr);
  const auto header_mask = LoadKeyDownsizeBitmapFromHeader(table_path);
  ASSERT_TRUE(header_mask.has_value());
  ASSERT_EQ(mask, *header_mask);

  ScopedPhTable keyed_table(LoadTableFromPath(table_path, loaded_keys.get()));
  ASSERT_NE(keyed_table.get(), nullptr);

  result = online_->Vtbl->CompileTable(online_,
                                       keyed_table.get(),
                                       &compile_flags);
  //
  // The keyless compile above already proved LLVM availability.  A failure here
  // is keyed-load specific, even if it reports backend availability, and should
  // fail loudly rather than skip.
  //
  ASSERT_EQ(result, S_OK)
      << "LLVM compile of keyed-loaded table failed: 0x"
      << std::hex << result;

  PPERFECT_HASH_TABLE_JIT_INTERFACE raw_keyed_jit = nullptr;
  auto *keyed_shim =
      reinterpret_cast<PerfectHashTableShim *>(keyed_table.get());
  result = keyed_shim->Vtbl->QueryInterface(
      keyed_table.get(),
      PH_TEST_IID(PERFECT_HASH_TABLE_JIT_INTERFACE),
      reinterpret_cast<void **>(&raw_keyed_jit));
  ScopedPhJit keyed_jit(raw_keyed_jit);
  ASSERT_GE(result, 0);
  ASSERT_NE(keyed_jit.get(), nullptr);

  std::unordered_set<ULONG> keyed_seen;
  keyed_seen.reserve(keys.size());

  for (size_t index = 0; index < keys.size(); ++index) {
    ULONGLONG key = keys[index];
    ULONG reloaded = 0;
    ULONG actual = 0;
    ULONG downsized = DownsizeKey(key, mask);

    ASSERT_GE(keyed_shim->Vtbl->Index(keyed_table.get(),
                                      downsized,
                                      &reloaded), 0);
    ASSERT_GE(keyed_jit.get()->Vtbl->Index64(keyed_jit.get(),
                                             key,
                                             &actual), 0);
    ASSERT_LT(index, keyless_indexes.size());
    EXPECT_EQ(keyless_indexes[index], reloaded);
    EXPECT_EQ(keyless_indexes[index], actual);
    EXPECT_LT(reloaded, keys.size());
    EXPECT_LT(actual, keys.size());
    EXPECT_TRUE(keyed_seen.insert(actual).second);
  }
  EXPECT_EQ(keyed_seen.size(), keys.size());
}
#endif

TEST_F(PerfectHashOnlineTests, GraphImpl4RejectsNonGoodHashes) {
  const auto keys = MakePseudoRandomKeys(64, 0x13579BDFu);
  PERFECT_HASH_KEYS_LOAD_FLAGS keysFlags = {0};
  PERFECT_HASH_TABLE_CREATE_FLAGS tableFlags = {0};
  PERFECT_HASH_TABLE_CREATE_PARAMETER param = {};
  PERFECT_HASH_TABLE_CREATE_PARAMETERS params = {};
  PPERFECT_HASH_TABLE table = nullptr;

  tableFlags.NoFileIo = TRUE;
  tableFlags.Quiet = TRUE;
  tableFlags.DoNotTryUseHash16Impl = FALSE;

  param.Id = TableCreateParameterGraphImplId;
  param.AsULong = 4;
  params.SizeOfStruct = sizeof(params);
  params.NumberOfElements = 1;
  params.Params = &param;

  HRESULT result = online_->Vtbl->CreateTableFromKeys(
      online_,
      PerfectHashChm01AlgorithmId,
      PerfectHashHashJenkinsFunctionId,
      PerfectHashAndMaskFunctionId,
      sizeof(ULONG),
      static_cast<ULONGLONG>(keys.size()),
      const_cast<ULONG *>(keys.data()),
      &keysFlags,
      &tableFlags,
      &params,
      &table);

  EXPECT_EQ(result, PH_E_NOT_IMPLEMENTED);
  EXPECT_EQ(table, nullptr);
}

TEST_F(PerfectHashOnlineTests, GraphImpl4Assigned8JitRejected) {
  const auto keys = MakePseudoRandomKeys(8, 0xD00DFEEDu);

  PPERFECT_HASH_TABLE table = CreateTableFromKeys(
      keys,
      PerfectHashHashMultiplyShiftRFunctionId,
      true,
      4,
      nullptr,
      5);
  ASSERT_NE(table, nullptr);

  auto *shim = reinterpret_cast<PerfectHashTableShim *>(table);
  PERFECT_HASH_TABLE_FLAGS flags = GetTableFlags(table);
  ASSERT_EQ(flags.AssignedElementSizeInBits << 3, 8u);

  bool tested_backend = false;

#if defined(PH_HAS_LLVM)
  {
    PERFECT_HASH_TABLE_COMPILE_FLAGS compileFlags = {0};
    compileFlags.JitBackendLlvm = TRUE;
    EXPECT_EQ(online_->Vtbl->CompileTable(online_, table, &compileFlags),
              PH_E_NOT_IMPLEMENTED);
    tested_backend = true;
  }
#endif

#if defined(PH_HAS_RAWDOG_JIT)
  {
    PERFECT_HASH_TABLE_COMPILE_FLAGS compileFlags = {0};
    compileFlags.JitBackendRawDog = TRUE;
    EXPECT_EQ(online_->Vtbl->CompileTable(online_, table, &compileFlags),
              PH_E_NOT_IMPLEMENTED);
    tested_backend = true;
  }
#endif

  if (!tested_backend) {
    shim->Vtbl->Release(table);
    GTEST_SKIP() << "No JIT backend enabled for this build.";
  }

  shim->Vtbl->Release(table);
}

#if defined(PH_HAS_RAWDOG_JIT)
TEST_F(PerfectHashOnlineTests, GraphImpl4RawDogJitMatchesIndexAssigned32) {
  const auto keys = MakePseudoRandomKeys(64, 0x31415926u);

  ScopedPhTable table(CreateTableFromKeys(
      keys,
      PerfectHashHashMultiplyShiftRFunctionId,
      false,
      4,
      nullptr,
      5));
  ASSERT_NE(table.get(), nullptr);

  auto *shim = reinterpret_cast<PerfectHashTableShim *>(table.get());
  PERFECT_HASH_TABLE_FLAGS flags = GetTableFlags(table.get());
  ASSERT_EQ(flags.AssignedElementSizeInBits << 3, 32u);
  ASSERT_EQ(shim->Vtbl->SlowIndex, nullptr);

  const auto expected = CaptureIndexes(table.get(), keys);

  PERFECT_HASH_TABLE_COMPILE_FLAGS compileFlags = {0};
  compileFlags.JitBackendRawDog = TRUE;

  HRESULT result = online_->Vtbl->CompileTable(online_,
                                               table.get(),
                                               &compileFlags);
  if (result == PH_E_NOT_IMPLEMENTED) {
    GTEST_SKIP() << "RawDog GraphImpl4 scalar JIT unavailable on this host.";
  }
  ASSERT_GE(result, 0);

  for (size_t i = 0; i < keys.size(); ++i) {
    ULONG index = 0;
    ASSERT_GE(shim->Vtbl->Index(table.get(), keys[i], &index), 0);
    EXPECT_EQ(expected[i], index);
  }
}

TEST_F(PerfectHashOnlineTests, GraphImpl4RawDogJitMatchesIndexSparse32) {
  const auto keys = MakeSparse32Keys(256);

  ScopedPhTable table(CreateTableFromKeys(
      keys,
      PerfectHashHashMultiplyShiftRFunctionId,
      false,
      4,
      nullptr,
      5));
  ASSERT_NE(table.get(), nullptr);

  auto *shim = reinterpret_cast<PerfectHashTableShim *>(table.get());
  PERFECT_HASH_TABLE_FLAGS flags = GetTableFlags(table.get());
  ASSERT_EQ(flags.AssignedElementSizeInBits << 3, 32u);
  ASSERT_EQ(shim->Vtbl->SlowIndex, nullptr);

  const auto expected = CaptureIndexes(table.get(), keys);

  PERFECT_HASH_TABLE_COMPILE_FLAGS compileFlags = {0};
  compileFlags.JitBackendRawDog = TRUE;

  HRESULT result = online_->Vtbl->CompileTable(online_,
                                               table.get(),
                                               &compileFlags);
  if (result == PH_E_NOT_IMPLEMENTED) {
    GTEST_SKIP() << "RawDog GraphImpl4 scalar JIT unavailable on this host.";
  }
  ASSERT_GE(result, 0);

  for (size_t i = 0; i < keys.size(); ++i) {
    ULONG index = 0;
    ASSERT_GE(shim->Vtbl->Index(table.get(), keys[i], &index), 0);
    EXPECT_EQ(expected[i], index);
  }
}

TEST_F(PerfectHashOnlineTests, GraphImpl4RawDogIndex32x4MatchesIndexAssigned16) {
  const auto keys = MakeSparse32Keys(256);

  ScopedPhTable table(CreateTableFromKeys(
      keys,
      PerfectHashHashMultiplyShiftRFunctionId,
      true,
      4,
      nullptr,
      5));
  ASSERT_NE(table.get(), nullptr);

  auto *shim = reinterpret_cast<PerfectHashTableShim *>(table.get());
  PERFECT_HASH_TABLE_FLAGS flags = GetTableFlags(table.get());
  ASSERT_EQ(flags.AssignedElementSizeInBits << 3, 16u);
  ASSERT_EQ(shim->Vtbl->SlowIndex, nullptr);

  const auto expected = CaptureIndexes(table.get(), keys);

  PERFECT_HASH_TABLE_COMPILE_FLAGS compileFlags = {0};
  compileFlags.JitBackendRawDog = TRUE;
  compileFlags.JitIndex32x4 = TRUE;

  HRESULT result = online_->Vtbl->CompileTable(online_,
                                               table.get(),
                                               &compileFlags);
  if (result == PH_E_NOT_IMPLEMENTED) {
    GTEST_SKIP() << "RawDog GraphImpl4 Index32x4 unavailable on this host.";
  }
  ASSERT_GE(result, 0);

  PPERFECT_HASH_TABLE_JIT_INTERFACE raw_jit = nullptr;
  result = shim->Vtbl->QueryInterface(
      table.get(),
#ifdef PH_WINDOWS
      IID_PERFECT_HASH_TABLE_JIT_INTERFACE,
#else
      &IID_PERFECT_HASH_TABLE_JIT_INTERFACE,
#endif
      reinterpret_cast<void **>(&raw_jit));
  ScopedPhJit jit(raw_jit);
  ASSERT_GE(result, 0);
  ASSERT_NE(jit.get(), nullptr);

  PERFECT_HASH_TABLE_JIT_INFO info = {0};
  result = jit.get()->Vtbl->GetInfo(jit.get(), &info);
  ASSERT_GE(result, 0);
  EXPECT_TRUE(info.Flags.Index32x4Compiled);

  for (size_t i = 0; i < keys.size(); i += 4) {
    ULONG index1 = 0;
    ULONG index2 = 0;
    ULONG index3 = 0;
    ULONG index4 = 0;

    result = jit.get()->Vtbl->Index32x4(jit.get(),
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

}

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

TEST_F(PerfectHashOnlineTests, RawDogJitMulshrolate4RXMatchesSlowIndex) {
  const std::vector<ULONG> keys = {
      1, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53,
      59, 61, 67, 71, 73, 79, 83, 89, 97, 101, 103, 107, 109, 113,
  };

  PPERFECT_HASH_TABLE table =
      CreateTableFromKeys(keys, PerfectHashHashMulshrolate4RXFunctionId);
  ASSERT_NE(table, nullptr);

  auto *shim = reinterpret_cast<PerfectHashTableShim *>(table);
  ASSERT_NE(shim->Vtbl->SlowIndex, nullptr);

  PERFECT_HASH_TABLE_COMPILE_FLAGS compileFlags = {0};
  compileFlags.Jit = TRUE;
  compileFlags.JitBackendRawDog = TRUE;

  HRESULT result = online_->Vtbl->CompileTable(online_, table, &compileFlags);
  if (result == PH_E_NOT_IMPLEMENTED) {
    shim->Vtbl->Release(table);
    GTEST_SKIP() << "RawDog Mulshrolate4RX unavailable on this host.";
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
TEST_F(PerfectHashOnlineTests, GraphImpl4LlvmJitMatchesIndexAssigned32) {
  const auto keys = MakePseudoRandomKeys(64, 0xC001D00Du);

  ScopedPhTable table(CreateTableFromKeys(
      keys,
      PerfectHashHashMultiplyShiftRFunctionId,
      false,
      4,
      nullptr,
      5));
  ASSERT_NE(table.get(), nullptr);

  auto *shim = reinterpret_cast<PerfectHashTableShim *>(table.get());
  PERFECT_HASH_TABLE_FLAGS flags = GetTableFlags(table.get());
  ASSERT_EQ(flags.AssignedElementSizeInBits << 3, 32u);
  ASSERT_EQ(shim->Vtbl->SlowIndex, nullptr);

  const auto expected = CaptureIndexes(table.get(), keys);

  PERFECT_HASH_TABLE_COMPILE_FLAGS compileFlags = MakeLlvmJitCompileFlags();

  HRESULT result = online_->Vtbl->CompileTable(online_,
                                               table.get(),
                                               &compileFlags);
  if (result == PH_E_LLVM_BACKEND_NOT_FOUND) {
    GTEST_SKIP() << "LLVM GraphImpl4 scalar JIT backend not found on this host.";
  }
  ASSERT_EQ(result, S_OK);

  for (size_t i = 0; i < keys.size(); ++i) {
    ULONG index = 0;
    ASSERT_GE(shim->Vtbl->Index(table.get(), keys[i], &index), 0);
    EXPECT_EQ(expected[i], index);
  }
}

TEST_F(PerfectHashOnlineTests, GraphImpl4LlvmJitMatchesIndexSparse32) {
  const auto keys = MakeSparse32Keys(256);

  ScopedPhTable table(CreateTableFromKeys(
      keys,
      PerfectHashHashMultiplyShiftRFunctionId,
      false,
      4,
      nullptr,
      5));
  ASSERT_NE(table.get(), nullptr);

  auto *shim = reinterpret_cast<PerfectHashTableShim *>(table.get());
  PERFECT_HASH_TABLE_FLAGS flags = GetTableFlags(table.get());
  ASSERT_EQ(flags.AssignedElementSizeInBits << 3, 32u);
  ASSERT_EQ(shim->Vtbl->SlowIndex, nullptr);

  const auto expected = CaptureIndexes(table.get(), keys);

  PERFECT_HASH_TABLE_COMPILE_FLAGS compileFlags = MakeLlvmJitCompileFlags();

  HRESULT result = online_->Vtbl->CompileTable(online_,
                                               table.get(),
                                               &compileFlags);
  if (result == PH_E_LLVM_BACKEND_NOT_FOUND) {
    GTEST_SKIP() << "LLVM GraphImpl4 sparse32 scalar JIT backend not found "
                 << "on this host.";
  }
  ASSERT_EQ(result, S_OK);

  for (size_t i = 0; i < keys.size(); ++i) {
    ULONG index = 0;
    ASSERT_GE(shim->Vtbl->Index(table.get(), keys[i], &index), 0);
    EXPECT_EQ(expected[i], index);
  }
}

TEST_F(PerfectHashOnlineTests, GraphImpl4LlvmJitMulshrolate3RXMatchesIndexAssigned32) {
  const auto keys = MakePseudoRandomKeys(64, 0xB16B00B5u);

  ScopedPhTable table(CreateTableFromKeys(
      keys,
      PerfectHashHashMulshrolate3RXFunctionId,
      false,
      4,
      nullptr,
      5));
  ASSERT_NE(table.get(), nullptr);

  auto *shim = reinterpret_cast<PerfectHashTableShim *>(table.get());
  PERFECT_HASH_TABLE_FLAGS flags = GetTableFlags(table.get());
  ASSERT_EQ(flags.AssignedElementSizeInBits << 3, 32u);
  ASSERT_EQ(shim->Vtbl->SlowIndex, nullptr);

  const auto expected = CaptureIndexes(table.get(), keys);

  PERFECT_HASH_TABLE_COMPILE_FLAGS compileFlags = MakeLlvmJitCompileFlags();

  HRESULT result = online_->Vtbl->CompileTable(online_,
                                               table.get(),
                                               &compileFlags);
  if (result == PH_E_LLVM_BACKEND_NOT_FOUND) {
    GTEST_SKIP() << "LLVM GraphImpl4 Mulshrolate3RX JIT backend not found on this host.";
  }
  ASSERT_EQ(result, S_OK);

  for (size_t i = 0; i < keys.size(); ++i) {
    ULONG index = 0;
    ASSERT_GE(shim->Vtbl->Index(table.get(), keys[i], &index), 0);
    EXPECT_EQ(expected[i], index);
  }
}

TEST_F(PerfectHashOnlineTests, GraphImpl4LlvmIndex32x4MatchesIndexAssigned16) {
  const auto keys = MakeSparse32Keys(256);

  ScopedPhTable table(CreateTableFromKeys(
      keys,
      PerfectHashHashMultiplyShiftRFunctionId,
      true,
      4,
      nullptr,
      5));
  ASSERT_NE(table.get(), nullptr);

  auto *shim = reinterpret_cast<PerfectHashTableShim *>(table.get());
  PERFECT_HASH_TABLE_FLAGS flags = GetTableFlags(table.get());
  ASSERT_EQ(flags.AssignedElementSizeInBits << 3, 16u);
  ASSERT_EQ(shim->Vtbl->SlowIndex, nullptr);

  const auto expected = CaptureIndexes(table.get(), keys);

  PERFECT_HASH_TABLE_COMPILE_FLAGS compileFlags = MakeLlvmJitCompileFlags();
  compileFlags.JitVectorIndex32x4 = TRUE;

  HRESULT result = online_->Vtbl->CompileTable(online_,
                                               table.get(),
                                               &compileFlags);
  if (result == PH_E_LLVM_BACKEND_NOT_FOUND) {
    GTEST_SKIP() << "LLVM GraphImpl4 Index32x4 backend not found on this host.";
  }
  ASSERT_EQ(result, S_OK);

  PPERFECT_HASH_TABLE_JIT_INTERFACE raw_jit = nullptr;
  result = shim->Vtbl->QueryInterface(
      table.get(),
#ifdef PH_WINDOWS
      IID_PERFECT_HASH_TABLE_JIT_INTERFACE,
#else
      &IID_PERFECT_HASH_TABLE_JIT_INTERFACE,
#endif
      reinterpret_cast<void **>(&raw_jit));
  ScopedPhJit jit(raw_jit);
  ASSERT_GE(result, 0);
  ASSERT_NE(jit.get(), nullptr);

  PERFECT_HASH_TABLE_JIT_INFO info = {0};
  result = jit.get()->Vtbl->GetInfo(jit.get(), &info);
  ASSERT_GE(result, 0);
  EXPECT_TRUE(info.Flags.Index32x4Vector);

  for (size_t i = 0; i < keys.size(); i += 4) {
    ULONG index1 = 0;
    ULONG index2 = 0;
    ULONG index3 = 0;
    ULONG index4 = 0;

    result = jit.get()->Vtbl->Index32x4(jit.get(),
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
}

TEST_F(PerfectHashOnlineTests, GraphImpl4LlvmIndex64x4MatchesDownsizedIndex) {
  const std::vector<ULONGLONG> keys = {
      1ull, 3ull, 5ull, 7ull, 11ull, 13ull, 17ull, 19ull,
      23ull, 29ull, 31ull, 37ull, 41ull, 43ull, 47ull, 53ull,
      59ull, 61ull, 67ull, 71ull, 73ull, 79ull, 83ull, 89ull,
      97ull, 101ull, 103ull, 107ull, 109ull, 113ull, 127ull, 131ull,
  };

  ScopedPhTable table(CreateTableFromKeys64(keys,
                                            PerfectHashHashMultiplyShiftRFunctionId,
                                            false,
                                            4));
  ASSERT_NE(table.get(), nullptr);

  auto *shim = reinterpret_cast<PerfectHashTableShim *>(table.get());
  PERFECT_HASH_TABLE_FLAGS flags = GetTableFlags(table.get());
  ASSERT_EQ(flags.AssignedElementSizeInBits << 3, 32u);
  ASSERT_EQ(shim->Vtbl->SlowIndex, nullptr);

  const auto expected = CaptureIndexes64(table.get(), keys);

  PERFECT_HASH_TABLE_COMPILE_FLAGS compileFlags = MakeLlvmJitCompileFlags();
  compileFlags.JitIndex64 = TRUE;
  compileFlags.JitIndex32x4 = TRUE;

  HRESULT result = online_->Vtbl->CompileTable(online_,
                                               table.get(),
                                               &compileFlags);
  if (result == PH_E_LLVM_BACKEND_NOT_FOUND) {
    GTEST_SKIP() << "LLVM GraphImpl4 Index64 JIT backend not found on this host.";
  }
  ASSERT_EQ(result, S_OK);

  PPERFECT_HASH_TABLE_JIT_INTERFACE raw_jit = nullptr;
  result = shim->Vtbl->QueryInterface(
      table.get(),
#ifdef PH_WINDOWS
      IID_PERFECT_HASH_TABLE_JIT_INTERFACE,
#else
      &IID_PERFECT_HASH_TABLE_JIT_INTERFACE,
#endif
      reinterpret_cast<void **>(&raw_jit));
  ScopedPhJit jit(raw_jit);
  ASSERT_GE(result, 0);
  ASSERT_NE(jit.get(), nullptr);

  for (size_t i = 0; i < keys.size(); ++i) {
    ULONG index = 0;
    result = jit.get()->Vtbl->Index64(jit.get(), keys[i], &index);
    ASSERT_GE(result, 0);
    EXPECT_EQ(expected[i], index);
  }

  for (size_t i = 0; i < keys.size(); i += 4) {
    ULONG index1 = 0;
    ULONG index2 = 0;
    ULONG index3 = 0;
    ULONG index4 = 0;

    result = jit.get()->Vtbl->Index64x4(jit.get(),
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
}

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
