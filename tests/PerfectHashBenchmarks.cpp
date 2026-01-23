#include <chrono>
#include <cinttypes>
#include <cstdint>
#include <cstring>
#include <algorithm>
#include <array>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <random>
#include <string>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#ifndef PH_WINDOWS
#include <limits.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>
#if defined(__APPLE__)
#include <mach-o/dyld.h>
#endif
#endif

#include <PerfectHash.h>

#ifdef min
#undef min
#endif
#ifdef max
#undef max
#endif

namespace {

struct PerfectHashTableShim {
  PPERFECT_HASH_TABLE_VTBL Vtbl;
};

struct BenchmarkOptions {
  size_t key_count = 100000;
  size_t iterations = 5;
  size_t loops = 50;
  uint64_t seed = 0x4b1d5d73ULL;
  bool jit = true;
  enum class JitBackend {
    Auto,
    Llvm,
    RawDog,
  };
  JitBackend jit_backend = JitBackend::Auto;
  bool index32x2 = false;
  bool index32x4 = false;
  bool index32x8 = false;
  bool index32x16 = false;
  bool jit_vector_index32x2 = false;
  bool jit_vector_index32x4 = false;
  bool jit_vector_index32x8 = false;
  PERFECT_HASH_JIT_MAX_ISA_ID jit_max_isa = PerfectHashJitMaxIsaAuto;
  bool compare_isa = false;
  bool compare_backends = false;
  bool compare_backends_process = false;
  bool process_child = false;
  size_t process_iterations = 3;
  bool no_std_map_baselines = false;
  std::string keys_file;
  bool use_keys_file = false;
  std::array<ULONG, 8> seeds = {};
  size_t seed_count = 0;
  bool seed3_byte1_set = false;
  bool seed3_byte2_set = false;
  ULONG seed3_byte1 = 0;
  ULONG seed3_byte2 = 0;
  uint64_t fixed_attempts = 0;
  bool use_fixed_attempts = false;
  uint64_t max_solve_time = 0;
  bool use_max_solve_time = false;
  PERFECT_HASH_HASH_FUNCTION_ID hash_function_id =
      PerfectHashHashMultiplyShiftRFunctionId;
  std::string hash_function_name = "MultiplyShiftR";
  ULONG graph_impl = 3;
};

struct BenchmarkResult {
  size_t key_count = 0;
  double sort_ms = 0.0;
  double table_create_ms = 0.0;
  double jit_compile_ms = 0.0;
  double unordered_build_ms = 0.0;
  double ordered_build_ms = 0.0;
  double lookup_ph_ms = 0.0;
  double lookup_unordered_ms = 0.0;
  double lookup_ordered_ms = 0.0;
  bool jit = false;
  PERFECT_HASH_JIT_MAX_ISA_ID jit_max_isa = PerfectHashJitMaxIsaAuto;
  bool index32x2 = false;
  bool index32x4 = false;
  bool index32x8 = false;
  bool index32x16 = false;
  bool index32x16_available = true;
  std::string jit_target_cpu;
  std::string jit_target_features;
};

bool StartsWith(const char *value, const char *prefix) {
  return std::strncmp(value, prefix, std::strlen(prefix)) == 0;
}

bool ParseUnsigned(const char *value, uint64_t *out) {
  if (!value || !out) {
    return false;
  }

  char *end = nullptr;
  unsigned long long parsed = std::strtoull(value, &end, 10);
  if (end == value || (end && *end != '\0')) {
    return false;
  }

  *out = static_cast<uint64_t>(parsed);
  return true;
}

bool ParseUlongAuto(const char *value, ULONG *out) {
  if (!value || !out) {
    return false;
  }

  char *end = nullptr;
  unsigned long long parsed = std::strtoull(value, &end, 0);
  if (end == value || (end && *end != '\0') || parsed > 0xffffffffULL) {
    return false;
  }

  *out = static_cast<ULONG>(parsed);
  return true;
}

std::string ToLower(std::string_view value) {
  std::string lowered(value);
  for (char &ch : lowered) {
    if (ch >= 'A' && ch <= 'Z') {
      ch = static_cast<char>(ch - 'A' + 'a');
    }
  }
  return lowered;
}

bool ParseHashFunction(std::string_view value,
                       PERFECT_HASH_HASH_FUNCTION_ID *id,
                       std::string *name) {
  if (!id || !name) {
    return false;
  }

  std::string lowered = ToLower(value);

  if (lowered == "multiplyshiftr" || lowered == "msr") {
    *id = PerfectHashHashMultiplyShiftRFunctionId;
    *name = "MultiplyShiftR";
    return true;
  }

  if (lowered == "multiplyshiftrx" || lowered == "msrx") {
    *id = PerfectHashHashMultiplyShiftRXFunctionId;
    *name = "MultiplyShiftRX";
    return true;
  }

  if (lowered == "multiplyshiftr2" || lowered == "msr2") {
    *id = PerfectHashHashMultiplyShiftR2FunctionId;
    *name = "MultiplyShiftR2";
    return true;
  }

  if (lowered == "multiplyshiftlr" || lowered == "mslr") {
    *id = PerfectHashHashMultiplyShiftLRFunctionId;
    *name = "MultiplyShiftLR";
    return true;
  }

  if (lowered == "mulshrolate1rx" || lowered == "msrol1rx") {
    *id = PerfectHashHashMulshrolate1RXFunctionId;
    *name = "Mulshrolate1RX";
    return true;
  }

  if (lowered == "mulshrolate2rx" || lowered == "msrol2rx") {
    *id = PerfectHashHashMulshrolate2RXFunctionId;
    *name = "Mulshrolate2RX";
    return true;
  }

  if (lowered == "mulshrolate3rx" || lowered == "msrol3rx") {
    *id = PerfectHashHashMulshrolate3RXFunctionId;
    *name = "Mulshrolate3RX";
    return true;
  }

  if (lowered == "mulshrolate4rx" || lowered == "msrol4rx") {
    *id = PerfectHashHashMulshrolate4RXFunctionId;
    *name = "Mulshrolate4RX";
    return true;
  }

  return false;
}

bool ParseArguments(int argc, char **argv, BenchmarkOptions *options) {
  if (!options) {
    return false;
  }

  for (int index = 1; index < argc; ++index) {
    const char *arg = argv[index];
    if (!arg) {
      continue;
    }

    if (std::strcmp(arg, "--help") == 0) {
      return false;
    }

    if (StartsWith(arg, "--keys=")) {
      uint64_t value = 0;
      if (!ParseUnsigned(arg + 7, &value)) {
        return false;
      }
      options->key_count = static_cast<size_t>(value);
      continue;
    }

    if (StartsWith(arg, "--iterations=")) {
      uint64_t value = 0;
      if (!ParseUnsigned(arg + 13, &value)) {
        return false;
      }
      options->iterations = static_cast<size_t>(value);
      continue;
    }

    if (StartsWith(arg, "--loops=")) {
      uint64_t value = 0;
      if (!ParseUnsigned(arg + 8, &value)) {
        return false;
      }
      options->loops = static_cast<size_t>(value);
      continue;
    }

    if (StartsWith(arg, "--seed=")) {
      uint64_t value = 0;
      if (!ParseUnsigned(arg + 7, &value)) {
        return false;
      }
      options->seed = value;
      continue;
    }

    if (StartsWith(arg, "--keys-file=")) {
      options->keys_file = arg + 12;
      options->use_keys_file = true;
      continue;
    }

    if (StartsWith(arg, "--seed1=")) {
      ULONG value = 0;
      if (!ParseUlongAuto(arg + 8, &value)) {
        return false;
      }
      options->seeds[0] = value;
      options->seed_count = std::max(options->seed_count, size_t{1});
      continue;
    }

    if (StartsWith(arg, "--seed2=")) {
      ULONG value = 0;
      if (!ParseUlongAuto(arg + 8, &value)) {
        return false;
      }
      options->seeds[1] = value;
      options->seed_count = std::max(options->seed_count, size_t{2});
      continue;
    }

    if (StartsWith(arg, "--seed3=")) {
      ULONG value = 0;
      if (!ParseUlongAuto(arg + 8, &value)) {
        return false;
      }
      options->seeds[2] = value;
      options->seed_count = std::max(options->seed_count, size_t{3});
      continue;
    }

    if (StartsWith(arg, "--seed3-byte1=")) {
      ULONG value = 0;
      if (!ParseUlongAuto(arg + 14, &value) || value > 0xff) {
        return false;
      }
      options->seed3_byte1 = value;
      options->seed3_byte1_set = true;
      continue;
    }

    if (StartsWith(arg, "--seed3-byte2=")) {
      ULONG value = 0;
      if (!ParseUlongAuto(arg + 14, &value) || value > 0xff) {
        return false;
      }
      options->seed3_byte2 = value;
      options->seed3_byte2_set = true;
      continue;
    }

    if (StartsWith(arg, "--hash=")) {
      std::string_view value = arg + 7;
      if (!ParseHashFunction(value,
                             &options->hash_function_id,
                             &options->hash_function_name)) {
        return false;
      }
      continue;
    }

    if (StartsWith(arg, "--preset=")) {
      std::string_view value = arg + 9;
      std::string preset = ToLower(value);
      if (preset == "hologramworld-msr" ||
          preset == "hologramworld-31016-msr") {
        options->keys_file = "keys/HologramWorld-31016.keys";
        options->use_keys_file = true;
        options->hash_function_id = PerfectHashHashMultiplyShiftRFunctionId;
        options->hash_function_name = "MultiplyShiftR";
        options->seeds[0] = 0xD5557D96u;
        options->seeds[1] = 0x37BC1058u;
        options->seeds[2] = 0x1010u;
        options->seed_count = 3;
        options->seed3_byte1_set = false;
        options->seed3_byte2_set = false;
        continue;
      }
      return false;
    }

    if (StartsWith(arg, "--graph-impl=")) {
      uint64_t value = 0;
      if (!ParseUnsigned(arg + 13, &value) || value > 0xffffffffULL) {
        return false;
      }
      options->graph_impl = static_cast<ULONG>(value);
      continue;
    }

    if (StartsWith(arg, "--fixed-attempts=")) {
      uint64_t value = 0;
      if (!ParseUnsigned(arg + 17, &value)) {
        return false;
      }
      options->fixed_attempts = value;
      options->use_fixed_attempts = true;
      continue;
    }

    if (StartsWith(arg, "--max-solve-time=")) {
      uint64_t value = 0;
      if (!ParseUnsigned(arg + 17, &value)) {
        return false;
      }
      options->max_solve_time = value;
      options->use_max_solve_time = true;
      continue;
    }

    if (std::strcmp(arg, "--compare-isa") == 0) {
      options->compare_isa = true;
      continue;
    }

    if (std::strcmp(arg, "--compare-backends") == 0) {
      options->compare_backends = true;
      continue;
    }

    if (std::strcmp(arg, "--compare-backends-process") == 0) {
      options->compare_backends_process = true;
      continue;
    }

    if (std::strcmp(arg, "--process-child") == 0) {
      options->process_child = true;
      continue;
    }

    if (StartsWith(arg, "--process-iterations=")) {
      uint64_t value = 0;
      if (!ParseUnsigned(arg + 21, &value)) {
        return false;
      }
      options->process_iterations = static_cast<size_t>(value);
      continue;
    }

    if (std::strcmp(arg, "--no-std-map-baselines") == 0) {
      options->no_std_map_baselines = true;
      continue;
    }

    if (std::strcmp(arg, "--no-jit") == 0) {
      options->jit = false;
      continue;
    }

    if (StartsWith(arg, "--jit-backend=")) {
      const char *value = arg + 14;
      if (std::strcmp(value, "auto") == 0 ||
          std::strcmp(value, "Auto") == 0) {
        options->jit_backend = BenchmarkOptions::JitBackend::Auto;
        continue;
      }
      if (std::strcmp(value, "llvm") == 0 ||
          std::strcmp(value, "Llvm") == 0 ||
          std::strcmp(value, "LLVM") == 0) {
        options->jit_backend = BenchmarkOptions::JitBackend::Llvm;
        continue;
      }
      if (std::strcmp(value, "rawdog") == 0 ||
          std::strcmp(value, "RawDog") == 0 ||
          std::strcmp(value, "RAWDOG") == 0) {
        options->jit_backend = BenchmarkOptions::JitBackend::RawDog;
        continue;
      }
      return false;
    }

    if (StartsWith(arg, "--jit-max-isa=")) {
      const char *value = arg + 14;
      if (std::strcmp(value, "auto") == 0 ||
          std::strcmp(value, "Auto") == 0) {
        options->jit_max_isa = PerfectHashJitMaxIsaAuto;
        continue;
      }
      if (std::strcmp(value, "avx") == 0 ||
          std::strcmp(value, "Avx") == 0) {
        options->jit_max_isa = PerfectHashJitMaxIsaAvx;
        continue;
      }
      if (std::strcmp(value, "avx2") == 0 ||
          std::strcmp(value, "Avx2") == 0) {
        options->jit_max_isa = PerfectHashJitMaxIsaAvx2;
        continue;
      }
      if (std::strcmp(value, "avx512") == 0 ||
          std::strcmp(value, "Avx512") == 0) {
        options->jit_max_isa = PerfectHashJitMaxIsaAvx512;
        continue;
      }
      if (std::strcmp(value, "neon") == 0 ||
          std::strcmp(value, "Neon") == 0) {
        options->jit_max_isa = PerfectHashJitMaxIsaNeon;
        continue;
      }
      if (std::strcmp(value, "sve") == 0 ||
          std::strcmp(value, "Sve") == 0) {
        options->jit_max_isa = PerfectHashJitMaxIsaSve;
        continue;
      }
      if (std::strcmp(value, "sve2") == 0 ||
          std::strcmp(value, "Sve2") == 0) {
        options->jit_max_isa = PerfectHashJitMaxIsaSve2;
        continue;
      }
      return false;
    }

    if (std::strcmp(arg, "--index2") == 0 ||
        std::strcmp(arg, "--index32x2") == 0) {
      options->index32x2 = true;
      continue;
    }

    if (std::strcmp(arg, "--index4") == 0 ||
        std::strcmp(arg, "--index32x4") == 0) {
      options->index32x4 = true;
      continue;
    }

    if (std::strcmp(arg, "--index8") == 0 ||
        std::strcmp(arg, "--index32x8") == 0) {
      options->index32x8 = true;
      continue;
    }

    if (std::strcmp(arg, "--index16") == 0 ||
        std::strcmp(arg, "--index32x16") == 0) {
      options->index32x16 = true;
      continue;
    }

    if (std::strcmp(arg, "--jit-vector-index2") == 0 ||
        std::strcmp(arg, "--jit-vector-index32x2") == 0) {
      options->jit_vector_index32x2 = true;
      options->index32x2 = true;
      continue;
    }

    if (std::strcmp(arg, "--jit-vector-index4") == 0 ||
        std::strcmp(arg, "--jit-vector-index32x4") == 0) {
      options->jit_vector_index32x4 = true;
      options->index32x4 = true;
      continue;
    }

    if (std::strcmp(arg, "--jit-vector-index8") == 0 ||
        std::strcmp(arg, "--jit-vector-index32x8") == 0) {
      options->jit_vector_index32x8 = true;
      options->index32x8 = true;
      continue;
    }

    return false;
  }

  return true;
}

void ApplySeed3Bytes(BenchmarkOptions *options) {
  if (!options) {
    return;
  }

  if ((options->seed3_byte1_set || options->seed3_byte2_set) &&
      options->seed_count < 3) {
    ULONG seed3 = (options->seed3_byte1 & 0xffu) |
                  ((options->seed3_byte2 & 0xffu) << 8);
    options->seeds[2] = seed3;
    options->seed_count = std::max(options->seed_count, size_t{3});
  }
}

void NormalizeVectorOptions(BenchmarkOptions *options) {
  if (!options) {
    return;
  }

  if (options->index32x16) {
    options->index32x2 = false;
    options->index32x4 = false;
    options->index32x8 = false;
  }
  if (options->index32x8) {
    options->index32x2 = false;
    options->index32x4 = false;
  }
  if (options->index32x4) {
    options->index32x2 = false;
  }
}

void PrintUsage(const char *exe) {
  std::cout << "Usage: " << (exe ? exe : "perfecthash_benchmarks")
            << " [--keys=N] [--iterations=N] [--loops=N] [--seed=N]"
            << " [--keys-file=PATH]"
            << " [--seed1=VALUE] [--seed2=VALUE] [--seed3=VALUE]"
            << " [--seed3-byte1=VALUE] [--seed3-byte2=VALUE]"
            << " [--hash=MultiplyShiftR|MultiplyShiftRX|MultiplyShiftR2"
            << "|MultiplyShiftLR|Mulshrolate1RX|Mulshrolate2RX"
            << "|Mulshrolate3RX|Mulshrolate4RX]"
            << " [--preset=hologramworld-msr]"
            << " [--graph-impl=N]"
            << " [--fixed-attempts=N] [--max-solve-time=N]"
            << " [--compare-isa] [--compare-backends]"
            << " [--compare-backends-process] [--process-iterations=N]"
            << " [--no-jit] [--jit-backend=auto|llvm|rawdog]"
            << " [--jit-max-isa=auto|avx|avx2|avx512|neon|sve|sve2]"
            << " [--no-std-map-baselines]"
            << " [--index32x2] [--index32x4] [--index32x8] [--index32x16]"
            << " [--jit-vector-index32x2] [--jit-vector-index32x4]"
            << " [--jit-vector-index32x8]\n";
}

const char *JitMaxIsaToString(PERFECT_HASH_JIT_MAX_ISA_ID value) {
  switch (value) {
    case PerfectHashJitMaxIsaAuto:
      return "auto";
    case PerfectHashJitMaxIsaAvx:
      return "avx";
    case PerfectHashJitMaxIsaAvx2:
      return "avx2";
    case PerfectHashJitMaxIsaAvx512:
      return "avx512";
    case PerfectHashJitMaxIsaNeon:
      return "neon";
    case PerfectHashJitMaxIsaSve:
      return "sve";
    case PerfectHashJitMaxIsaSve2:
      return "sve2";
    default:
      return "unknown";
  }
}

const char *JitBackendToString(BenchmarkOptions::JitBackend backend) {
  switch (backend) {
    case BenchmarkOptions::JitBackend::Auto:
      return "auto";
    case BenchmarkOptions::JitBackend::Llvm:
      return "llvm";
    case BenchmarkOptions::JitBackend::RawDog:
      return "rawdog";
  }
  return "unknown";
}

std::vector<ULONG> BuildKeys(size_t count, uint64_t seed) {
  std::vector<ULONG> keys;
  keys.reserve(count);

  std::unordered_set<ULONG> seen;
  seen.reserve(count * 2);

  std::mt19937_64 rng(seed);
  while (keys.size() < count) {
    ULONG value = static_cast<ULONG>(rng());
    if (value == 0) {
      continue;
    }

    if (seen.insert(value).second) {
      keys.push_back(value);
    }
  }

  return keys;
}

template <typename Fn>
double MeasureMillis(Fn &&fn) {
  auto start = std::chrono::steady_clock::now();
  fn();
  auto end = std::chrono::steady_clock::now();
  std::chrono::duration<double, std::milli> elapsed = end - start;
  return elapsed.count();
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

std::string GetExecutablePath(const char *argv0) {
#ifndef PH_WINDOWS
#if defined(__APPLE__)
  uint32_t size = 0;
  if (_NSGetExecutablePath(nullptr, &size) != 0 && size > 0) {
    std::string path(size, '\0');
    if (_NSGetExecutablePath(path.data(), &size) == 0) {
      path.resize(std::strlen(path.c_str()));
      return path;
    }
  }
#else
  char pathBuffer[PATH_MAX] = {};
  ssize_t length = readlink("/proc/self/exe",
                            pathBuffer,
                            static_cast<size_t>(PATH_MAX) - 1);
  if (length > 0) {
    pathBuffer[length] = '\0';
    return std::string(pathBuffer);
  }
#endif
#endif
  return argv0 ? std::string(argv0) : std::string("perfecthash_benchmarks");
}

void AppendUnsignedArg(std::vector<std::string> *args,
                       const char *name,
                       uint64_t value) {
  if (!args || !name) {
    return;
  }
  args->emplace_back(std::string(name) + "=" + std::to_string(value));
}

void AppendUlongArg(std::vector<std::string> *args,
                    const char *name,
                    ULONG value) {
  AppendUnsignedArg(args, name, static_cast<uint64_t>(value));
}

std::vector<std::string> BuildChildArgs(const BenchmarkOptions &options,
                                        const std::string &exe) {
  std::vector<std::string> args;
  args.reserve(32);
  args.push_back(exe);
  args.push_back("--process-child");

  AppendUnsignedArg(&args, "--keys", options.key_count);
  AppendUnsignedArg(&args, "--iterations", options.iterations);
  AppendUnsignedArg(&args, "--loops", options.loops);
  AppendUnsignedArg(&args, "--seed", options.seed);

  if (options.use_keys_file) {
    args.emplace_back(std::string("--keys-file=") + options.keys_file);
  }

  if (!options.hash_function_name.empty()) {
    args.emplace_back(std::string("--hash=") + options.hash_function_name);
  }

  AppendUnsignedArg(&args, "--graph-impl", options.graph_impl);

  if (options.seed_count > 0) {
    AppendUlongArg(&args, "--seed1", options.seeds[0]);
  }
  if (options.seed_count > 1) {
    AppendUlongArg(&args, "--seed2", options.seeds[1]);
  }
  if (options.seed_count > 2) {
    AppendUlongArg(&args, "--seed3", options.seeds[2]);
  }
  if (options.seed3_byte1_set) {
    AppendUlongArg(&args, "--seed3-byte1", options.seed3_byte1);
  }
  if (options.seed3_byte2_set) {
    AppendUlongArg(&args, "--seed3-byte2", options.seed3_byte2);
  }

  if (options.use_fixed_attempts) {
    AppendUnsignedArg(&args, "--fixed-attempts", options.fixed_attempts);
  }
  if (options.use_max_solve_time) {
    AppendUnsignedArg(&args, "--max-solve-time", options.max_solve_time);
  }

  args.emplace_back(std::string("--jit-max-isa=") +
                    JitMaxIsaToString(options.jit_max_isa));

  if (options.index32x2) {
    args.emplace_back("--index32x2");
  }
  if (options.index32x4) {
    args.emplace_back("--index32x4");
  }
  if (options.index32x8) {
    args.emplace_back("--index32x8");
  }
  if (options.index32x16) {
    args.emplace_back("--index32x16");
  }
  if (options.jit_vector_index32x2) {
    args.emplace_back("--jit-vector-index32x2");
  }
  if (options.jit_vector_index32x4) {
    args.emplace_back("--jit-vector-index32x4");
  }
  if (options.jit_vector_index32x8) {
    args.emplace_back("--jit-vector-index32x8");
  }

  if (options.no_std_map_baselines) {
    args.emplace_back("--no-std-map-baselines");
  }

  if (!options.jit) {
    args.emplace_back("--no-jit");
  }

  return args;
}

int RunChildProcess(const std::vector<std::string> &args,
                    double *elapsed_ms) {
#ifdef PH_WINDOWS
  (void)args;
  (void)elapsed_ms;
  std::cerr << "Process-based benchmarks are not supported on Windows.\n";
  return 1;
#else
  if (args.empty()) {
    std::cerr << "Process benchmark args empty.\n";
    return 1;
  }

  std::vector<char *> argv;
  argv.reserve(args.size() + 1);
  for (const auto &arg : args) {
    argv.push_back(const_cast<char *>(arg.c_str()));
  }
  argv.push_back(nullptr);

  auto start = std::chrono::steady_clock::now();
  pid_t pid = fork();
  if (pid == 0) {
    execv(argv[0], argv.data());
    execvp(argv[0], argv.data());
    _exit(127);
  }
  if (pid < 0) {
    std::cerr << "Failed to fork benchmark process.\n";
    return 1;
  }

  int status = 0;
  if (waitpid(pid, &status, 0) < 0) {
    std::cerr << "Failed to wait for benchmark process.\n";
    return 1;
  }
  auto end = std::chrono::steady_clock::now();

  if (elapsed_ms) {
    std::chrono::duration<double, std::milli> elapsed = end - start;
    *elapsed_ms = elapsed.count();
  }

  if (!WIFEXITED(status)) {
    std::cerr << "Benchmark process terminated unexpectedly.\n";
    return 1;
  }

  int exit_code = WEXITSTATUS(status);
  if (exit_code != 0) {
    std::cerr << "Benchmark process exited with code " << exit_code << ".\n";
    return exit_code;
  }

  return 0;
#endif
}

int RunBenchmark(const BenchmarkOptions &base,
                 BenchmarkResult *result,
                 bool verbose) {
  BenchmarkOptions options = base;
  ApplySeed3Bytes(&options);
  NormalizeVectorOptions(&options);

#ifndef PH_HAS_LLVM
  if (options.jit &&
      options.jit_backend != BenchmarkOptions::JitBackend::RawDog &&
      verbose) {
    std::cout << "LLVM support disabled; JIT benchmarks skipped.\n";
  }
  if (options.jit_backend != BenchmarkOptions::JitBackend::RawDog) {
    options.jit = false;
  }
#endif
#ifndef PH_HAS_RAWDOG_JIT
  if (options.jit &&
      options.jit_backend == BenchmarkOptions::JitBackend::RawDog &&
      verbose) {
    std::cout << "RawDog support disabled; JIT benchmarks skipped.\n";
  }
  if (options.jit_backend == BenchmarkOptions::JitBackend::RawDog) {
    options.jit = false;
  }
#endif

  if (options.jit_backend == BenchmarkOptions::JitBackend::RawDog &&
      (options.index32x2 || options.index32x4 ||
       options.index32x8 || options.index32x16 ||
       options.jit_vector_index32x2 ||
       options.jit_vector_index32x4 ||
       options.jit_vector_index32x8)) {
    std::cerr << "RawDog JIT only supports scalar Index32().\n";
    return 1;
  }

  std::vector<ULONG> keys;
  if (options.use_keys_file) {
    if (!LoadKeysFromFile(options.keys_file, &keys)) {
      std::cerr << "Failed to load keys file: " << options.keys_file << "\n";
      return 1;
    }
  } else {
    keys = BuildKeys(options.key_count, options.seed);
  }
  auto sorted_keys = keys;
  double sortMs = MeasureMillis([&]() {
    std::sort(sorted_keys.begin(), sorted_keys.end());
  });

  if (verbose) {
    std::cout << "PerfectHash benchmarks\n";
    std::cout << "  Keys:       " << keys.size() << "\n";
    if (options.use_keys_file) {
      std::cout << "  Keys file:  " << options.keys_file << "\n";
    }
    std::cout << "  Hash:       " << options.hash_function_name << "\n";
    std::cout << "  Graph Impl: " << options.graph_impl << "\n";
    std::cout << "  Iterations: " << options.iterations << "\n";
    std::cout << "  Loops:      " << options.loops << "\n";
    std::cout << "  JIT:        " << (options.jit ? "on" : "off") << "\n";
    std::cout << "  JIT Backend: " << JitBackendToString(options.jit_backend)
              << "\n";
    std::cout << "  JIT Max ISA: " << JitMaxIsaToString(options.jit_max_isa)
              << "\n";
    if (options.seed_count > 0) {
      std::cout << "  Seeds:      ";
      for (size_t i = 0; i < options.seed_count; ++i) {
        std::cout << "0x" << std::hex << options.seeds[i] << std::dec;
        if (i + 1 < options.seed_count) {
          std::cout << ",";
        }
      }
      std::cout << "\n";
    }
    if (options.use_fixed_attempts) {
      std::cout << "  Fixed Attempts: " << options.fixed_attempts << "\n";
    }
    if (options.use_max_solve_time) {
      std::cout << "  Max Solve Time: " << options.max_solve_time << "s\n";
    }
    std::cout << "  Index32x2:  " << (options.index32x2 ? "on" : "off")
              << "\n";
    std::cout << "  Index32x4:  " << (options.index32x4 ? "on" : "off")
              << "\n";
    std::cout << "  Index32x8:  " << (options.index32x8 ? "on" : "off")
              << "\n";
    std::cout << "  Index32x16: " << (options.index32x16 ? "on" : "off")
              << "\n";
    std::cout << "  JIT Vec32x2: "
              << (options.jit_vector_index32x2 ? "on" : "off") << "\n";
    std::cout << "  JIT Vec32x4: "
              << (options.jit_vector_index32x4 ? "on" : "off") << "\n";
    std::cout << "  JIT Vec32x8: "
              << (options.jit_vector_index32x8 ? "on" : "off") << "\n";
  }

  PICLASSFACTORY classFactory = nullptr;
  PPERFECT_HASH_ONLINE online = nullptr;
  PPERFECT_HASH_PRINT_ERROR printError = nullptr;
  PPERFECT_HASH_PRINT_MESSAGE printMessage = nullptr;
  HMODULE module = nullptr;

  HRESULT resultValue = PerfectHashBootstrap(
      &classFactory, &printError, &printMessage, &module);
  if (resultValue < 0) {
    std::cerr << "PerfectHashBootstrap failed: 0x" << std::hex
              << static_cast<unsigned>(resultValue) << std::dec << "\n";
    return 1;
  }

  resultValue = classFactory->Vtbl->CreateInstance(
      classFactory,
      nullptr,
#ifdef PH_WINDOWS
      IID_PERFECT_HASH_ONLINE,
#else
      &IID_PERFECT_HASH_ONLINE,
#endif
      reinterpret_cast<void **>(&online));

  if (resultValue < 0) {
    std::cerr << "CreateInstance(Online) failed: 0x" << std::hex
              << static_cast<unsigned>(resultValue) << std::dec << "\n";
    classFactory->Vtbl->Release(classFactory);
#ifdef PH_WINDOWS
    if (module) {
      FreeLibrary(module);
    }
#endif
    return 1;
  }

  PERFECT_HASH_TABLE_CREATE_FLAGS tableFlags = {0};
  tableFlags.NoFileIo = TRUE;
  tableFlags.Quiet = TRUE;
  tableFlags.DoNotTryUseHash16Impl = TRUE;

  PPERFECT_HASH_TABLE table = nullptr;
  std::vector<PERFECT_HASH_TABLE_CREATE_PARAMETER> table_params;
  PERFECT_HASH_TABLE_CREATE_PARAMETERS tableCreateParams = {};
  PPERFECT_HASH_TABLE_CREATE_PARAMETERS tableCreateParamsPtr = nullptr;

  {
    PERFECT_HASH_TABLE_CREATE_PARAMETER param = {};
    param.Id = TableCreateParameterGraphImplId;
    param.AsULong = options.graph_impl;
    table_params.push_back(param);
  }

  if (options.seed_count > 0) {
    PERFECT_HASH_TABLE_CREATE_PARAMETER param = {};
    param.Id = TableCreateParameterSeedsId;
    param.AsValueArray.Values = options.seeds.data();
    param.AsValueArray.NumberOfValues =
        static_cast<ULONG>(options.seed_count);
    param.AsValueArray.ValueSizeInBytes = sizeof(ULONG);
    table_params.push_back(param);
  }

  if (options.use_fixed_attempts) {
    PERFECT_HASH_TABLE_CREATE_PARAMETER param = {};
    param.Id = TableCreateParameterFixedAttemptsId;
    param.AsULong = static_cast<ULONG>(options.fixed_attempts);
    table_params.push_back(param);
  }

  if (options.use_max_solve_time) {
    PERFECT_HASH_TABLE_CREATE_PARAMETER param = {};
    param.Id = TableCreateParameterMaxSolveTimeInSecondsId;
    param.AsULong = static_cast<ULONG>(options.max_solve_time);
    table_params.push_back(param);
  }

  if (!table_params.empty()) {
    tableCreateParams.SizeOfStruct = sizeof(tableCreateParams);
    tableCreateParams.NumberOfElements =
        static_cast<ULONG>(table_params.size());
    tableCreateParams.Params = table_params.data();
    tableCreateParamsPtr = &tableCreateParams;
  }

  double tableCreateMs = MeasureMillis([&]() {
    resultValue = online->Vtbl->CreateTableFromKeys(
        online,
        PerfectHashChm01AlgorithmId,
        options.hash_function_id,
        PerfectHashAndMaskFunctionId,
        sizeof(ULONG),
        static_cast<ULONGLONG>(sorted_keys.size()),
        const_cast<ULONG *>(sorted_keys.data()),
        nullptr,
        &tableFlags,
        tableCreateParamsPtr,
        &table);
  });

  if (resultValue < 0 || !table) {
    std::cerr << "CreateTableFromKeys failed: 0x" << std::hex
              << static_cast<unsigned>(resultValue) << std::dec << "\n";
    online->Vtbl->Release(online);
    classFactory->Vtbl->Release(classFactory);
#ifdef PH_WINDOWS
    if (module) {
      FreeLibrary(module);
    }
#endif
    return 1;
  }

  auto *shim = reinterpret_cast<PerfectHashTableShim *>(table);

  PERFECT_HASH_TABLE_COMPILE_FLAGS compileFlags = {0};
  if (options.index32x2) {
    compileFlags.JitIndex32x2 = TRUE;
  }
  if (options.index32x4) {
    compileFlags.JitIndex32x4 = TRUE;
  }
  if (options.index32x8) {
    compileFlags.JitIndex32x8 = TRUE;
  }
  if (options.index32x16) {
    compileFlags.JitIndex32x16 = TRUE;
  }
  if (options.jit_vector_index32x2) {
    compileFlags.JitVectorIndex32x2 = TRUE;
  }
  if (options.jit_vector_index32x4) {
    compileFlags.JitVectorIndex32x4 = TRUE;
  }
  if (options.jit_vector_index32x8) {
    compileFlags.JitVectorIndex32x8 = TRUE;
  }
  switch (options.jit_backend) {
    case BenchmarkOptions::JitBackend::Llvm:
      compileFlags.JitBackendLlvm = TRUE;
      break;
    case BenchmarkOptions::JitBackend::RawDog:
      compileFlags.JitBackendRawDog = TRUE;
      break;
    case BenchmarkOptions::JitBackend::Auto:
      break;
  }
  compileFlags.JitMaxIsa = options.jit_max_isa;

  double jitCompileMs = 0.0;
  if (options.jit) {
    jitCompileMs = MeasureMillis([&]() {
      resultValue = online->Vtbl->CompileTable(online, table, &compileFlags);
    });

    if (resultValue < 0) {
      std::cerr << "CompileTable failed: 0x" << std::hex
                << static_cast<unsigned>(resultValue) << std::dec << "\n";
      shim->Vtbl->Release(table);
      online->Vtbl->Release(online);
      classFactory->Vtbl->Release(classFactory);
#ifdef PH_WINDOWS
      if (module) {
        FreeLibrary(module);
      }
#endif
      return 1;
    }
  }

  PPERFECT_HASH_TABLE_JIT_INTERFACE jitInterface = nullptr;
  if (options.jit) {
    resultValue = shim->Vtbl->QueryInterface(
        table,
#ifdef PH_WINDOWS
        IID_PERFECT_HASH_TABLE_JIT_INTERFACE,
#else
        &IID_PERFECT_HASH_TABLE_JIT_INTERFACE,
#endif
        reinterpret_cast<void **>(&jitInterface));
    if (resultValue < 0 || !jitInterface) {
      jitInterface = nullptr;
    }
  }

  bool index32x16_available = true;
  if (jitInterface) {
    PERFECT_HASH_TABLE_JIT_INFO info = {0};
    resultValue = jitInterface->Vtbl->GetInfo(jitInterface, &info);
    if (resultValue >= 0) {
      const char *cpu = info.TargetCpu[0] ? info.TargetCpu : "<default>";
      const char *features = info.TargetFeatures[0]
                                 ? info.TargetFeatures
                                 : "<default>";
      if (verbose) {
        std::cout << "  JIT Target CPU: " << cpu << "\n";
        std::cout << "  JIT Target Features: " << features << "\n";
      }
      if (result) {
        result->jit_target_cpu = cpu;
        result->jit_target_features = features;
      }
      if (options.index32x16 && !info.Flags.Index32x16Compiled) {
        index32x16_available = false;
        if (verbose) {
          std::cout << "  JIT Index32x16 unavailable; falling back to scalar.\n";
        }
        options.index32x16 = false;
      }
    }
  }

  std::unordered_map<ULONG, ULONG> unordered;
  double unorderedBuildMs = 0.0;
  std::map<ULONG, ULONG> ordered;
  double orderedBuildMs = 0.0;

  if (!options.no_std_map_baselines) {
    unorderedBuildMs = MeasureMillis([&]() {
      unordered.reserve(keys.size());
      for (size_t i = 0; i < keys.size(); ++i) {
        unordered.emplace(keys[i], static_cast<ULONG>(i));
      }
    });

    orderedBuildMs = MeasureMillis([&]() {
      for (size_t i = 0; i < keys.size(); ++i) {
        ordered.emplace(keys[i], static_cast<ULONG>(i));
      }
    });
  }

  volatile ULONG sink = 0;

  auto perfectHashLookup = [&]() {
    if (options.jit && jitInterface &&
        (options.index32x2 || options.index32x4 ||
         options.index32x8 || options.index32x16)) {
      if (options.index32x16) {
        for (size_t loop = 0; loop < options.loops; ++loop) {
          size_t limit = keys.size() - (keys.size() % 16);
          for (size_t i = 0; i < limit; i += 16) {
            ULONG index1 = 0;
            ULONG index2 = 0;
            ULONG index3 = 0;
            ULONG index4 = 0;
            ULONG index5 = 0;
            ULONG index6 = 0;
            ULONG index7 = 0;
            ULONG index8 = 0;
            ULONG index9 = 0;
            ULONG index10 = 0;
            ULONG index11 = 0;
            ULONG index12 = 0;
            ULONG index13 = 0;
            ULONG index14 = 0;
            ULONG index15 = 0;
            ULONG index16 = 0;
            jitInterface->Vtbl->Index32x16(jitInterface,
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
                                           &index1,
                                           &index2,
                                           &index3,
                                           &index4,
                                           &index5,
                                           &index6,
                                           &index7,
                                           &index8,
                                           &index9,
                                           &index10,
                                           &index11,
                                           &index12,
                                           &index13,
                                           &index14,
                                           &index15,
                                           &index16);
            sink ^= index1 ^ index2 ^ index3 ^ index4 ^
                    index5 ^ index6 ^ index7 ^ index8 ^
                    index9 ^ index10 ^ index11 ^ index12 ^
                    index13 ^ index14 ^ index15 ^ index16;
          }
          for (size_t i = limit; i < keys.size(); ++i) {
            ULONG index = 0;
            jitInterface->Vtbl->Index32(jitInterface, keys[i], &index);
            sink ^= index;
          }
        }
      } else if (options.index32x8) {
        for (size_t loop = 0; loop < options.loops; ++loop) {
          size_t limit = keys.size() - (keys.size() % 8);
          for (size_t i = 0; i < limit; i += 8) {
            ULONG index1 = 0;
            ULONG index2 = 0;
            ULONG index3 = 0;
            ULONG index4 = 0;
            ULONG index5 = 0;
            ULONG index6 = 0;
            ULONG index7 = 0;
            ULONG index8 = 0;
            jitInterface->Vtbl->Index32x8(jitInterface,
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
            sink ^= index1 ^ index2 ^ index3 ^ index4 ^
                    index5 ^ index6 ^ index7 ^ index8;
          }
          for (size_t i = limit; i < keys.size(); ++i) {
            ULONG index = 0;
            jitInterface->Vtbl->Index32(jitInterface, keys[i], &index);
            sink ^= index;
          }
        }
      } else if (options.index32x4) {
        for (size_t loop = 0; loop < options.loops; ++loop) {
          size_t limit = keys.size() - (keys.size() % 4);
          for (size_t i = 0; i < limit; i += 4) {
            ULONG index1 = 0;
            ULONG index2 = 0;
            ULONG index3 = 0;
            ULONG index4 = 0;
            jitInterface->Vtbl->Index32x4(jitInterface,
                                          keys[i],
                                          keys[i + 1],
                                          keys[i + 2],
                                          keys[i + 3],
                                          &index1,
                                          &index2,
                                          &index3,
                                          &index4);
            sink ^= index1 ^ index2 ^ index3 ^ index4;
          }
          for (size_t i = limit; i < keys.size(); ++i) {
            ULONG index = 0;
            jitInterface->Vtbl->Index32(jitInterface, keys[i], &index);
            sink ^= index;
          }
        }
      } else {
        for (size_t loop = 0; loop < options.loops; ++loop) {
          size_t limit = keys.size() - (keys.size() % 2);
          for (size_t i = 0; i < limit; i += 2) {
            ULONG index1 = 0;
            ULONG index2 = 0;
            jitInterface->Vtbl->Index32x2(jitInterface,
                                          keys[i],
                                          keys[i + 1],
                                          &index1,
                                          &index2);
            sink ^= index1 ^ index2;
          }
          for (size_t i = limit; i < keys.size(); ++i) {
            ULONG index = 0;
            jitInterface->Vtbl->Index32(jitInterface, keys[i], &index);
            sink ^= index;
          }
        }
      }

      return;
    }

    for (size_t loop = 0; loop < options.loops; ++loop) {
      for (ULONG key : keys) {
        ULONG index = 0;
        shim->Vtbl->Index(table, key, &index);
        sink ^= index;
      }
    }
  };

  auto unorderedLookup = [&]() {
    if (options.no_std_map_baselines) {
      return;
    }
    for (size_t loop = 0; loop < options.loops; ++loop) {
      for (ULONG key : keys) {
        auto it = unordered.find(key);
        if (it != unordered.end()) {
          sink ^= it->second;
        }
      }
    }
  };

  auto orderedLookup = [&]() {
    if (options.no_std_map_baselines) {
      return;
    }
    for (size_t loop = 0; loop < options.loops; ++loop) {
      for (ULONG key : keys) {
        auto it = ordered.find(key);
        if (it != ordered.end()) {
          sink ^= it->second;
        }
      }
    }
  };

  double perfectHashTotal = 0.0;
  double unorderedTotal = 0.0;
  double orderedTotal = 0.0;

  perfectHashLookup();
  unorderedLookup();
  orderedLookup();

  for (size_t iteration = 0; iteration < options.iterations; ++iteration) {
    perfectHashTotal += MeasureMillis(perfectHashLookup);
    unorderedTotal += MeasureMillis(unorderedLookup);
    orderedTotal += MeasureMillis(orderedLookup);
  }

  double perfectHashAvg = perfectHashTotal / options.iterations;
  double unorderedAvg = unorderedTotal / options.iterations;
  double orderedAvg = orderedTotal / options.iterations;

  if (verbose) {
    std::cout << "Key sort time:           " << sortMs << " ms\n";
    std::cout << "Build time (PerfectHash): " << tableCreateMs << " ms\n";
    if (options.no_std_map_baselines) {
      std::cout << "Build time (unordered_map): n/a\n";
      std::cout << "Build time (map): n/a\n";
    } else {
      std::cout << "Build time (unordered_map): " << unorderedBuildMs << " ms\n";
      std::cout << "Build time (map): " << orderedBuildMs << " ms\n";
    }
    if (options.jit) {
      std::cout << "JIT compile time:        " << jitCompileMs << " ms\n";
    }

    std::cout << "Lookup avg (PerfectHash): " << perfectHashAvg << " ms\n";
    if (options.no_std_map_baselines) {
      std::cout << "Lookup avg (unordered_map): n/a\n";
      std::cout << "Lookup avg (map): n/a\n";
    } else {
      std::cout << "Lookup avg (unordered_map): " << unorderedAvg << " ms\n";
      std::cout << "Lookup avg (map): " << orderedAvg << " ms\n";
    }
  }

  if (jitInterface) {
    jitInterface->Vtbl->Release(jitInterface);
  }

  shim->Vtbl->Release(table);
  online->Vtbl->Release(online);
  classFactory->Vtbl->Release(classFactory);

#ifdef PH_WINDOWS
  if (module) {
    FreeLibrary(module);
  }
#endif

  if (sink == 0 && verbose) {
    std::cout << "Sink: 0\n";
  }

  if (result) {
    result->key_count = keys.size();
    result->sort_ms = sortMs;
    result->table_create_ms = tableCreateMs;
    result->jit_compile_ms = jitCompileMs;
    result->unordered_build_ms = unorderedBuildMs;
    result->ordered_build_ms = orderedBuildMs;
    result->lookup_ph_ms = perfectHashAvg;
    result->lookup_unordered_ms = unorderedAvg;
    result->lookup_ordered_ms = orderedAvg;
    result->jit = options.jit;
    result->jit_max_isa = options.jit_max_isa;
    result->index32x2 = options.index32x2;
    result->index32x4 = options.index32x4;
    result->index32x8 = options.index32x8;
    result->index32x16 = options.index32x16;
    result->index32x16_available = index32x16_available;
  }

  return 0;
}

int RunCompare(const BenchmarkOptions &base) {
  BenchmarkOptions options = base;
  ApplySeed3Bytes(&options);

#ifndef PH_HAS_LLVM
  if (options.jit) {
    std::cout << "LLVM support disabled; JIT benchmarks skipped.\n";
  }
  options.jit = false;
#endif

  if (options.jit_backend == BenchmarkOptions::JitBackend::RawDog) {
    std::cerr << "--compare-isa requires the LLVM backend.\n";
    return 1;
  }

  if (!options.jit) {
    std::cerr << "--compare-isa requires JIT support.\n";
    return 1;
  }

  BenchmarkOptions templateOptions = options;
  templateOptions.index32x2 = false;
  templateOptions.index32x4 = false;
  templateOptions.index32x8 = false;
  templateOptions.index32x16 = false;
  templateOptions.jit_vector_index32x2 = false;
  templateOptions.jit_vector_index32x4 = false;
  templateOptions.jit_vector_index32x8 = false;

  auto isaRank = [](PERFECT_HASH_JIT_MAX_ISA_ID isa) -> int {
    switch (isa) {
      case PerfectHashJitMaxIsaAvx:
        return 1;
      case PerfectHashJitMaxIsaAvx2:
        return 2;
      case PerfectHashJitMaxIsaAvx512:
        return 3;
      case PerfectHashJitMaxIsaNeon:
        return 1;
      case PerfectHashJitMaxIsaSve:
        return 2;
      case PerfectHashJitMaxIsaSve2:
        return 3;
      default:
        return 3;
    }
  };

  auto isaAllowed = [&](PERFECT_HASH_JIT_MAX_ISA_ID isa) -> bool {
    if (options.jit_max_isa == PerfectHashJitMaxIsaAuto) {
      return true;
    }
    return isaRank(isa) <= isaRank(options.jit_max_isa);
  };

  struct CompareEntry {
    const char *label = nullptr;
    const char *index_label = nullptr;
    BenchmarkOptions options;
    BenchmarkResult result;
  };

  std::vector<CompareEntry> entries;
  entries.reserve(4);

  CompareEntry scalar;
  scalar.label = "scalar";
  scalar.index_label = "x1";
  scalar.options = templateOptions;
  scalar.options.jit_max_isa =
      options.jit_max_isa == PerfectHashJitMaxIsaAuto
          ? PerfectHashJitMaxIsaAvx
          : options.jit_max_isa;
  entries.push_back(scalar);

  if (isaAllowed(PerfectHashJitMaxIsaAvx)) {
    CompareEntry avx;
    avx.label = "avx";
    avx.index_label = "x4";
    avx.options = templateOptions;
    avx.options.jit_max_isa = PerfectHashJitMaxIsaAvx;
    avx.options.jit_vector_index32x4 = true;
    avx.options.index32x4 = true;
    entries.push_back(avx);
  }

  if (isaAllowed(PerfectHashJitMaxIsaAvx2)) {
    CompareEntry avx2;
    avx2.label = "avx2";
    avx2.index_label = "x8";
    avx2.options = templateOptions;
    avx2.options.jit_max_isa = PerfectHashJitMaxIsaAvx2;
    avx2.options.jit_vector_index32x8 = true;
    avx2.options.index32x8 = true;
    entries.push_back(avx2);
  }

  if (isaAllowed(PerfectHashJitMaxIsaAvx512)) {
    CompareEntry avx512;
    avx512.label = "avx512";
    avx512.index_label = "x16";
    avx512.options = templateOptions;
    avx512.options.jit_max_isa = PerfectHashJitMaxIsaAvx512;
    avx512.options.index32x16 = true;
    entries.push_back(avx512);
  }

  for (auto &entry : entries) {
    int runResult = RunBenchmark(entry.options, &entry.result, false);
    if (runResult != 0) {
      return runResult;
    }
  }

  const BenchmarkResult &baseline = entries.front().result;
  std::cout << "PerfectHash benchmark comparison\n";
  std::cout << "  Keys:       " << baseline.key_count << "\n";
  if (options.use_keys_file) {
    std::cout << "  Keys file:  " << options.keys_file << "\n";
  }
  std::cout << "  Hash:       " << options.hash_function_name << "\n";
  std::cout << "  Graph Impl: " << options.graph_impl << "\n";
  std::cout << "  Iterations: " << options.iterations << "\n";
  std::cout << "  Loops:      " << options.loops << "\n";
  std::cout << "  JIT Max ISA cap: "
            << JitMaxIsaToString(options.jit_max_isa) << "\n";
  if (options.seed_count > 0) {
    std::cout << "  Seeds:      ";
    for (size_t i = 0; i < options.seed_count; ++i) {
      std::cout << "0x" << std::hex << options.seeds[i] << std::dec;
      if (i + 1 < options.seed_count) {
        std::cout << ",";
      }
    }
    std::cout << "\n";
  }
  if (options.use_fixed_attempts) {
    std::cout << "  Fixed Attempts: " << options.fixed_attempts << "\n";
  }
  if (options.use_max_solve_time) {
    std::cout << "  Max Solve Time: " << options.max_solve_time << "s\n";
  }
  if (!baseline.jit_target_cpu.empty()) {
    std::cout << "  JIT Target CPU: " << baseline.jit_target_cpu << "\n";
  }
  if (!baseline.jit_target_features.empty()) {
    std::cout << "  JIT Target Features: " << baseline.jit_target_features << "\n";
  }

  std::cout << "Comparison summary (PerfectHash lookup avg, ms)\n";
  std::cout << std::left << std::setw(12) << "Mode"
            << std::setw(8) << "ISA"
            << std::setw(8) << "Index"
            << std::right << std::setw(12) << "Avg ms"
            << std::setw(12) << "Speedup"
            << " Delta\n";

  std::cout << std::fixed;
  for (const auto &entry : entries) {
    const BenchmarkResult &res = entry.result;
    double baselineMs = baseline.lookup_ph_ms;
    double speedup = baselineMs > 0.0 ? (baselineMs / res.lookup_ph_ms) : 0.0;
    double improvement = baselineMs > 0.0
                             ? (1.0 - (res.lookup_ph_ms / baselineMs)) * 100.0
                             : 0.0;
    std::cout << std::left << std::setw(12) << entry.label
              << std::setw(8) << JitMaxIsaToString(entry.options.jit_max_isa)
              << std::setw(8) << entry.index_label
              << std::right << std::setw(12) << std::setprecision(2)
              << res.lookup_ph_ms
              << std::setw(11) << std::setprecision(2) << speedup << "x"
              << " " << std::showpos << std::setprecision(1) << improvement
              << "%" << std::noshowpos << "\n";
  }
  std::cout << std::defaultfloat;

  if (!options.no_std_map_baselines) {
    double baselineMs = baseline.lookup_ph_ms;
    double unorderedSpeedup =
        baselineMs > 0.0 ? (baselineMs / baseline.lookup_unordered_ms) : 0.0;
    double orderedSpeedup =
        baselineMs > 0.0 ? (baselineMs / baseline.lookup_ordered_ms) : 0.0;
    std::cout << "Reference lookup avg (ms)\n";
    std::cout << "  unordered_map: " << baseline.lookup_unordered_ms << " ms ("
              << std::fixed << std::setprecision(2) << unorderedSpeedup << "x)\n";
    std::cout << "  map:           " << baseline.lookup_ordered_ms << " ms ("
              << std::fixed << std::setprecision(2) << orderedSpeedup << "x)\n";
    std::cout << std::defaultfloat;
  }

  for (const auto &entry : entries) {
    if (entry.options.index32x16 && !entry.result.index32x16_available) {
      std::cout << "Note: AVX-512 x16 unavailable; fell back to scalar.\n";
      break;
    }
  }

  return 0;
}

int RunBackendCompare(const BenchmarkOptions &base) {
  BenchmarkOptions options = base;
  ApplySeed3Bytes(&options);

#ifndef PH_HAS_LLVM
  std::cout << "LLVM support disabled; backend compare skipped.\n";
  return 1;
#endif

#ifndef PH_HAS_RAWDOG_JIT
  std::cout << "RawDog support disabled; backend compare skipped.\n";
  return 1;
#endif

  if (!options.jit) {
    std::cerr << "--compare-backends requires JIT support.\n";
    return 1;
  }

  if (options.jit_backend != BenchmarkOptions::JitBackend::Auto) {
    std::cerr << "--compare-backends ignores --jit-backend.\n";
  }

  BenchmarkOptions templateOptions = options;
  templateOptions.index32x2 = false;
  templateOptions.index32x4 = false;
  templateOptions.index32x8 = false;
  templateOptions.index32x16 = false;
  templateOptions.jit_vector_index32x2 = false;
  templateOptions.jit_vector_index32x4 = false;
  templateOptions.jit_vector_index32x8 = false;

  struct BackendEntry {
    const char *label = nullptr;
    BenchmarkOptions options;
    BenchmarkResult result;
  };

  std::vector<BackendEntry> entries;
  entries.reserve(2);

  BackendEntry llvm;
  llvm.label = "llvm";
  llvm.options = templateOptions;
  llvm.options.jit_backend = BenchmarkOptions::JitBackend::Llvm;
  entries.push_back(llvm);

  BackendEntry rawdog;
  rawdog.label = "rawdog";
  rawdog.options = templateOptions;
  rawdog.options.jit_backend = BenchmarkOptions::JitBackend::RawDog;
  entries.push_back(rawdog);

  for (auto &entry : entries) {
    int runResult = RunBenchmark(entry.options, &entry.result, false);
    if (runResult != 0) {
      return runResult;
    }
  }

  const BenchmarkResult &baseline = entries.front().result;
  std::cout << "PerfectHash JIT backend comparison\n";
  std::cout << "  Keys:       " << baseline.key_count << "\n";
  if (options.use_keys_file) {
    std::cout << "  Keys file:  " << options.keys_file << "\n";
  }
  std::cout << "  Hash:       " << options.hash_function_name << "\n";
  std::cout << "  Graph Impl: " << options.graph_impl << "\n";
  std::cout << "  Iterations: " << options.iterations << "\n";
  std::cout << "  Loops:      " << options.loops << "\n";
  if (options.seed_count > 0) {
    std::cout << "  Seeds:      ";
    for (size_t i = 0; i < options.seed_count; ++i) {
      std::cout << "0x" << std::hex << options.seeds[i] << std::dec;
      if (i + 1 < options.seed_count) {
        std::cout << ",";
      }
    }
    std::cout << "\n";
  }
  if (options.use_fixed_attempts) {
    std::cout << "  Fixed Attempts: " << options.fixed_attempts << "\n";
  }
  if (options.use_max_solve_time) {
    std::cout << "  Max Solve Time: " << options.max_solve_time << "s\n";
  }

  std::cout << "Backend summary (PerfectHash)\n";
  std::cout << std::left << std::setw(10) << "Backend"
            << std::right << std::setw(14) << "JIT ms"
            << std::setw(14) << "Lookup ms"
            << std::setw(12) << "Speedup"
            << " Delta\n";

  std::cout << std::fixed;
  for (const auto &entry : entries) {
    const BenchmarkResult &res = entry.result;
    double baselineMs = baseline.lookup_ph_ms;
    double speedup = baselineMs > 0.0 ? (baselineMs / res.lookup_ph_ms) : 0.0;
    double improvement = baselineMs > 0.0
                             ? (1.0 - (res.lookup_ph_ms / baselineMs)) * 100.0
                             : 0.0;
    std::cout << std::left << std::setw(10) << entry.label
              << std::right << std::setw(14) << std::setprecision(2)
              << res.jit_compile_ms
              << std::setw(14) << std::setprecision(2)
              << res.lookup_ph_ms
              << std::setw(11) << std::setprecision(2) << speedup << "x"
              << " " << std::showpos << std::setprecision(1) << improvement
              << "%" << std::noshowpos << "\n";
  }
  std::cout << std::defaultfloat;

  return 0;
}

int RunBackendCompareProcess(const BenchmarkOptions &base, const char *argv0) {
  BenchmarkOptions options = base;
  ApplySeed3Bytes(&options);

#ifdef PH_WINDOWS
  std::cout << "Process-based backend compare unsupported on Windows.\n";
  return 1;
#endif

#ifndef PH_HAS_LLVM
  std::cout << "LLVM support disabled; backend compare skipped.\n";
  return 1;
#endif

#ifndef PH_HAS_RAWDOG_JIT
  std::cout << "RawDog support disabled; backend compare skipped.\n";
  return 1;
#endif

  if (!options.jit) {
    std::cerr << "--compare-backends-process requires JIT support.\n";
    return 1;
  }

  if (options.process_iterations == 0) {
    std::cerr << "--process-iterations must be > 0.\n";
    return 1;
  }

  if (options.jit_backend != BenchmarkOptions::JitBackend::Auto) {
    std::cerr << "--compare-backends-process ignores --jit-backend.\n";
  }

  BenchmarkOptions templateOptions = options;
  templateOptions.index32x2 = false;
  templateOptions.index32x4 = false;
  templateOptions.index32x8 = false;
  templateOptions.index32x16 = false;
  templateOptions.jit_vector_index32x2 = false;
  templateOptions.jit_vector_index32x4 = false;
  templateOptions.jit_vector_index32x8 = false;
  templateOptions.compare_backends = false;
  templateOptions.compare_backends_process = false;
  templateOptions.compare_isa = false;
  templateOptions.process_child = false;

  std::string exe = GetExecutablePath(argv0);
  std::vector<std::string> baseArgs = BuildChildArgs(templateOptions, exe);

  struct BackendEntry {
    const char *label = nullptr;
    BenchmarkOptions::JitBackend backend = BenchmarkOptions::JitBackend::Auto;
    double avg_ms = 0.0;
  };

  std::vector<BackendEntry> entries;
  entries.reserve(2);
  entries.push_back({"llvm", BenchmarkOptions::JitBackend::Llvm, 0.0});
  entries.push_back({"rawdog", BenchmarkOptions::JitBackend::RawDog, 0.0});

  for (auto &entry : entries) {
    std::vector<std::string> args = baseArgs;
    args.emplace_back(std::string("--jit-backend=") +
                      JitBackendToString(entry.backend));

    double total_ms = 0.0;
    for (size_t i = 0; i < options.process_iterations; ++i) {
      double elapsed_ms = 0.0;
      int runResult = RunChildProcess(args, &elapsed_ms);
      if (runResult != 0) {
        return runResult;
      }
      total_ms += elapsed_ms;
    }
    entry.avg_ms = total_ms / static_cast<double>(options.process_iterations);
  }

  size_t key_count = options.key_count;
  if (options.use_keys_file) {
    std::ifstream file(options.keys_file, std::ios::binary | std::ios::ate);
    if (file) {
      std::streamsize size = file.tellg();
      if (size > 0 &&
          (size % static_cast<std::streamsize>(sizeof(ULONG))) == 0) {
        key_count = static_cast<size_t>(size / sizeof(ULONG));
      }
    }
  }

  const BackendEntry &baseline = entries.front();
  std::cout << "PerfectHash JIT backend process comparison\n";
  std::cout << "  Keys:       " << key_count << "\n";
  if (options.use_keys_file) {
    std::cout << "  Keys file:  " << options.keys_file << "\n";
  }
  std::cout << "  Hash:       " << options.hash_function_name << "\n";
  std::cout << "  Graph Impl: " << options.graph_impl << "\n";
  std::cout << "  Iterations: " << options.iterations << "\n";
  std::cout << "  Loops:      " << options.loops << "\n";
  std::cout << "  Proc iters: " << options.process_iterations << "\n";
  if (options.seed_count > 0) {
    std::cout << "  Seeds:      ";
    for (size_t i = 0; i < options.seed_count; ++i) {
      std::cout << "0x" << std::hex << options.seeds[i] << std::dec;
      if (i + 1 < options.seed_count) {
        std::cout << ",";
      }
    }
    std::cout << "\n";
  }
  if (options.use_fixed_attempts) {
    std::cout << "  Fixed Attempts: " << options.fixed_attempts << "\n";
  }
  if (options.use_max_solve_time) {
    std::cout << "  Max Solve Time: " << options.max_solve_time << "s\n";
  }

  std::cout << "Backend process summary (avg wall time, ms)\n";
  std::cout << std::left << std::setw(10) << "Backend"
            << std::right << std::setw(14) << "Avg ms"
            << std::setw(12) << "Speedup"
            << " Delta\n";

  std::cout << std::fixed;
  for (const auto &entry : entries) {
    double baselineMs = baseline.avg_ms;
    double speedup = baselineMs > 0.0 ? (baselineMs / entry.avg_ms) : 0.0;
    double improvement =
        baselineMs > 0.0 ? (1.0 - (entry.avg_ms / baselineMs)) * 100.0 : 0.0;
    std::cout << std::left << std::setw(10) << entry.label
              << std::right << std::setw(14) << std::setprecision(2)
              << entry.avg_ms
              << std::setw(11) << std::setprecision(2) << speedup << "x"
              << " " << std::showpos << std::setprecision(1) << improvement
              << "%" << std::noshowpos << "\n";
  }
  std::cout << std::defaultfloat;

  return 0;
}

} // namespace

int main(int argc, char **argv) {
  BenchmarkOptions options;
  if (!ParseArguments(argc, argv, &options)) {
    PrintUsage(argv ? argv[0] : nullptr);
    return 1;
  }

  if (options.process_child) {
    return RunBenchmark(options, nullptr, false);
  }

  if (options.compare_backends_process) {
    if (options.compare_isa || options.compare_backends) {
      std::cerr << "--compare-backends-process is mutually exclusive with "
                   "--compare-isa and --compare-backends.\n";
      return 1;
    }
    return RunBackendCompareProcess(options, argv ? argv[0] : nullptr);
  }

  if (options.compare_backends) {
    if (options.compare_isa) {
      std::cerr << "--compare-isa and --compare-backends are mutually "
                   "exclusive.\n";
      return 1;
    }
    return RunBackendCompare(options);
  }

  if (options.compare_isa) {
    return RunCompare(options);
  }

  return RunBenchmark(options, nullptr, true);
}
