#include <chrono>
#include <cinttypes>
#include <cstdint>
#include <cstring>
#include <algorithm>
#include <iostream>
#include <map>
#include <random>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

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
  bool index2 = false;
  bool index4 = false;
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

    if (std::strcmp(arg, "--no-jit") == 0) {
      options->jit = false;
      continue;
    }

    if (std::strcmp(arg, "--index2") == 0) {
      options->index2 = true;
      continue;
    }

    if (std::strcmp(arg, "--index4") == 0) {
      options->index4 = true;
      continue;
    }

    return false;
  }

  return true;
}

void PrintUsage(const char *exe) {
  std::cout << "Usage: " << (exe ? exe : "perfecthash_benchmarks")
            << " [--keys=N] [--iterations=N] [--loops=N] [--seed=N]"
            << " [--no-jit] [--index2] [--index4]\n";
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

} // namespace

int main(int argc, char **argv) {
  BenchmarkOptions options;
  if (!ParseArguments(argc, argv, &options)) {
    PrintUsage(argv ? argv[0] : nullptr);
    return 1;
  }

  if (options.index4) {
    options.index2 = false;
  }

#ifndef PH_HAS_LLVM
  if (options.jit) {
    std::cout << "LLVM support disabled; JIT benchmarks skipped.\n";
  }
  options.jit = false;
#endif

  const auto keys = BuildKeys(options.key_count, options.seed);
  auto sorted_keys = keys;
  double sortMs = MeasureMillis([&]() {
    std::sort(sorted_keys.begin(), sorted_keys.end());
  });

  std::cout << "PerfectHash benchmarks\n";
  std::cout << "  Keys:       " << keys.size() << "\n";
  std::cout << "  Iterations: " << options.iterations << "\n";
  std::cout << "  Loops:      " << options.loops << "\n";
  std::cout << "  JIT:        " << (options.jit ? "on" : "off") << "\n";
  std::cout << "  Index2:     " << (options.index2 ? "on" : "off") << "\n";
  std::cout << "  Index4:     " << (options.index4 ? "on" : "off") << "\n";

  PICLASSFACTORY classFactory = nullptr;
  PPERFECT_HASH_ONLINE online = nullptr;
  PPERFECT_HASH_PRINT_ERROR printError = nullptr;
  PPERFECT_HASH_PRINT_MESSAGE printMessage = nullptr;
  HMODULE module = nullptr;

  HRESULT result = PerfectHashBootstrap(
      &classFactory, &printError, &printMessage, &module);
  if (result < 0) {
    std::cerr << "PerfectHashBootstrap failed: 0x" << std::hex
              << static_cast<unsigned>(result) << std::dec << "\n";
    return 1;
  }

  result = classFactory->Vtbl->CreateInstance(
      classFactory,
      nullptr,
#ifdef PH_WINDOWS
      IID_PERFECT_HASH_ONLINE,
#else
      &IID_PERFECT_HASH_ONLINE,
#endif
      reinterpret_cast<void **>(&online));

  if (result < 0) {
    std::cerr << "CreateInstance(Online) failed: 0x" << std::hex
              << static_cast<unsigned>(result) << std::dec << "\n";
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

  auto tableCreateMs = MeasureMillis([&]() {
    result = online->Vtbl->CreateTableFromKeys(
        online,
        PerfectHashChm01AlgorithmId,
        PerfectHashHashMultiplyShiftRFunctionId,
        PerfectHashAndMaskFunctionId,
        sizeof(ULONG),
        static_cast<ULONGLONG>(sorted_keys.size()),
        const_cast<ULONG *>(sorted_keys.data()),
        nullptr,
        &tableFlags,
        nullptr,
        &table);
  });

  if (result < 0 || !table) {
    std::cerr << "CreateTableFromKeys failed: 0x" << std::hex
              << static_cast<unsigned>(result) << std::dec << "\n";
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
  if (options.index2) {
    compileFlags.JitIndex2 = TRUE;
  }
  if (options.index4) {
    compileFlags.JitIndex4 = TRUE;
  }

  double jitCompileMs = 0.0;
  if (options.jit) {
    jitCompileMs = MeasureMillis([&]() {
      result = online->Vtbl->CompileTable(online, table, &compileFlags);
    });

    if (result < 0) {
      std::cerr << "CompileTable failed: 0x" << std::hex
                << static_cast<unsigned>(result) << std::dec << "\n";
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
  if (options.jit && (options.index2 || options.index4)) {
    result = shim->Vtbl->QueryInterface(
        table,
#ifdef PH_WINDOWS
        IID_PERFECT_HASH_TABLE_JIT_INTERFACE,
#else
        &IID_PERFECT_HASH_TABLE_JIT_INTERFACE,
#endif
        reinterpret_cast<void **>(&jitInterface));
    if (result < 0 || !jitInterface) {
      jitInterface = nullptr;
    }
  }

  std::unordered_map<ULONG, ULONG> unordered;
  double unorderedBuildMs = MeasureMillis([&]() {
    unordered.reserve(keys.size());
    for (size_t i = 0; i < keys.size(); ++i) {
      unordered.emplace(keys[i], static_cast<ULONG>(i));
    }
  });

  std::map<ULONG, ULONG> ordered;
  double orderedBuildMs = MeasureMillis([&]() {
    for (size_t i = 0; i < keys.size(); ++i) {
      ordered.emplace(keys[i], static_cast<ULONG>(i));
    }
  });

  volatile ULONG sink = 0;

  auto perfectHashLookup = [&]() {
    if (options.jit && jitInterface && (options.index2 || options.index4)) {
      if (options.index4) {
        for (size_t loop = 0; loop < options.loops; ++loop) {
          size_t limit = keys.size() - (keys.size() % 4);
          for (size_t i = 0; i < limit; i += 4) {
            ULONG index1 = 0;
            ULONG index2 = 0;
            ULONG index3 = 0;
            ULONG index4 = 0;
            jitInterface->Vtbl->Index4(jitInterface,
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
            jitInterface->Vtbl->Index(jitInterface, keys[i], &index);
            sink ^= index;
          }
        }
      } else {
        for (size_t loop = 0; loop < options.loops; ++loop) {
          size_t limit = keys.size() - (keys.size() % 2);
          for (size_t i = 0; i < limit; i += 2) {
            ULONG index1 = 0;
            ULONG index2 = 0;
            jitInterface->Vtbl->Index2(jitInterface,
                                       keys[i],
                                       keys[i + 1],
                                       &index1,
                                       &index2);
            sink ^= index1 ^ index2;
          }
          for (size_t i = limit; i < keys.size(); ++i) {
            ULONG index = 0;
            jitInterface->Vtbl->Index(jitInterface, keys[i], &index);
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

  std::cout << "Key sort time:           " << sortMs << " ms\n";
  std::cout << "Build time (PerfectHash): " << tableCreateMs << " ms\n";
  std::cout << "Build time (unordered_map): " << unorderedBuildMs << " ms\n";
  std::cout << "Build time (map): " << orderedBuildMs << " ms\n";
  if (options.jit) {
    std::cout << "JIT compile time:        " << jitCompileMs << " ms\n";
  }

  std::cout << "Lookup avg (PerfectHash): "
            << (perfectHashTotal / options.iterations) << " ms\n";
  std::cout << "Lookup avg (unordered_map): "
            << (unorderedTotal / options.iterations) << " ms\n";
  std::cout << "Lookup avg (map): "
            << (orderedTotal / options.iterations) << " ms\n";

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

  if (sink == 0) {
    std::cout << "Sink: 0\n";
  }

  return 0;
}
