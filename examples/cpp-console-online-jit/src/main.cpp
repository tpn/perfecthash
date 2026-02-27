#include <PerfectHashOnlineJit.h>

#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <unordered_set>
#include <vector>

#if !defined(_WIN32) && defined(PH_ONLINE_JIT_LLVM_LIBRARY_PATH)
#include <dlfcn.h>
#endif

namespace {

std::string ToHex(int32_t hr) {
  std::ostringstream stream;
  stream << "0x" << std::uppercase << std::hex
         << static_cast<uint32_t>(hr);
  return stream.str();
}

bool Succeeded(int32_t hr) { return hr >= 0; }

PH_ONLINE_JIT_HASH_FUNCTION ParseHashFunction(const std::string &name) {
  if (name == "multiplyshiftr") {
    return PhOnlineJitHashMultiplyShiftR;
  } else if (name == "multiplyshiftlr") {
    return PhOnlineJitHashMultiplyShiftLR;
  } else if (name == "multiplyshiftrmultiply") {
    return PhOnlineJitHashMultiplyShiftRMultiply;
  } else if (name == "multiplyshiftr2") {
    return PhOnlineJitHashMultiplyShiftR2;
  } else if (name == "multiplyshiftrx") {
    return PhOnlineJitHashMultiplyShiftRX;
  } else if (name == "mulshrolate1rx") {
    return PhOnlineJitHashMulshrolate1RX;
  } else if (name == "mulshrolate2rx") {
    return PhOnlineJitHashMulshrolate2RX;
  } else if (name == "mulshrolate3rx") {
    return PhOnlineJitHashMulshrolate3RX;
  } else if (name == "mulshrolate4rx") {
    return PhOnlineJitHashMulshrolate4RX;
  }

  return PhOnlineJitHashMulshrolate2RX;
}

PH_ONLINE_JIT_BACKEND ParseBackend(const std::string &name) {
  if (name == "rawdog-jit") {
    return PhOnlineJitBackendRawDogJit;
  }
  if (name == "llvm-jit") {
    return PhOnlineJitBackendLlvmJit;
  }
  if (name == "auto") {
    return PhOnlineJitBackendAuto;
  }

  return PhOnlineJitBackendRawDogJit;
}

const char *BackendToString(PH_ONLINE_JIT_BACKEND backend) {
  switch (backend) {
    case PhOnlineJitBackendAuto:
      return "auto";
    case PhOnlineJitBackendRawDogJit:
      return "rawdog-jit";
    case PhOnlineJitBackendLlvmJit:
      return "llvm-jit";
    default:
      return "unknown";
  }
}

void PreloadLlvmRuntimeLibrary(PH_ONLINE_JIT_BACKEND backend) {
#if !defined(_WIN32) && defined(PH_ONLINE_JIT_LLVM_LIBRARY_PATH)
  if (backend == PhOnlineJitBackendLlvmJit ||
      backend == PhOnlineJitBackendAuto) {
    void *handle = dlopen(PH_ONLINE_JIT_LLVM_LIBRARY_PATH, RTLD_NOW | RTLD_GLOBAL);
    if (!handle) {
      std::cerr << "Warning: unable to preload LLVM runtime from "
                << PH_ONLINE_JIT_LLVM_LIBRARY_PATH << ": " << dlerror() << "\n";
    }
  }
#else
  (void)backend;
#endif
}

void PrintUsage(const char *argv0) {
  std::cout << "Usage: " << argv0
            << " [--backend <rawdog-jit|llvm-jit|auto>]"
               " [--hash <name>] [--vector-width <0|1|2|4|8|16>]\n";
  std::cout << "Hash names: multiplyshiftr, multiplyshiftlr, "
               "multiplyshiftrmultiply,\n";
  std::cout << "            multiplyshiftr2, multiplyshiftrx, mulshrolate1rx,\n";
  std::cout << "            mulshrolate2rx, mulshrolate3rx, mulshrolate4rx\n";
}

}  // namespace

int main(int argc, char **argv) {
  constexpr int32_t kPhNotImplemented = static_cast<int32_t>(0xE0040230u);
  constexpr int32_t kPhLlvmBackendNotFound =
      static_cast<int32_t>(0xE004041Cu);

  std::string hash_name = "mulshrolate2rx";
  std::string backend_name = "rawdog-jit";
  uint32_t vector_width = 16;

  for (int index = 1; index < argc; ++index) {
    const std::string arg = argv[index];
    if (arg == "--help" || arg == "-h") {
      PrintUsage(argv[0]);
      return 0;
    }

    if (arg == "--hash" && index + 1 < argc) {
      hash_name = argv[++index];
      continue;
    }

    if (arg == "--backend" && index + 1 < argc) {
      backend_name = argv[++index];
      continue;
    }

    if (arg == "--vector-width" && index + 1 < argc) {
      vector_width = static_cast<uint32_t>(std::stoul(argv[++index]));
      continue;
    }

    std::cerr << "Unknown argument: " << arg << "\n";
    PrintUsage(argv[0]);
    return 2;
  }

  const PH_ONLINE_JIT_BACKEND backend = ParseBackend(backend_name);
  PreloadLlvmRuntimeLibrary(backend);

  const std::vector<uint32_t> keys = {
      1,   3,   5,   7,   11,  13,  17,  19,  23,  29,  31,  37,  41,
      43,  47,  53,  59,  61,  67,  71,  73,  79,  83,  89,  97,  101,
      103, 107, 109, 113, 127, 131, 137, 139, 149, 151, 157, 163, 167,
      173, 179, 181, 191, 193, 197, 199, 211, 223, 227, 229, 233, 239,
      241, 251, 257, 263, 269, 271, 277, 281, 283, 293, 307, 311,
  };

  PH_ONLINE_JIT_CONTEXT *context = nullptr;
  PH_ONLINE_JIT_TABLE *table = nullptr;

  int32_t result = PhOnlineJitOpen(&context);
  if (!Succeeded(result)) {
    std::cerr << "PhOnlineJitOpen() failed: " << ToHex(result) << "\n";
    return 1;
  }

  result = PhOnlineJitCreateTable32(context,
                                    ParseHashFunction(hash_name),
                                    keys.data(),
                                    static_cast<uint64_t>(keys.size()),
                                    &table);
  if (!Succeeded(result)) {
    std::cerr << "PhOnlineJitCreateTable32() failed: " << ToHex(result)
              << "\n";
    PhOnlineJitClose(context);
    return 1;
  }

  result = PhOnlineJitCompileTable(context,
                                   table,
                                   backend,
                                   vector_width,
                                   PhOnlineJitMaxIsaAuto);
  bool jit_enabled = Succeeded(result);
  if (!jit_enabled) {
    if (result == kPhNotImplemented || result == kPhLlvmBackendNotFound) {
      std::cerr << "Selected JIT backend is not available for this "
                   "host/table combination; continuing with non-JIT path.\n";
    } else {
      std::cerr << "PhOnlineJitCompileTable() failed: " << ToHex(result)
                << "\n";
      PhOnlineJitReleaseTable(table);
      PhOnlineJitClose(context);
      return 1;
    }
  }

  std::unordered_set<uint32_t> seen;
  seen.reserve(keys.size());

  for (const uint32_t key : keys) {
    uint32_t index = 0;
    result = PhOnlineJitIndex32(table, key, &index);
    if (!Succeeded(result)) {
      std::cerr << "PhOnlineJitIndex32(" << key
                << ") failed: " << ToHex(result) << "\n";
      PhOnlineJitReleaseTable(table);
      PhOnlineJitClose(context);
      return 1;
    }
    if (!seen.insert(index).second) {
      std::cerr << "Duplicate index detected for key " << key << ": " << index
                << "\n";
      PhOnlineJitReleaseTable(table);
      PhOnlineJitClose(context);
      return 1;
    }
  }

  std::cout << "Success: " << keys.size()
            << " keys mapped to unique indices with hash=" << hash_name
            << ", backend=" << BackendToString(backend)
            << ", vector-width=" << vector_width << ", mode="
            << (jit_enabled ? "jit" : "slow-index-fallback") << ".\n";

  PhOnlineJitReleaseTable(table);
  PhOnlineJitClose(context);
  return 0;
}
