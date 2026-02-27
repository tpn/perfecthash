#include <PerfectHashOnlineRawdog.h>

#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <unordered_set>
#include <vector>

namespace {

std::string ToHex(int32_t hr) {
  std::ostringstream stream;
  stream << "0x" << std::uppercase << std::hex
         << static_cast<uint32_t>(hr);
  return stream.str();
}

bool Succeeded(int32_t hr) { return hr >= 0; }

PH_ONLINE_RAWDOG_HASH_FUNCTION ParseHashFunction(const std::string &name) {
  if (name == "multiplyshiftr") {
    return PhOnlineRawdogHashMultiplyShiftR;
  } else if (name == "multiplyshiftlr") {
    return PhOnlineRawdogHashMultiplyShiftLR;
  } else if (name == "multiplyshiftrmultiply") {
    return PhOnlineRawdogHashMultiplyShiftRMultiply;
  } else if (name == "multiplyshiftr2") {
    return PhOnlineRawdogHashMultiplyShiftR2;
  } else if (name == "multiplyshiftrx") {
    return PhOnlineRawdogHashMultiplyShiftRX;
  } else if (name == "mulshrolate1rx") {
    return PhOnlineRawdogHashMulshrolate1RX;
  } else if (name == "mulshrolate2rx") {
    return PhOnlineRawdogHashMulshrolate2RX;
  } else if (name == "mulshrolate3rx") {
    return PhOnlineRawdogHashMulshrolate3RX;
  } else if (name == "mulshrolate4rx") {
    return PhOnlineRawdogHashMulshrolate4RX;
  }

  return PhOnlineRawdogHashMulshrolate2RX;
}

void PrintUsage(const char *argv0) {
  std::cout << "Usage: " << argv0
            << " [--hash <name>] [--vector-width <0|1|2|4|8|16>]\n";
  std::cout << "Hash names: multiplyshiftr, multiplyshiftlr, "
               "multiplyshiftrmultiply,\n";
  std::cout << "            multiplyshiftr2, multiplyshiftrx, mulshrolate1rx,\n";
  std::cout << "            mulshrolate2rx, mulshrolate3rx, mulshrolate4rx\n";
}

}  // namespace

int main(int argc, char **argv) {
  constexpr int32_t kPhNotImplemented = static_cast<int32_t>(0xE0040230u);
  std::string hash_name = "mulshrolate2rx";
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

    if (arg == "--vector-width" && index + 1 < argc) {
      vector_width = static_cast<uint32_t>(std::stoul(argv[++index]));
      continue;
    }

    std::cerr << "Unknown argument: " << arg << "\n";
    PrintUsage(argv[0]);
    return 2;
  }

  const std::vector<uint32_t> keys = {
      1,   3,   5,   7,   11,  13,  17,  19,  23,  29,  31,  37,  41,
      43,  47,  53,  59,  61,  67,  71,  73,  79,  83,  89,  97,  101,
      103, 107, 109, 113, 127, 131, 137, 139, 149, 151, 157, 163, 167,
      173, 179, 181, 191, 193, 197, 199, 211, 223, 227, 229, 233, 239,
      241, 251, 257, 263, 269, 271, 277, 281, 283, 293, 307, 311,
  };

  PH_ONLINE_RAWDOG_CONTEXT *context = nullptr;
  PH_ONLINE_RAWDOG_TABLE *table = nullptr;

  int32_t result = PhOnlineRawdogOpen(&context);
  if (!Succeeded(result)) {
    std::cerr << "PhOnlineRawdogOpen() failed: " << ToHex(result) << "\n";
    return 1;
  }

  result = PhOnlineRawdogCreateTable32(context,
                                       ParseHashFunction(hash_name),
                                       keys.data(),
                                       static_cast<uint64_t>(keys.size()),
                                       &table);
  if (!Succeeded(result)) {
    std::cerr << "PhOnlineRawdogCreateTable32() failed: " << ToHex(result)
              << "\n";
    PhOnlineRawdogClose(context);
    return 1;
  }

  result = PhOnlineRawdogCompileTable(context,
                                      table,
                                      vector_width,
                                      PhOnlineRawdogJitMaxIsaAuto);
  bool rawdog_jit_enabled = Succeeded(result);
  if (!rawdog_jit_enabled) {
    if (result == kPhNotImplemented) {
      std::cerr
          << "RawDog JIT is not available for this host/table combination; "
             "continuing with the non-JIT path.\n";
    } else {
      std::cerr << "PhOnlineRawdogCompileTable() failed: " << ToHex(result)
                << "\n";
      PhOnlineRawdogReleaseTable(table);
      PhOnlineRawdogClose(context);
      return 1;
    }
  }

  std::unordered_set<uint32_t> seen;
  seen.reserve(keys.size());

  for (const uint32_t key : keys) {
    uint32_t index = 0;
    result = PhOnlineRawdogIndex32(table, key, &index);
    if (!Succeeded(result)) {
      std::cerr << "PhOnlineRawdogIndex32(" << key
                << ") failed: " << ToHex(result) << "\n";
      PhOnlineRawdogReleaseTable(table);
      PhOnlineRawdogClose(context);
      return 1;
    }
    if (!seen.insert(index).second) {
      std::cerr << "Duplicate index detected for key " << key << ": " << index
                << "\n";
      PhOnlineRawdogReleaseTable(table);
      PhOnlineRawdogClose(context);
      return 1;
    }
  }

  std::cout << "Success: " << keys.size()
            << " keys mapped to unique indices with hash=" << hash_name
            << ", vector-width=" << vector_width << ", mode="
            << (rawdog_jit_enabled ? "rawdog-jit" : "slow-index-fallback")
            << ".\n";

  PhOnlineRawdogReleaseTable(table);
  PhOnlineRawdogClose(context);
  return 0;
}
