// Ask for the LLVM backend twice and print each result. Tests run this where
// the PerfectHashLLVM plugin it loads lacks entry points, is absent, or cannot
// load LLVM.

#include <PerfectHash/PerfectHashOnlineJit.h>

#include <cstdint>
#include <cstdio>
#include <vector>

int main() {
  PH_ONLINE_JIT_CONTEXT *context = nullptr;
  PH_ONLINE_JIT_TABLE *table = nullptr;
  std::vector<uint32_t> keys = {1, 3, 5, 7, 11, 13, 17, 19};

  if (PhOnlineJitOpen(&context) < 0 || !context) {
    return 2;
  }
  if (PhOnlineJitCreateTable32(context,
                               PhOnlineJitHashMultiplyShiftR,
                               keys.data(),
                               static_cast<uint64_t>(keys.size()),
                               &table) < 0 || !table) {
    PhOnlineJitClose(context);
    return 3;
  }

  for (int attempt = 0; attempt < 2; attempt++) {
    int32_t result = PhOnlineJitCompileTable(context,
                                             table,
                                             PhOnlineJitBackendLlvmJit,
                                             1,
                                             PhOnlineJitMaxIsaAuto);
    std::printf("0x%08x\n", static_cast<unsigned>(result));
  }

  PhOnlineJitReleaseTable(table);
  PhOnlineJitClose(context);
  return 0;
}
