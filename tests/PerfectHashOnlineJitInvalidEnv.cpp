#include <PerfectHash/PerfectHashOnlineJit.h>

#include <cstdint>
#include <memory>
#include <vector>

int main() {
  auto context = std::unique_ptr<PH_ONLINE_JIT_CONTEXT, decltype(&PhOnlineJitClose)>(
      nullptr, &PhOnlineJitClose);
  PH_ONLINE_JIT_CONTEXT *raw_context = nullptr;
  PH_ONLINE_JIT_TABLE *raw_table = nullptr;
  std::vector<uint32_t> keys = {1, 3, 5, 7, 11, 13, 17, 19};

  if (PhOnlineJitOpen(&raw_context) < 0 || !raw_context) {
    return 2;
  }
  context.reset(raw_context);

  auto result = PhOnlineJitCreateTable32(context.get(),
                                         PhOnlineJitHashMulshrolate3RX,
                                         keys.data(),
                                         static_cast<uint64_t>(keys.size()),
                                         &raw_table);
  if (raw_table) {
    PhOnlineJitReleaseTable(raw_table);
    raw_table = nullptr;
  }

  return result < 0 ? 0 : 1;
}
