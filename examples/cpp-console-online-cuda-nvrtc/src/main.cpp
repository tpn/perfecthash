#include <PerfectHash/PerfectHashOnlineJit.h>

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <nvJitLink.h>
#include <nvrtc.h>

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#if !defined(_WIN32) && defined(PH_ONLINE_JIT_LLVM_LIBRARY_PATH)
#include <dlfcn.h>
#endif

#if !defined(_WIN32)
#include <sys/resource.h>
#include <unistd.h>
#endif

namespace {

using steady_clock = std::chrono::steady_clock;

std::string to_hex(std::uint32_t value)
{
  std::ostringstream stream;
  stream << "0x" << std::uppercase << std::hex << value;
  return stream.str();
}

bool succeeded(std::int32_t hr) { return hr >= 0; }

bool is_supported_hash_name(std::string const& name)
{
  return (name == "multiplyshiftr" || name == "multiplyshiftlr" ||
          name == "multiplyshiftrmultiply" || name == "multiplyshiftr2" ||
          name == "multiplyshiftrx" ||
          name == "mulshrolate1rx" || name == "mulshrolate2rx" ||
          name == "mulshrolate3rx" || name == "mulshrolate4rx");
}

PH_ONLINE_JIT_HASH_FUNCTION parse_hash_function(std::string const& name)
{
  if (name == "multiplyshiftr") { return PhOnlineJitHashMultiplyShiftR; }
  if (name == "multiplyshiftlr") { return PhOnlineJitHashMultiplyShiftLR; }
  if (name == "multiplyshiftrmultiply") { return PhOnlineJitHashMultiplyShiftRMultiply; }
  if (name == "multiplyshiftr2") { return PhOnlineJitHashMultiplyShiftR2; }
  if (name == "multiplyshiftrx") { return PhOnlineJitHashMultiplyShiftRX; }
  if (name == "mulshrolate1rx") { return PhOnlineJitHashMulshrolate1RX; }
  if (name == "mulshrolate2rx") { return PhOnlineJitHashMulshrolate2RX; }
  if (name == "mulshrolate3rx") { return PhOnlineJitHashMulshrolate3RX; }
  if (name == "mulshrolate4rx") { return PhOnlineJitHashMulshrolate4RX; }
  return PhOnlineJitHashMulshrolate3RX;
}

enum class compile_mode { ptx, lto };
enum class lookup_mode { direct, split, warpcache, blocksort };
enum class table_load_mode { generic, readonly };

compile_mode parse_compile_mode(std::string const& name)
{
  if (name == "ptx") { return compile_mode::ptx; }
  if (name == "lto") { return compile_mode::lto; }
  return compile_mode::ptx;
}

char const* compile_mode_to_string(compile_mode mode)
{
  switch (mode) {
    case compile_mode::ptx: return "ptx";
    case compile_mode::lto: return "lto";
  }
  return "unknown";
}

bool is_supported_backend_name(std::string const& name)
{
  return (name == "none" || name == "auto" || name == "rawdog-jit" ||
          name == "llvm-jit");
}

PH_ONLINE_JIT_BACKEND parse_backend(std::string const& name)
{
  if (name == "rawdog-jit") { return PhOnlineJitBackendRawDogJit; }
  if (name == "llvm-jit") { return PhOnlineJitBackendLlvmJit; }
  if (name == "auto") { return PhOnlineJitBackendAuto; }
  return PhOnlineJitBackendRawDogJit;
}

char const* backend_to_string(PH_ONLINE_JIT_BACKEND backend)
{
  switch (backend) {
    case PhOnlineJitBackendAuto: return "auto";
    case PhOnlineJitBackendRawDogJit: return "rawdog-jit";
    case PhOnlineJitBackendLlvmJit: return "llvm-jit";
  }
  return "unknown";
}

void preload_llvm_runtime_library(std::string const& backend_name)
{
#if !defined(_WIN32) && defined(PH_ONLINE_JIT_LLVM_LIBRARY_PATH)
  if (backend_name == "llvm-jit" || backend_name == "auto") {
    void* handle = dlopen(PH_ONLINE_JIT_LLVM_LIBRARY_PATH, RTLD_NOW | RTLD_GLOBAL);
    if (!handle) {
      std::cerr << "Warning: unable to preload LLVM runtime from "
                << PH_ONLINE_JIT_LLVM_LIBRARY_PATH << ": " << dlerror() << "\n";
    }
  }
#else
  (void)backend_name;
#endif
}

lookup_mode parse_lookup_mode(std::string const& name)
{
  if (name == "direct") { return lookup_mode::direct; }
  if (name == "split") { return lookup_mode::split; }
  if (name == "warpcache") { return lookup_mode::warpcache; }
  if (name == "blocksort") { return lookup_mode::blocksort; }
  return lookup_mode::direct;
}

char const* lookup_mode_to_string(lookup_mode mode)
{
  switch (mode) {
    case lookup_mode::direct: return "direct";
    case lookup_mode::split: return "split";
    case lookup_mode::warpcache: return "warpcache";
    case lookup_mode::blocksort: return "blocksort";
  }
  return "unknown";
}

table_load_mode parse_table_load_mode(std::string const& name)
{
  if (name == "generic") { return table_load_mode::generic; }
  if (name == "readonly") { return table_load_mode::readonly; }
  return table_load_mode::generic;
}

char const* table_load_mode_to_string(table_load_mode mode)
{
  switch (mode) {
    case table_load_mode::generic: return "generic";
    case table_load_mode::readonly: return "readonly";
  }
  return "unknown";
}

struct options {
  std::string hash_name = "mulshrolate3rx";
  std::string compile_mode_name = "ptx";
  std::string lookup_mode_name = "direct";
  std::string table_load_mode_name = "generic";
  std::string cpu_backend_name = "none";
  std::string keys_file;
  std::string probe_keys_file;
  std::string source_out_path;
  int device_ordinal = 0;
  int items_per_thread = 4;
  int threads = 128;
  int cpu_vector_width = 16;
  int iterations = 10;
  int warmup = 2;
  std::uint64_t max_keys = 0;
  std::uint64_t max_probe_keys = 0;
  bool dump_fragment = false;
  bool csv = false;
  bool csv_header = false;
  bool cpu_strict_vector_width = false;
  bool analyze_slot_reuse = false;
  bool analysis_only = false;
  bool embed_table_data = false;
  bool verify = true;
};

struct benchmark_result {
  std::string key_source;
  std::string probe_source;
  std::string hash_name;
  std::string device_name;
  std::string cpu_backend_requested;
  std::string cpu_backend_effective;
  std::string cpu_key_mode;
  compile_mode mode = compile_mode::ptx;
  lookup_mode lookup = lookup_mode::direct;
  table_load_mode table_load = table_load_mode::generic;
  int device_ordinal = 0;
  int items_per_thread = 0;
  int threads = 0;
  int blocks = 0;
  int warmup = 0;
  int iterations = 0;
  int cpu_vector_requested = 0;
  int cpu_vector_effective = 0;
  int major = 0;
  int minor = 0;
  bool embedded_table_data = false;
  bool cpu_enabled = false;
  bool cpu_strict_vector_width = false;
  bool cpu_jit_enabled = false;
  bool slot_reuse_analyzed = false;
  std::size_t key_count = 0;
  std::size_t key_bytes = 0;
  std::size_t probe_key_count = 0;
  std::size_t probe_key_bytes = 0;
  std::size_t table_data_bytes = 0;
  std::size_t table_data_elements = 0;
  std::size_t fragment_bytes = 0;
  std::size_t combined_bytes = 0;
  std::size_t image_bytes = 0;
  std::uint32_t table_data_element_size = 0;
  std::int32_t cpu_compile_hr = 0;
  std::size_t host_rss_before_build_bytes = 0;
  std::size_t host_rss_after_build_bytes = 0;
  std::size_t host_rss_delta_bytes = 0;
  std::size_t host_peak_rss_bytes = 0;
  std::size_t vram_explicit_bytes = 0;
  std::size_t vram_total_bytes = 0;
  double build_ms = 0.0;
  double table_export_ms = 0.0;
  double emit_ms = 0.0;
  double compose_ms = 0.0;
  double compile_ms = 0.0;
  double link_ms = 0.0;
  double module_load_ms = 0.0;
  double alloc_ms = 0.0;
  double h2d_ms = 0.0;
  double table_h2d_ms = 0.0;
  double cpu_compile_ms = 0.0;
  double cpu_lookup_avg_ms = 0.0;
  double cpu_lookup_min_ms = 0.0;
  double cpu_lookup_max_ms = 0.0;
  double cpu_lookup_ns_per_key = 0.0;
  double slot_compute_avg_ms = 0.0;
  double slot_gather_avg_ms = 0.0;
  double warp_unique_requests_avg = 0.0;
  double warp_duplicate_ratio = 0.0;
  double block_unique_requests_avg = 0.0;
  double block_duplicate_ratio = 0.0;
  double kernel_avg_ms = 0.0;
  double kernel_min_ms = 0.0;
  double kernel_max_ms = 0.0;
  double gpu_lookup_ns_per_key = 0.0;
  double slot_compute_ns_per_key = 0.0;
  double slot_gather_ns_per_key = 0.0;
  double d2h_ms = 0.0;
  double verify_ms = 0.0;
  std::uint32_t index_min = 0;
  std::uint32_t index_max = 0;
  std::size_t index_span = 0;
};

void print_usage(char const* argv0)
{
  std::cout << "Usage: " << argv0
            << " [--keys-file <path>] [--probe-keys-file <path>] [--max-keys <N>] [--max-probe-keys <N>] [--hash <name>]\n"
               "       [--items-per-thread <N>] [--threads <N>] [--iterations <N>]\n"
               "       [--warmup <N>] [--device <ordinal>] [--compile-mode <ptx|lto>]\n"
               "       [--lookup-mode <direct|split|warpcache|blocksort>]\n"
               "       [--table-load-mode <generic|readonly>]\n"
               "       [--cpu-backend <none|rawdog-jit|llvm-jit|auto>]\n"
               "       [--cpu-vector-width <1|2|4|8|16>] [--cpu-strict-vector-width <0|1>]\n"
               "       [--analyze-slot-reuse] [--analysis-only]\n"
               "       [--dump-fragment] [--source-out <path>] [--embed-table-data]\n"
               "       [--csv] [--csv-header]\n"
               "       [--no-verify]\n";
  std::cout << "Hash names: multiplyshiftr, multiplyshiftrx, mulshrolate1rx,\n";
  std::cout << "            mulshrolate2rx, mulshrolate3rx, mulshrolate4rx\n";
}

double elapsed_ms(steady_clock::time_point start, steady_clock::time_point end)
{
  return std::chrono::duration<double, std::milli>(end - start).count();
}

std::string format_bytes(std::size_t bytes)
{
  static constexpr char const* suffixes[] = {"B", "KiB", "MiB", "GiB", "TiB"};
  double value = static_cast<double>(bytes);
  std::size_t suffix_index = 0;
  while (value >= 1024.0 && suffix_index < (std::size(suffixes) - 1)) {
    value /= 1024.0;
    ++suffix_index;
  }

  std::ostringstream stream;
  stream << std::fixed << std::setprecision(suffix_index == 0 ? 0 : 2)
         << value << suffixes[suffix_index];
  return stream.str();
}

std::string csv_escape(std::string const& value)
{
  std::string escaped = "\"";
  escaped.reserve(value.size() + 2);
  for (char c : value) {
    if (c == '"') { escaped += "\"\""; }
    else { escaped.push_back(c); }
  }
  escaped.push_back('"');
  return escaped;
}

std::string cu_result_to_string(CUresult result)
{
  char const* name = nullptr;
  char const* desc = nullptr;
  cuGetErrorName(result, &name);
  cuGetErrorString(result, &desc);
  std::ostringstream stream;
  stream << (name ? name : "CUDA_ERROR") << ": " << (desc ? desc : "unknown");
  return stream.str();
}

std::string nvrtc_result_to_string(nvrtcResult result)
{
  return nvrtcGetErrorString(result);
}

std::string nvjitlink_result_to_string(nvJitLinkResult result)
{
  std::ostringstream stream;
  stream << "nvJitLink(" << static_cast<int>(result) << ")";
  return stream.str();
}

void throw_if_bad(CUresult result, char const* what)
{
  if (result != CUDA_SUCCESS) {
    throw std::runtime_error(std::string(what) + " failed: " +
                             cu_result_to_string(result));
  }
}

void throw_if_bad(cudaError_t result, char const* what)
{
  if (result != cudaSuccess) {
    throw std::runtime_error(std::string(what) + " failed: " +
                             cudaGetErrorString(result));
  }
}

void throw_if_bad(std::int32_t hr, char const* what)
{
  if (!succeeded(hr)) {
    throw std::runtime_error(std::string(what) + " failed: " +
                             to_hex(static_cast<std::uint32_t>(hr)));
  }
}

std::string get_nvrtc_log(nvrtcProgram program)
{
  std::size_t log_size = 0;
  if (nvrtcGetProgramLogSize(program, &log_size) != NVRTC_SUCCESS || log_size == 0) {
    return {};
  }
  std::string log(log_size, '\0');
  if (nvrtcGetProgramLog(program, log.data()) != NVRTC_SUCCESS) { return {}; }
  return log;
}

std::string get_nvjitlink_log(nvJitLinkHandle handle, bool error_log)
{
  std::size_t log_size = 0;
  nvJitLinkResult result =
    error_log ? nvJitLinkGetErrorLogSize(handle, &log_size)
              : nvJitLinkGetInfoLogSize(handle, &log_size);
  if (result != NVJITLINK_SUCCESS || log_size == 0) { return {}; }
  std::string log(log_size, '\0');
  result = error_log ? nvJitLinkGetErrorLog(handle, log.data())
                     : nvJitLinkGetInfoLog(handle, log.data());
  if (result != NVJITLINK_SUCCESS) { return {}; }
  return log;
}

std::vector<std::uint64_t> make_sample_keys()
{
  std::vector<std::uint64_t> keys;
  keys.reserve(64);
  constexpr std::uint64_t high_bits = (1ULL << 40);
  for (std::uint64_t key : {1ull,  3ull,  5ull,  7ull,  11ull, 13ull, 17ull, 19ull,
                            23ull, 29ull, 31ull, 37ull, 41ull, 43ull, 47ull, 53ull,
                            59ull, 61ull, 67ull, 71ull, 73ull, 79ull, 83ull, 89ull,
                            97ull, 101ull, 103ull, 107ull, 109ull, 113ull, 127ull, 131ull,
                            137ull, 139ull, 149ull, 151ull, 157ull, 163ull, 167ull, 173ull,
                            179ull, 181ull, 191ull, 193ull, 197ull, 199ull, 211ull, 223ull,
                            227ull, 229ull, 233ull, 239ull, 241ull, 251ull, 257ull, 263ull,
                            269ull, 271ull, 277ull, 281ull, 283ull, 293ull, 307ull, 311ull}) {
    keys.push_back(high_bits | key);
  }
  return keys;
}

std::vector<std::uint64_t> load_keys_file(std::string const& path, std::uint64_t max_keys)
{
  std::ifstream input(path, std::ios::binary | std::ios::ate);
  if (!input) {
    throw std::runtime_error("Unable to open keys file: " + path);
  }

  auto const size = input.tellg();
  if (size < 0) {
    throw std::runtime_error("Unable to determine keys file size: " + path);
  }
  if ((static_cast<std::uint64_t>(size) % sizeof(std::uint64_t)) != 0) {
    throw std::runtime_error("Keys file size is not a multiple of 8 bytes: " + path);
  }

  auto const available_keys =
    static_cast<std::uint64_t>(size) / static_cast<std::uint64_t>(sizeof(std::uint64_t));
  auto const keys_to_read =
    (max_keys != 0 && max_keys < available_keys) ? max_keys : available_keys;

  std::vector<std::uint64_t> keys(static_cast<std::size_t>(keys_to_read));
  input.seekg(0, std::ios::beg);
  input.read(reinterpret_cast<char*>(keys.data()),
             static_cast<std::streamsize>(keys.size() * sizeof(std::uint64_t)));
  if (!input) {
    throw std::runtime_error("Unable to read keys file contents: " + path);
  }

  return keys;
}

double ns_per_key(double milliseconds, std::size_t key_count)
{
  if (key_count == 0) { return 0.0; }
  return (milliseconds * 1'000'000.0) / static_cast<double>(key_count);
}

std::size_t current_rss_bytes()
{
#if defined(__linux__)
  long page_size = sysconf(_SC_PAGESIZE);
  std::ifstream input("/proc/self/statm");
  std::size_t total_pages = 0;
  std::size_t resident_pages = 0;
  if (input >> total_pages >> resident_pages) {
    return resident_pages * static_cast<std::size_t>(page_size);
  }
#endif
  return 0;
}

std::size_t peak_rss_bytes()
{
#if !defined(_WIN32)
  struct rusage usage {};
  if (getrusage(RUSAGE_SELF, &usage) == 0) {
#if defined(__APPLE__)
    return static_cast<std::size_t>(usage.ru_maxrss);
#else
    return static_cast<std::size_t>(usage.ru_maxrss) * 1024ull;
#endif
  }
#endif
  return 0;
}

std::uint64_t extract_bits64(std::uint64_t value, std::uint64_t bitmap)
{
  std::uint64_t result = 0;
  std::uint64_t out_bit = 0;
  while (bitmap != 0) {
    auto const lsb = bitmap & (~bitmap + 1);
    if ((value & lsb) != 0) {
      result |= (1ULL << out_bit);
    }
    bitmap ^= lsb;
    ++out_bit;
  }
  return result;
}

std::uint64_t mask_from_bits(std::uint32_t bits)
{
  return (bits >= 64u) ? std::numeric_limits<std::uint64_t>::max()
                       : ((1ULL << bits) - 1ULL);
}

std::uint32_t downsize_key64(std::uint64_t key, PH_ONLINE_JIT_TABLE_INFO const& info)
{
  auto const original_key_mask = mask_from_bits(info.OriginalKeySizeInBytes * 8u);
  auto const key_mask = mask_from_bits(info.KeySizeInBytes * 8u);
  key &= original_key_mask;
  if (info.DownsizeBitmap != 0) {
    return static_cast<std::uint32_t>(extract_bits64(key, info.DownsizeBitmap) & key_mask);
  }
  return static_cast<std::uint32_t>(key & key_mask);
}

std::vector<std::uint32_t> make_downsized_keys32(std::vector<std::uint64_t> const& keys,
                                                 PH_ONLINE_JIT_TABLE_INFO const& info)
{
  std::vector<std::uint32_t> downsized;
  downsized.reserve(keys.size());
  for (auto key : keys) {
    downsized.push_back(downsize_key64(key, info));
  }
  return downsized;
}

std::uint32_t rotr32_host(std::uint32_t value, std::uint32_t shift)
{
  shift &= 31u;
  if (shift == 0u) { return value; }
  return (value >> shift) | (value << (32u - shift));
}

struct host_slot_pair {
  std::uint32_t first = 0;
  std::uint32_t second = 0;
};

std::uint32_t table_value_from_host_data(void const* table_data,
                                         PH_ONLINE_JIT_TABLE_INFO const& info,
                                         std::uint32_t index)
{
  if (info.AssignedElementSizeInBytes == sizeof(std::uint16_t)) {
    auto const* values = static_cast<std::uint16_t const*>(table_data);
    return static_cast<std::uint32_t>(values[index]);
  }
  auto const* values = static_cast<std::uint32_t const*>(table_data);
  return values[index];
}

host_slot_pair slot_pair_from_key_host(std::uint64_t key,
                                       PH_ONLINE_JIT_HASH_FUNCTION hash_function,
                                       PH_ONLINE_JIT_TABLE_INFO const& info)
{
  auto const downsized = downsize_key64(key, info);
  auto const seed1 = static_cast<std::uint64_t>(info.Seed1);
  auto const seed2 = static_cast<std::uint64_t>(info.Seed2);
  auto const seed4 = static_cast<std::uint64_t>(info.Seed4);
  auto const seed5 = static_cast<std::uint64_t>(info.Seed5);
  auto const seed3_byte1 = (info.Seed3 & 0xffu);
  auto const seed3_byte2 = ((info.Seed3 >> 8u) & 0xffu);
  auto const seed3_byte3 = ((info.Seed3 >> 16u) & 0xffu);
  auto const seed3_byte4 = ((info.Seed3 >> 24u) & 0xffu);
  auto const use_32bit_math = (info.KeySizeInBytes <= sizeof(std::uint32_t));

  if (use_32bit_math) {
    auto const downsized32 = static_cast<std::uint32_t>(downsized);
    auto const seed1_32 = static_cast<std::uint32_t>(seed1);
    auto const seed2_32 = static_cast<std::uint32_t>(seed2);
    auto const seed4_32 = static_cast<std::uint32_t>(seed4);
    auto const seed5_32 = static_cast<std::uint32_t>(seed5);

    switch (hash_function) {
      case PhOnlineJitHashMultiplyShiftR: {
        auto const vertex1 = (downsized32 * seed1_32) >> seed3_byte1;
        auto const vertex2 = (downsized32 * seed2_32) >> seed3_byte2;
        return {static_cast<std::uint32_t>(vertex1 & info.HashMask),
                static_cast<std::uint32_t>(vertex2 & info.HashMask)};
      }
      case PhOnlineJitHashMultiplyShiftLR: {
        auto const vertex1 = (downsized32 * seed1_32) << seed3_byte1;
        auto const vertex2 = (downsized32 * seed2_32) >> seed3_byte2;
        return {static_cast<std::uint32_t>(vertex1 & info.HashMask),
                static_cast<std::uint32_t>(vertex2 & info.HashMask)};
      }
      case PhOnlineJitHashMultiplyShiftRMultiply: {
        auto const vertex1 = ((downsized32 * seed1_32) >> seed3_byte1) * seed2_32;
        auto const vertex2 = ((downsized32 * seed4_32) >> seed3_byte2) * seed5_32;
        return {static_cast<std::uint32_t>(vertex1 & info.HashMask),
                static_cast<std::uint32_t>(vertex2 & info.HashMask)};
      }
      case PhOnlineJitHashMultiplyShiftR2: {
        auto const vertex1 = (((downsized32 * seed1_32) >> seed3_byte1) * seed2_32) >> seed3_byte2;
        auto const vertex2 = (((downsized32 * seed4_32) >> seed3_byte3) * seed5_32) >> seed3_byte4;
        return {static_cast<std::uint32_t>(vertex1 & info.HashMask),
                static_cast<std::uint32_t>(vertex2 & info.HashMask)};
      }
      case PhOnlineJitHashMultiplyShiftRX: {
        auto const vertex1 = (downsized32 * seed1_32) >> seed3_byte1;
        auto const vertex2 = (downsized32 * seed2_32) >> seed3_byte1;
        return {vertex1, vertex2};
      }
      case PhOnlineJitHashMulshrolate1RX: {
        auto vertex1 = rotr32_host(downsized32 * seed1_32, seed3_byte2);
        vertex1 >>= seed3_byte1;
        auto vertex2 = downsized32 * seed2_32;
        vertex2 >>= seed3_byte1;
        return {vertex1, vertex2};
      }
      case PhOnlineJitHashMulshrolate2RX: {
        auto vertex1 = rotr32_host(downsized32 * seed1_32, seed3_byte2);
        vertex1 >>= seed3_byte1;
        auto vertex2 = rotr32_host(downsized32 * seed2_32, seed3_byte3);
        vertex2 >>= seed3_byte1;
        return {vertex1, vertex2};
      }
      case PhOnlineJitHashMulshrolate3RX: {
        auto vertex1 = rotr32_host(downsized32 * seed1_32, seed3_byte2);
        vertex1 = vertex1 * seed4_32;
        vertex1 >>= seed3_byte1;
        auto vertex2 = rotr32_host(downsized32 * seed2_32, seed3_byte3);
        vertex2 >>= seed3_byte1;
        return {vertex1, vertex2};
      }
      case PhOnlineJitHashMulshrolate4RX: {
        auto vertex1 = rotr32_host(downsized32 * seed1_32, seed3_byte2);
        vertex1 = vertex1 * seed4_32;
        vertex1 >>= seed3_byte1;
        auto vertex2 = rotr32_host(downsized32 * seed2_32, seed3_byte3);
        vertex2 = vertex2 * seed5_32;
        vertex2 >>= seed3_byte1;
        return {vertex1, vertex2};
      }
    }
  }

  auto const downsized64 = static_cast<std::uint64_t>(downsized);

  switch (hash_function) {
    case PhOnlineJitHashMultiplyShiftR: {
      auto vertex1 = (downsized64 * seed1) >> seed3_byte1;
      auto vertex2 = (downsized64 * seed2) >> seed3_byte2;
      return {static_cast<std::uint32_t>(vertex1 & info.HashMask),
              static_cast<std::uint32_t>(vertex2 & info.HashMask)};
    }
    case PhOnlineJitHashMultiplyShiftLR: {
      auto vertex1 = (downsized64 * seed1) << seed3_byte1;
      auto vertex2 = (downsized64 * seed2) >> seed3_byte2;
      return {static_cast<std::uint32_t>(vertex1 & info.HashMask),
              static_cast<std::uint32_t>(vertex2 & info.HashMask)};
    }
    case PhOnlineJitHashMultiplyShiftRMultiply: {
      auto vertex1 = ((downsized64 * seed1) >> seed3_byte1) * seed2;
      auto vertex2 = ((downsized64 * seed4) >> seed3_byte2) * seed5;
      return {static_cast<std::uint32_t>(vertex1 & info.HashMask),
              static_cast<std::uint32_t>(vertex2 & info.HashMask)};
    }
    case PhOnlineJitHashMultiplyShiftR2: {
      auto vertex1 = (((downsized64 * seed1) >> seed3_byte1) * seed2) >> seed3_byte2;
      auto vertex2 = (((downsized64 * seed4) >> seed3_byte3) * seed5) >> seed3_byte4;
      return {static_cast<std::uint32_t>(vertex1 & info.HashMask),
              static_cast<std::uint32_t>(vertex2 & info.HashMask)};
    }
    case PhOnlineJitHashMultiplyShiftRX: {
      auto vertex1 = (downsized64 * seed1) >> seed3_byte1;
      auto vertex2 = (downsized64 * seed2) >> seed3_byte1;
      return {static_cast<std::uint32_t>(vertex1),
              static_cast<std::uint32_t>(vertex2)};
    }
    case PhOnlineJitHashMulshrolate1RX: {
      auto downsized32 = static_cast<std::uint32_t>(downsized64);
      auto vertex1 = rotr32_host(downsized32 * static_cast<std::uint32_t>(seed1), seed3_byte2);
      vertex1 >>= seed3_byte1;
      auto vertex2 = downsized32 * static_cast<std::uint32_t>(seed2);
      vertex2 >>= seed3_byte1;
      return {vertex1, vertex2};
    }
    case PhOnlineJitHashMulshrolate2RX: {
      auto downsized32 = static_cast<std::uint32_t>(downsized);
      auto vertex1 = rotr32_host(downsized32 * static_cast<std::uint32_t>(seed1), seed3_byte2);
      vertex1 >>= seed3_byte1;
      auto vertex2 = rotr32_host(downsized32 * static_cast<std::uint32_t>(seed2), seed3_byte3);
      vertex2 >>= seed3_byte1;
      return {vertex1, vertex2};
    }
    case PhOnlineJitHashMulshrolate3RX: {
      auto downsized32 = static_cast<std::uint32_t>(downsized);
      auto vertex1 = rotr32_host(downsized32 * static_cast<std::uint32_t>(seed1), seed3_byte2);
      vertex1 = vertex1 * static_cast<std::uint32_t>(seed4);
      vertex1 >>= seed3_byte1;
      auto vertex2 = rotr32_host(downsized32 * static_cast<std::uint32_t>(seed2), seed3_byte3);
      vertex2 >>= seed3_byte1;
      return {vertex1, vertex2};
    }
    case PhOnlineJitHashMulshrolate4RX: {
      auto downsized32 = static_cast<std::uint32_t>(downsized);
      auto vertex1 = rotr32_host(downsized32 * static_cast<std::uint32_t>(seed1), seed3_byte2);
      vertex1 = vertex1 * static_cast<std::uint32_t>(seed4);
      vertex1 >>= seed3_byte1;
      auto vertex2 = rotr32_host(downsized32 * static_cast<std::uint32_t>(seed2), seed3_byte3);
      vertex2 = vertex2 * static_cast<std::uint32_t>(seed5);
      vertex2 >>= seed3_byte1;
      return {vertex1, vertex2};
    }
  }

  throw std::runtime_error("Unsupported hash function for host slot analysis");
}

std::uint32_t index_from_key_host(std::uint64_t key,
                                  PH_ONLINE_JIT_HASH_FUNCTION hash_function,
                                  PH_ONLINE_JIT_TABLE_INFO const& info,
                                  void const* table_data)
{
  auto const slots = slot_pair_from_key_host(key, hash_function, info);
  auto const low = table_value_from_host_data(table_data, info, slots.first);
  auto const high = table_value_from_host_data(table_data, info, slots.second);
  return static_cast<std::uint32_t>((low + high) & info.IndexMask);
}

struct tile_reuse_stats {
  double unique_requests_avg = 0.0;
  double duplicate_ratio = 0.0;
};

tile_reuse_stats analyze_slot_reuse_for_tile(std::vector<std::uint64_t> const& keys,
                                             PH_ONLINE_JIT_HASH_FUNCTION hash_function,
                                             PH_ONLINE_JIT_TABLE_INFO const& info,
                                             std::size_t threads_per_tile,
                                             std::size_t items_per_thread)
{
  tile_reuse_stats stats;
  if (keys.empty() || threads_per_tile == 0 || items_per_thread == 0) {
    return stats;
  }

  auto const keys_per_tile = threads_per_tile * items_per_thread;
  std::vector<std::uint32_t> requests;
  requests.reserve(keys_per_tile * 2u);

  std::size_t total_tiles = 0;
  std::size_t total_requests = 0;
  std::size_t total_unique = 0;

  for (std::size_t base = 0; base < keys.size(); base += keys_per_tile) {
    requests.clear();
    auto const limit = std::min(base + keys_per_tile, keys.size());
    for (std::size_t i = base; i < limit; ++i) {
      auto const slots = slot_pair_from_key_host(keys[i], hash_function, info);
      requests.push_back(slots.first);
      requests.push_back(slots.second);
    }
    std::sort(requests.begin(), requests.end());
    auto const unique_end = std::unique(requests.begin(), requests.end());
    auto const unique_count = static_cast<std::size_t>(unique_end - requests.begin());
    total_tiles += 1;
    total_requests += requests.size();
    total_unique += unique_count;
  }

  if (total_tiles != 0) {
    stats.unique_requests_avg = static_cast<double>(total_unique) /
                                static_cast<double>(total_tiles);
  }
  if (total_requests != 0) {
    stats.duplicate_ratio = 1.0 -
      (static_cast<double>(total_unique) / static_cast<double>(total_requests));
  }

  return stats;
}

std::string compose_translation_unit(std::string const& fragment,
                                     int items_per_thread,
                                     int threads_per_block,
                                     bool embed_table_data,
                                     bool validate_membership,
                                     lookup_mode mode,
                                     table_load_mode table_load)
{
  std::ostringstream source;
  source << fragment << "\n";
  source << R"(
namespace perfecthash::consumer {
namespace generated = perfecthash::generated::online_jit_table;

__device__ __forceinline__ uint32_t load_table_value(
  const generated::table_data_type* table_data,
  uint32_t index)
{
)";
  if (table_load == table_load_mode::readonly) {
    source << R"(#if defined(__CUDA_ARCH__)
  return static_cast<uint32_t>(__ldg(table_data + index));
#else
  return static_cast<uint32_t>(table_data[index]);
#endif
)";
  } else {
    source << R"(  return static_cast<uint32_t>(table_data[index]);
)";
  }
  source << R"(}

template <int ITEMS_PER_THREAD>
__device__ __forceinline__ void direct_probe_tile(
  const generated::original_key_type (&input)[ITEMS_PER_THREAD],
  const generated::table_data_type* table_data,
  const generated::original_key_type* build_keys,
  size_t build_key_count,
  uint32_t (&output)[ITEMS_PER_THREAD])
{
#pragma unroll
  for (int item = 0; item < ITEMS_PER_THREAD; ++item) {
    const generated::slot_pair_type slots = generated::slot_pair_from_key(input[item]);
    const uint32_t value_low = load_table_value(table_data, slots.first);
    const uint32_t value_high = load_table_value(table_data, slots.second);
    const uint32_t index = static_cast<uint32_t>((value_low + value_high) & generated::index_mask);
)";
  if (validate_membership) {
    source << R"(    output[item] =
      (index < build_key_count && build_keys[index] == input[item]) ? index : 0xFFFFFFFFu;
)";
  } else {
    source << R"(    output[item] = index;
)";
  }
  source << R"(  }
}

__device__ __forceinline__ uint32_t warp_cached_load(
  const generated::table_data_type* table_data,
  uint32_t slot)
{
  const unsigned active = __activemask();
  const unsigned match = __match_any_sync(active, slot);
  const int lane = static_cast<int>(threadIdx.x & 31);
  const int leader = __ffs(static_cast<int>(match)) - 1;
  uint32_t value = 0;
  if (lane == leader) {
    value = load_table_value(table_data, slot);
  }
  return __shfl_sync(active, value, leader);
}

template <int ITEMS_PER_THREAD>
__device__ __forceinline__ void warp_cached_probe_tile(
  const generated::original_key_type (&input)[ITEMS_PER_THREAD],
  const generated::table_data_type* table_data,
  uint32_t (&output)[ITEMS_PER_THREAD])
{
#pragma unroll
  for (int item = 0; item < ITEMS_PER_THREAD; ++item) {
    const generated::slot_pair_type slots = generated::slot_pair_from_key(input[item]);
    const uint32_t value_low = warp_cached_load(table_data, slots.first);
    const uint32_t value_high = warp_cached_load(table_data, slots.second);
    output[item] = static_cast<uint32_t>((value_low + value_high) & generated::index_mask);
  }
}

template <int ITEMS_PER_THREAD>
__device__ __forceinline__ void compute_slot_tile(
  const generated::original_key_type (&input)[ITEMS_PER_THREAD],
  uint32_t (&slot1)[ITEMS_PER_THREAD],
  uint32_t (&slot2)[ITEMS_PER_THREAD])
{
#pragma unroll
  for (int item = 0; item < ITEMS_PER_THREAD; ++item) {
    const generated::slot_pair_type slots = generated::slot_pair_from_key(input[item]);
    slot1[item] = slots.first;
    slot2[item] = slots.second;
  }
}

template <int ITEMS_PER_THREAD>
__device__ __forceinline__ void gather_tile(
  const uint32_t (&slot1)[ITEMS_PER_THREAD],
  const uint32_t (&slot2)[ITEMS_PER_THREAD],
  const generated::table_data_type* table_data,
  uint32_t (&output)[ITEMS_PER_THREAD])
{
#pragma unroll
  for (int item = 0; item < ITEMS_PER_THREAD; ++item) {
    const uint32_t value_low = load_table_value(table_data, slot1[item]);
    const uint32_t value_high = load_table_value(table_data, slot2[item]);
    output[item] = static_cast<uint32_t>((value_low + value_high) & generated::index_mask);
  }
}

__device__ __forceinline__ void swap_request(
  uint32_t& slot_a,
  uint32_t& slot_b,
  uint32_t& dest_a,
  uint32_t& dest_b)
{
  const uint32_t slot_tmp = slot_a;
  slot_a = slot_b;
  slot_b = slot_tmp;
  const uint32_t dest_tmp = dest_a;
  dest_a = dest_b;
  dest_b = dest_tmp;
}

}  // namespace perfecthash::consumer
)";
  if (mode == lookup_mode::direct) {
    source << R"(
extern "C" __global__ void probe_kernel(
  const perfecthash::generated::online_jit_table::original_key_type* keys,
)";
    if (validate_membership) {
      source << R"(  const perfecthash::generated::online_jit_table::original_key_type* build_keys,
  size_t build_key_count,
)";
    }
    if (!embed_table_data) {
      source << R"(  const perfecthash::generated::online_jit_table::table_data_type* table_data,
)";
    }
    source << R"(  uint32_t* out,
  size_t count)
{
  namespace generated = perfecthash::generated::online_jit_table;
  constexpr int ITEMS = )";
    source << items_per_thread;
    source << R"(;
  const size_t thread_base =
    (static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x) * ITEMS;

  generated::original_key_type local_keys[ITEMS] = {};
  uint32_t local_out[ITEMS] = {};

#pragma unroll
  for (int item = 0; item < ITEMS; ++item) {
    const size_t idx = thread_base + static_cast<size_t>(item);
    if (idx < count) { local_keys[item] = keys[idx]; }
  }

  perfecthash::consumer::direct_probe_tile<ITEMS>(local_keys, )";
    if (embed_table_data) {
      source << "generated::table_data";
    } else {
      source << "table_data";
    }
    source << ", ";
    if (validate_membership) {
      source << "build_keys, build_key_count";
    } else {
      source << "nullptr, 0";
    }
    source << R"(, local_out);

#pragma unroll
  for (int item = 0; item < ITEMS; ++item) {
    const size_t idx = thread_base + static_cast<size_t>(item);
    if (idx < count) { out[idx] = local_out[item]; }
  }
}
)";
  } else if (mode == lookup_mode::split) {
    source << R"(
extern "C" __global__ void compute_slots_kernel(
  const perfecthash::generated::online_jit_table::original_key_type* keys,
  uint32_t* slot1_out,
  uint32_t* slot2_out,
  size_t count)
{
  namespace generated = perfecthash::generated::online_jit_table;
  constexpr int ITEMS = )";
    source << items_per_thread;
    source << R"(;
  const size_t thread_base =
    (static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x) * ITEMS;

  generated::original_key_type local_keys[ITEMS] = {};
  uint32_t local_slot1[ITEMS] = {};
  uint32_t local_slot2[ITEMS] = {};

#pragma unroll
  for (int item = 0; item < ITEMS; ++item) {
    const size_t idx = thread_base + static_cast<size_t>(item);
    if (idx < count) { local_keys[item] = keys[idx]; }
  }

  perfecthash::consumer::compute_slot_tile<ITEMS>(local_keys, local_slot1, local_slot2);

#pragma unroll
  for (int item = 0; item < ITEMS; ++item) {
    const size_t idx = thread_base + static_cast<size_t>(item);
    if (idx < count) {
      slot1_out[idx] = local_slot1[item];
      slot2_out[idx] = local_slot2[item];
    }
  }
}

extern "C" __global__ void gather_kernel(
  const uint32_t* slot1_in,
  const uint32_t* slot2_in,
)";
    if (!embed_table_data) {
      source << R"(  const perfecthash::generated::online_jit_table::table_data_type* table_data,
)";
    }
    source << R"(  uint32_t* out,
  size_t count)
{
  namespace generated = perfecthash::generated::online_jit_table;
  constexpr int ITEMS = )";
    source << items_per_thread;
    source << R"(;
  const size_t thread_base =
    (static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x) * ITEMS;

  uint32_t local_slot1[ITEMS] = {};
  uint32_t local_slot2[ITEMS] = {};
  uint32_t local_out[ITEMS] = {};

#pragma unroll
  for (int item = 0; item < ITEMS; ++item) {
    const size_t idx = thread_base + static_cast<size_t>(item);
    if (idx < count) {
      local_slot1[item] = slot1_in[idx];
      local_slot2[item] = slot2_in[idx];
    }
  }

  perfecthash::consumer::gather_tile<ITEMS>(local_slot1, local_slot2, )";
    if (embed_table_data) {
      source << "generated::table_data";
    } else {
      source << "table_data";
    }
    source << R"(, local_out);

#pragma unroll
  for (int item = 0; item < ITEMS; ++item) {
    const size_t idx = thread_base + static_cast<size_t>(item);
    if (idx < count) { out[idx] = local_out[item]; }
  }
}
)";
  } else if (mode == lookup_mode::warpcache) {
    source << R"(
extern "C" __global__ void warpcache_probe_kernel(
  const perfecthash::generated::online_jit_table::original_key_type* keys,
)";
    if (!embed_table_data) {
      source << R"(  const perfecthash::generated::online_jit_table::table_data_type* table_data,
)";
    }
    source << R"(  uint32_t* out,
  size_t count)
{
  namespace generated = perfecthash::generated::online_jit_table;
  constexpr int ITEMS = )";
    source << items_per_thread;
    source << R"(;
  const size_t thread_base =
    (static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x) * ITEMS;

  generated::original_key_type local_keys[ITEMS] = {};
  uint32_t local_out[ITEMS] = {};

#pragma unroll
  for (int item = 0; item < ITEMS; ++item) {
    const size_t idx = thread_base + static_cast<size_t>(item);
    if (idx < count) { local_keys[item] = keys[idx]; }
  }

  perfecthash::consumer::warp_cached_probe_tile<ITEMS>(local_keys, )";
    if (embed_table_data) {
      source << "generated::table_data";
    } else {
      source << "table_data";
    }
    source << R"(, local_out);

#pragma unroll
  for (int item = 0; item < ITEMS; ++item) {
    const size_t idx = thread_base + static_cast<size_t>(item);
    if (idx < count) { out[idx] = local_out[item]; }
  }
}
)";
  } else {
    source << R"(
extern "C" __global__ void blocksort_probe_kernel(
  const perfecthash::generated::online_jit_table::original_key_type* keys,
)";
    if (!embed_table_data) {
      source << R"(  const perfecthash::generated::online_jit_table::table_data_type* table_data,
)";
    }
    source << R"(  uint32_t* out,
  size_t count)
{
  namespace generated = perfecthash::generated::online_jit_table;
  constexpr int ITEMS = )";
    source << items_per_thread;
    source << R"(;
  constexpr int THREADS = )";
    source << threads_per_block;
    source << R"(;
  constexpr int TOTAL_ITEMS = THREADS * ITEMS;
  constexpr int TOTAL_REQUESTS = TOTAL_ITEMS * 2;
  constexpr uint32_t INVALID_SLOT = 0xFFFFFFFFu;

  __shared__ uint32_t shared_slots[TOTAL_REQUESTS];
  __shared__ uint32_t shared_dests[TOTAL_REQUESTS];
  __shared__ uint32_t shared_partials[TOTAL_REQUESTS];

  const uint32_t tid = threadIdx.x;
  const size_t block_base = static_cast<size_t>(blockIdx.x) * static_cast<size_t>(TOTAL_ITEMS);
  const uint32_t local_base = tid * ITEMS;

#pragma unroll
  for (int item = 0; item < ITEMS; ++item) {
    const size_t global_idx = block_base + static_cast<size_t>(local_base + item);
    const uint32_t request_base = (local_base + item) * 2u;
    if (global_idx < count) {
      const generated::slot_pair_type slots = generated::slot_pair_from_key(keys[global_idx]);
      shared_slots[request_base] = slots.first;
      shared_slots[request_base + 1] = slots.second;
      shared_dests[request_base] = request_base;
      shared_dests[request_base + 1] = request_base + 1u;
    } else {
      shared_slots[request_base] = INVALID_SLOT;
      shared_slots[request_base + 1] = INVALID_SLOT;
      shared_dests[request_base] = request_base;
      shared_dests[request_base + 1] = request_base + 1u;
    }
  }
  __syncthreads();

  for (uint32_t k = 2; k <= TOTAL_REQUESTS; k <<= 1) {
    for (uint32_t j = k >> 1; j > 0; j >>= 1) {
      for (uint32_t i = tid; i < TOTAL_REQUESTS; i += THREADS) {
        const uint32_t ixj = i ^ j;
        if (ixj > i) {
          const bool ascending = ((i & k) == 0);
          const bool should_swap =
            ascending ? (shared_slots[i] > shared_slots[ixj])
                      : (shared_slots[i] < shared_slots[ixj]);
          if (should_swap) {
            perfecthash::consumer::swap_request(shared_slots[i],
                                                shared_slots[ixj],
                                                shared_dests[i],
                                                shared_dests[ixj]);
          }
        }
      }
      __syncthreads();
    }
  }

  for (uint32_t i = tid; i < TOTAL_REQUESTS; i += THREADS) {
    const uint32_t slot = shared_slots[i];
    if (slot != INVALID_SLOT) {
)";
  if (embed_table_data) {
    source << R"(      shared_partials[shared_dests[i]] =
        perfecthash::consumer::load_table_value(generated::table_data, slot);
)";
  } else {
    source << R"(      shared_partials[shared_dests[i]] =
        perfecthash::consumer::load_table_value(table_data, slot);
)";
  }
  source << R"(
    }
  }
  __syncthreads();

#pragma unroll
  for (int item = 0; item < ITEMS; ++item) {
    const uint32_t output_base = (local_base + item) * 2u;
    const size_t global_idx = block_base + static_cast<size_t>(local_base + item);
    if (global_idx < count) {
      const uint32_t value_low = shared_partials[output_base];
      const uint32_t value_high = shared_partials[output_base + 1u];
      out[global_idx] = static_cast<uint32_t>((value_low + value_high) & generated::index_mask);
    }
  }
}
)";
  }
  return source.str();
}

std::string compile_to_ptx(std::string const& source, int major, int minor)
{
  nvrtcProgram program{};
  auto create_result =
    nvrtcCreateProgram(&program, source.c_str(), "perfecthash_online_nvrtc.cu", 0, nullptr, nullptr);
  if (create_result != NVRTC_SUCCESS) {
    throw std::runtime_error("nvrtcCreateProgram failed: " +
                             nvrtc_result_to_string(create_result));
  }

  std::string arch = "--gpu-architecture=compute_" + std::to_string(major) + std::to_string(minor);
  std::string include_dir = std::string("--include-path=") + PH_CUDA_INCLUDE_DIR;

  std::vector<char const*> options{
    "--std=c++17",
    "--device-as-default-execution-space",
    arch.c_str(),
    include_dir.c_str(),
  };

  auto compile_result =
    nvrtcCompileProgram(program, static_cast<int>(options.size()), options.data());
  auto const log = get_nvrtc_log(program);
  if (!log.empty()) { std::cerr << log; }
  if (compile_result != NVRTC_SUCCESS) {
    nvrtcDestroyProgram(&program);
    throw std::runtime_error("nvrtcCompileProgram failed: " +
                             nvrtc_result_to_string(compile_result));
  }

  std::size_t ptx_size = 0;
  if (nvrtcGetPTXSize(program, &ptx_size) != NVRTC_SUCCESS) {
    nvrtcDestroyProgram(&program);
    throw std::runtime_error("nvrtcGetPTXSize failed");
  }

  std::string ptx(ptx_size, '\0');
  if (nvrtcGetPTX(program, ptx.data()) != NVRTC_SUCCESS) {
    nvrtcDestroyProgram(&program);
    throw std::runtime_error("nvrtcGetPTX failed");
  }

  nvrtcDestroyProgram(&program);
  return ptx;
}

std::vector<char> compile_to_ltoir(std::string const& source, int major, int minor)
{
  nvrtcProgram program{};
  auto create_result =
    nvrtcCreateProgram(&program, source.c_str(), "perfecthash_online_nvrtc_lto.cu", 0, nullptr, nullptr);
  if (create_result != NVRTC_SUCCESS) {
    throw std::runtime_error("nvrtcCreateProgram failed: " +
                             nvrtc_result_to_string(create_result));
  }

  std::string arch = "--gpu-architecture=compute_" + std::to_string(major) + std::to_string(minor);
  std::string include_dir = std::string("--include-path=") + PH_CUDA_INCLUDE_DIR;

  std::vector<char const*> options{
    "--std=c++17",
    "--device-as-default-execution-space",
    "--dlink-time-opt",
    "--gen-opt-lto",
    arch.c_str(),
    include_dir.c_str(),
  };

  auto compile_result =
    nvrtcCompileProgram(program, static_cast<int>(options.size()), options.data());
  auto const log = get_nvrtc_log(program);
  if (!log.empty()) { std::cerr << log; }
  if (compile_result != NVRTC_SUCCESS) {
    nvrtcDestroyProgram(&program);
    throw std::runtime_error("nvrtcCompileProgram failed: " +
                             nvrtc_result_to_string(compile_result));
  }

  std::size_t ir_size = 0;
  if (nvrtcGetLTOIRSize(program, &ir_size) != NVRTC_SUCCESS) {
    nvrtcDestroyProgram(&program);
    throw std::runtime_error("nvrtcGetLTOIRSize failed");
  }

  std::vector<char> ir(ir_size);
  if (nvrtcGetLTOIR(program, ir.data()) != NVRTC_SUCCESS) {
    nvrtcDestroyProgram(&program);
    throw std::runtime_error("nvrtcGetLTOIR failed");
  }

  nvrtcDestroyProgram(&program);
  return ir;
}

std::vector<char> link_ltoir_to_cubin(std::vector<char> const& ir, int major, int minor)
{
  nvJitLinkHandle handle{};
  std::string arch = "-arch=sm_" + std::to_string(major) + std::to_string(minor);
  std::vector<char const*> options{
    arch.c_str(),
    "-lto",
  };

  auto result = nvJitLinkCreate(&handle,
                                static_cast<uint32_t>(options.size()),
                                options.data());
  if (result != NVJITLINK_SUCCESS) {
    throw std::runtime_error("nvJitLinkCreate failed: " +
                             nvjitlink_result_to_string(result));
  }

  result = nvJitLinkAddData(handle,
                            NVJITLINK_INPUT_LTOIR,
                            ir.data(),
                            ir.size(),
                            "perfecthash_online_nvrtc_lto");
  if (result != NVJITLINK_SUCCESS) {
    auto const error_log = get_nvjitlink_log(handle, true);
    nvJitLinkDestroy(&handle);
    throw std::runtime_error("nvJitLinkAddData failed: " +
                             nvjitlink_result_to_string(result) +
                             (error_log.empty() ? std::string{} : (": " + error_log)));
  }

  result = nvJitLinkComplete(handle);
  auto const info_log = get_nvjitlink_log(handle, false);
  auto const error_log = get_nvjitlink_log(handle, true);
  if (!info_log.empty()) { std::cerr << info_log; }
  if (!error_log.empty()) { std::cerr << error_log; }
  if (result != NVJITLINK_SUCCESS) {
    nvJitLinkDestroy(&handle);
    throw std::runtime_error("nvJitLinkComplete failed: " +
                             nvjitlink_result_to_string(result));
  }

  std::size_t cubin_size = 0;
  result = nvJitLinkGetLinkedCubinSize(handle, &cubin_size);
  if (result != NVJITLINK_SUCCESS) {
    nvJitLinkDestroy(&handle);
    throw std::runtime_error("nvJitLinkGetLinkedCubinSize failed: " +
                             nvjitlink_result_to_string(result));
  }

  std::vector<char> cubin(cubin_size);
  result = nvJitLinkGetLinkedCubin(handle, cubin.data());
  nvJitLinkDestroy(&handle);
  if (result != NVJITLINK_SUCCESS) {
    throw std::runtime_error("nvJitLinkGetLinkedCubin failed: " +
                             nvjitlink_result_to_string(result));
  }

  return cubin;
}

struct ph_context_handle {
  PH_ONLINE_JIT_CONTEXT* value = nullptr;
  ~ph_context_handle()
  {
    if (value != nullptr) { PhOnlineJitClose(value); }
  }
};

struct ph_table_handle {
  PH_ONLINE_JIT_TABLE* value = nullptr;
  ~ph_table_handle()
  {
    if (value != nullptr) { PhOnlineJitReleaseTable(value); }
  }
};

struct cuda_source_handle {
  char* value = nullptr;
  ~cuda_source_handle()
  {
    if (value != nullptr) { PhOnlineJitFreeCudaSource(value); }
  }
};

struct table_data_handle {
  void* value = nullptr;
  ~table_data_handle()
  {
    if (value != nullptr) { PhOnlineJitFreeCudaTableData(value); }
  }
};

struct primary_context_handle {
  CUdevice device{};
  bool retained = false;
  ~primary_context_handle()
  {
    if (retained) { cuDevicePrimaryCtxRelease(device); }
  }
};

struct module_handle {
  CUmodule value{};
  ~module_handle()
  {
    if (value != nullptr) { cuModuleUnload(value); }
  }
};

struct device_memory {
  CUdeviceptr value{};
  ~device_memory()
  {
    if (value != 0) { cuMemFree(value); }
  }
};

struct event_handle {
  CUevent value{};
  ~event_handle()
  {
    if (value != nullptr) { cuEventDestroy(value); }
  }
};

std::vector<double> run_probe_iterations(CUfunction function,
                                         void** params,
                                         int blocks,
                                         int threads,
                                         int warmup,
                                         int iterations)
{
  event_handle start_event;
  event_handle stop_event;
  throw_if_bad(cuEventCreate(&start_event.value, CU_EVENT_DEFAULT), "cuEventCreate(start)");
  throw_if_bad(cuEventCreate(&stop_event.value, CU_EVENT_DEFAULT), "cuEventCreate(stop)");

  for (int iteration = 0; iteration < warmup; ++iteration) {
    throw_if_bad(cuLaunchKernel(function,
                                static_cast<unsigned>(blocks),
                                1,
                                1,
                                static_cast<unsigned>(threads),
                                1,
                                1,
                                0,
                                0,
                                params,
                                nullptr),
                 "cuLaunchKernel(warmup)");
    throw_if_bad(cuCtxSynchronize(), "cuCtxSynchronize(warmup)");
  }

  std::vector<double> timings;
  timings.reserve(static_cast<std::size_t>(iterations));
  for (int iteration = 0; iteration < iterations; ++iteration) {
    throw_if_bad(cuEventRecord(start_event.value, 0), "cuEventRecord(start)");
    throw_if_bad(cuLaunchKernel(function,
                                static_cast<unsigned>(blocks),
                                1,
                                1,
                                static_cast<unsigned>(threads),
                                1,
                                1,
                                0,
                                0,
                                params,
                                nullptr),
                 "cuLaunchKernel");
    throw_if_bad(cuEventRecord(stop_event.value, 0), "cuEventRecord(stop)");
    throw_if_bad(cuEventSynchronize(stop_event.value), "cuEventSynchronize(stop)");

    float kernel_ms = 0.0f;
    throw_if_bad(cuEventElapsedTime(&kernel_ms, start_event.value, stop_event.value),
                 "cuEventElapsedTime");
    timings.push_back(static_cast<double>(kernel_ms));
  }

  return timings;
}

std::vector<double> run_split_probe_iterations(CUfunction compute_function,
                                               void** compute_params,
                                               CUfunction gather_function,
                                               void** gather_params,
                                               int blocks,
                                               int threads,
                                               int warmup,
                                               int iterations)
{
  event_handle start_event;
  event_handle stop_event;
  throw_if_bad(cuEventCreate(&start_event.value, CU_EVENT_DEFAULT), "cuEventCreate(start)");
  throw_if_bad(cuEventCreate(&stop_event.value, CU_EVENT_DEFAULT), "cuEventCreate(stop)");

  for (int iteration = 0; iteration < warmup; ++iteration) {
    throw_if_bad(cuLaunchKernel(compute_function,
                                static_cast<unsigned>(blocks),
                                1,
                                1,
                                static_cast<unsigned>(threads),
                                1,
                                1,
                                0,
                                0,
                                compute_params,
                                nullptr),
                 "cuLaunchKernel(split warmup compute)");
    throw_if_bad(cuLaunchKernel(gather_function,
                                static_cast<unsigned>(blocks),
                                1,
                                1,
                                static_cast<unsigned>(threads),
                                1,
                                1,
                                0,
                                0,
                                gather_params,
                                nullptr),
                 "cuLaunchKernel(split warmup gather)");
    throw_if_bad(cuCtxSynchronize(), "cuCtxSynchronize(split warmup)");
  }

  std::vector<double> timings;
  timings.reserve(static_cast<std::size_t>(iterations));
  for (int iteration = 0; iteration < iterations; ++iteration) {
    throw_if_bad(cuEventRecord(start_event.value, 0), "cuEventRecord(split start)");
    throw_if_bad(cuLaunchKernel(compute_function,
                                static_cast<unsigned>(blocks),
                                1,
                                1,
                                static_cast<unsigned>(threads),
                                1,
                                1,
                                0,
                                0,
                                compute_params,
                                nullptr),
                 "cuLaunchKernel(split compute)");
    throw_if_bad(cuLaunchKernel(gather_function,
                                static_cast<unsigned>(blocks),
                                1,
                                1,
                                static_cast<unsigned>(threads),
                                1,
                                1,
                                0,
                                0,
                                gather_params,
                                nullptr),
                 "cuLaunchKernel(split gather)");
    throw_if_bad(cuEventRecord(stop_event.value, 0), "cuEventRecord(split stop)");
    throw_if_bad(cuEventSynchronize(stop_event.value), "cuEventSynchronize(split stop)");

    float kernel_ms = 0.0f;
    throw_if_bad(cuEventElapsedTime(&kernel_ms, start_event.value, stop_event.value),
                 "cuEventElapsedTime(split)");
    timings.push_back(static_cast<double>(kernel_ms));
  }

  return timings;
}

struct timing_stats {
  double avg = 0.0;
  double min = 0.0;
  double max = 0.0;
};

timing_stats summarize_timings(std::vector<double> const& timings)
{
  timing_stats stats;
  if (timings.empty()) { return stats; }

  auto const [min_it, max_it] = std::minmax_element(timings.begin(), timings.end());
  double total = 0.0;
  for (double value : timings) { total += value; }
  stats.avg = total / static_cast<double>(timings.size());
  stats.min = *min_it;
  stats.max = *max_it;
  return stats;
}

void run_cpu_bulk_index_once(PH_ONLINE_JIT_TABLE* table,
                             std::vector<std::uint32_t> const& keys32,
                             int vector_width,
                             std::vector<std::uint32_t>* indexes)
{
  if (indexes == nullptr) {
    throw std::runtime_error("indexes output cannot be null");
  }
  indexes->resize(keys32.size());

  std::size_t index = 0;
  auto* output = indexes->data();
  auto const* input = keys32.data();

  auto call_scalar = [&](std::size_t i) {
    throw_if_bad(PhOnlineJitIndex32(table, input[i], &output[i]), "PhOnlineJitIndex32");
  };

  if (vector_width == 16) {
    for (; index + 16 <= keys32.size(); index += 16) {
      throw_if_bad(PhOnlineJitIndex32x16(table, input + index, output + index),
                   "PhOnlineJitIndex32x16");
    }
  } else if (vector_width == 8) {
    for (; index + 8 <= keys32.size(); index += 8) {
      throw_if_bad(PhOnlineJitIndex32x8(table, input + index, output + index),
                   "PhOnlineJitIndex32x8");
    }
  } else if (vector_width == 4) {
    for (; index + 4 <= keys32.size(); index += 4) {
      throw_if_bad(PhOnlineJitIndex32x4(table, input + index, output + index),
                   "PhOnlineJitIndex32x4");
    }
  } else if (vector_width == 2) {
    for (; index + 2 <= keys32.size(); index += 2) {
      throw_if_bad(PhOnlineJitIndex32x2(table, input + index, output + index),
                   "PhOnlineJitIndex32x2");
    }
  }

  for (; index < keys32.size(); ++index) {
    call_scalar(index);
  }
}

timing_stats benchmark_cpu_lookup(PH_ONLINE_JIT_TABLE* table,
                                  std::vector<std::uint32_t> const& keys32,
                                  int vector_width,
                                  int warmup,
                                  int iterations,
                                  std::vector<std::uint32_t>* last_indexes)
{
  for (int iteration = 0; iteration < warmup; ++iteration) {
    run_cpu_bulk_index_once(table, keys32, vector_width, last_indexes);
  }

  std::vector<double> timings;
  timings.reserve(static_cast<std::size_t>(iterations));
  for (int iteration = 0; iteration < iterations; ++iteration) {
    auto const start = steady_clock::now();
    run_cpu_bulk_index_once(table, keys32, vector_width, last_indexes);
    auto const stop = steady_clock::now();
    timings.push_back(elapsed_ms(start, stop));
  }

  return summarize_timings(timings);
}

void verify_indexes(std::vector<std::uint32_t> const& indexes,
                    std::vector<std::uint64_t> const& build_keys,
                    std::vector<std::uint64_t> const& probe_keys,
                    benchmark_result* result)
{
  if (indexes.empty()) { return; }
  if (build_keys.size() != probe_keys.size() || indexes.size() != probe_keys.size()) {
    throw std::runtime_error("Verification inputs must have matching sizes");
  }

  auto const [min_it, max_it] = std::minmax_element(indexes.begin(), indexes.end());
  result->index_min = *min_it;
  result->index_max = *max_it;
  result->index_span = static_cast<std::size_t>(result->index_max) + 1;

  std::vector<std::uint8_t> seen(result->index_span, 0);
  for (std::size_t i = 0; i < indexes.size(); ++i) {
    auto const index = indexes[i];
    if (static_cast<std::size_t>(index) >= build_keys.size()) {
      throw std::runtime_error("Out-of-range index detected at position " + std::to_string(i));
    }
    if (build_keys[static_cast<std::size_t>(index)] != probe_keys[i]) {
      throw std::runtime_error("Key mismatch detected at position " + std::to_string(i));
    }
    auto& slot = seen[static_cast<std::size_t>(index)];
    if (slot != 0) {
      throw std::runtime_error("Duplicate index detected: " + std::to_string(index));
    }
    slot = 1;
  }
}

void print_csv_header()
{
  std::cout
    << "key_source,key_count,key_bytes,probe_source,probe_key_count,probe_key_bytes,table_data_bytes,table_data_elements,"
       "table_data_element_size,embed_table_data,hash,mode,lookup_mode,table_load_mode,device_ordinal,device_name,"
       "arch,items_per_thread,threads,blocks,warmup,iterations,cpu_backend_req,"
       "cpu_backend_eff,cpu_vector_req,cpu_vector_eff,cpu_strict,cpu_key_mode,"
       "cpu_compile_hr,cpu_compile_ms,cpu_lookup_avg_ms,cpu_lookup_min_ms,"
       "cpu_lookup_max_ms,cpu_lookup_ns_per_key,fragment_bytes,combined_bytes,image_bytes,"
       "host_rss_before_build_bytes,host_rss_after_build_bytes,host_rss_delta_bytes,host_peak_rss_bytes,"
       "vram_explicit_bytes,vram_total_bytes,build_ms,table_export_ms,emit_ms,compose_ms,compile_ms,link_ms,module_load_ms,alloc_ms,h2d_ms,table_h2d_ms,"
       "slot_compute_avg_ms,slot_gather_avg_ms,kernel_avg_ms,kernel_min_ms,kernel_max_ms,d2h_ms,verify_ms,index_min,"
       "index_max,index_span,slot_compute_ns_per_key,slot_gather_ns_per_key,gpu_lookup_ns_per_key,"
       "warp_unique_requests_avg,warp_duplicate_ratio,block_unique_requests_avg,block_duplicate_ratio\n";
}

void print_csv_row(benchmark_result const& result)
{
  std::cout << csv_escape(result.key_source) << ','
            << result.key_count << ','
            << result.key_bytes << ','
            << csv_escape(result.probe_source) << ','
            << result.probe_key_count << ','
            << result.probe_key_bytes << ','
            << result.table_data_bytes << ','
            << result.table_data_elements << ','
            << result.table_data_element_size << ','
            << (result.embedded_table_data ? 1 : 0) << ','
            << csv_escape(result.hash_name) << ','
            << csv_escape(compile_mode_to_string(result.mode)) << ','
            << csv_escape(lookup_mode_to_string(result.lookup)) << ','
            << csv_escape(table_load_mode_to_string(result.table_load)) << ','
            << result.device_ordinal << ','
            << csv_escape(result.device_name) << ','
            << csv_escape("compute_" + std::to_string(result.major) +
                          std::to_string(result.minor)) << ','
            << result.items_per_thread << ','
            << result.threads << ','
            << result.blocks << ','
            << result.warmup << ','
            << result.iterations << ','
            << csv_escape(result.cpu_backend_requested) << ','
            << csv_escape(result.cpu_backend_effective) << ','
            << result.cpu_vector_requested << ','
            << result.cpu_vector_effective << ','
            << (result.cpu_strict_vector_width ? 1 : 0) << ','
            << csv_escape(result.cpu_key_mode) << ','
            << csv_escape(to_hex(static_cast<std::uint32_t>(result.cpu_compile_hr))) << ','
            << result.cpu_compile_ms << ','
            << result.cpu_lookup_avg_ms << ','
            << result.cpu_lookup_min_ms << ','
            << result.cpu_lookup_max_ms << ','
            << result.cpu_lookup_ns_per_key << ','
            << result.fragment_bytes << ','
            << result.combined_bytes << ','
            << result.image_bytes << ','
            << result.host_rss_before_build_bytes << ','
            << result.host_rss_after_build_bytes << ','
            << result.host_rss_delta_bytes << ','
            << result.host_peak_rss_bytes << ','
            << result.vram_explicit_bytes << ','
            << result.vram_total_bytes << ','
            << std::fixed << std::setprecision(3)
            << result.build_ms << ','
            << result.table_export_ms << ','
            << result.emit_ms << ','
            << result.compose_ms << ','
            << result.compile_ms << ','
            << result.link_ms << ','
            << result.module_load_ms << ','
            << result.alloc_ms << ','
            << result.h2d_ms << ','
            << result.table_h2d_ms << ','
            << result.slot_compute_avg_ms << ','
            << result.slot_gather_avg_ms << ','
            << result.kernel_avg_ms << ','
            << result.kernel_min_ms << ','
            << result.kernel_max_ms << ','
            << result.d2h_ms << ','
            << result.verify_ms << ','
            << result.index_min << ','
            << result.index_max << ','
            << result.index_span << ','
            << result.slot_compute_ns_per_key << ','
            << result.slot_gather_ns_per_key << ','
            << result.gpu_lookup_ns_per_key << ','
            << result.warp_unique_requests_avg << ','
            << result.warp_duplicate_ratio << ','
            << result.block_unique_requests_avg << ','
            << result.block_duplicate_ratio << '\n';
}

void print_human_summary(benchmark_result const& result)
{
  std::cout << "Keys: " << result.key_count << " (" << format_bytes(result.key_bytes)
            << ") from " << result.key_source << "\n";
  std::cout << "Probe: " << result.probe_key_count << " (" << format_bytes(result.probe_key_bytes)
            << ") from " << result.probe_source << "\n";
  std::cout << "Table data: " << result.table_data_elements
            << " elements x " << result.table_data_element_size
            << " bytes = " << format_bytes(result.table_data_bytes)
            << ", embedded=" << (result.embedded_table_data ? "yes" : "no") << "\n";
  std::cout << "Config: hash=" << result.hash_name
            << ", mode=" << compile_mode_to_string(result.mode)
            << ", lookup=" << lookup_mode_to_string(result.lookup)
            << ", table-load=" << table_load_mode_to_string(result.table_load)
            << ", device=" << result.device_ordinal << " (" << result.device_name << ")"
            << ", arch=compute_" << result.major << result.minor
            << ", items/thread=" << result.items_per_thread
            << ", threads/block=" << result.threads
            << ", blocks=" << result.blocks
            << ", warmup=" << result.warmup
            << ", iterations=" << result.iterations << "\n";
  if (result.cpu_enabled) {
    std::cout << "CPU: backend=" << result.cpu_backend_requested
              << " -> " << result.cpu_backend_effective
              << ", vector=" << result.cpu_vector_requested
              << " -> " << result.cpu_vector_effective
              << ", strict=" << (result.cpu_strict_vector_width ? 1 : 0)
              << ", key-mode=" << result.cpu_key_mode << "\n";
  }
  std::cout << "Sizes: fragment=" << result.fragment_bytes
            << " bytes, combined=" << result.combined_bytes
            << " bytes, image=" << result.image_bytes << " bytes\n";
  std::cout << "Memory: host_rss_before_build=" << result.host_rss_before_build_bytes
            << " bytes, host_rss_after_build=" << result.host_rss_after_build_bytes
            << " bytes, host_rss_delta=" << result.host_rss_delta_bytes
            << " bytes, host_peak_rss=" << result.host_peak_rss_bytes
            << " bytes, vram_explicit=" << result.vram_explicit_bytes
            << " bytes, vram_total=" << result.vram_total_bytes << " bytes\n";
  std::cout << std::fixed << std::setprecision(3)
            << "Timings (ms): build=" << result.build_ms
            << ", table_export=" << result.table_export_ms
            << ", emit=" << result.emit_ms
            << ", compose=" << result.compose_ms
            << ", cpu_compile=" << result.cpu_compile_ms
            << ", cpu_lookup_avg=" << result.cpu_lookup_avg_ms
            << ", compile=" << result.compile_ms
            << ", link=" << result.link_ms
            << ", module_load=" << result.module_load_ms
            << ", alloc=" << result.alloc_ms
            << ", h2d=" << result.h2d_ms
            << ", table_h2d=" << result.table_h2d_ms
            << ", slot_compute=" << result.slot_compute_avg_ms
            << ", slot_gather=" << result.slot_gather_avg_ms
            << ", kernel_avg=" << result.kernel_avg_ms
            << ", kernel_min=" << result.kernel_min_ms
            << ", kernel_max=" << result.kernel_max_ms
            << ", d2h=" << result.d2h_ms
            << ", verify=" << result.verify_ms << "\n";
  std::cout << "Throughput: gpu=" << std::setprecision(3) << result.gpu_lookup_ns_per_key
            << " ns/key";
  if (result.lookup == lookup_mode::split) {
    std::cout << " (slot_compute=" << result.slot_compute_ns_per_key
              << ", slot_gather=" << result.slot_gather_ns_per_key << ")";
  }
  if (result.cpu_enabled) {
    std::cout << ", cpu=" << result.cpu_lookup_ns_per_key << " ns/key";
  }
  std::cout << "\n";
  if (result.slot_reuse_analyzed) {
    std::cout << "Slot reuse: warp unique avg=" << result.warp_unique_requests_avg
              << ", warp duplicate ratio=" << result.warp_duplicate_ratio
              << ", block unique avg=" << result.block_unique_requests_avg
              << ", block duplicate ratio=" << result.block_duplicate_ratio << "\n";
  }
  if (result.index_span != 0) {
    double const load_factor =
      static_cast<double>(result.key_count) / static_cast<double>(result.index_span);
    std::cout << "Index range: min=" << result.index_min
              << ", max=" << result.index_max
              << ", span=" << result.index_span
              << ", load-factor=" << std::setprecision(6) << load_factor << "\n";
  }
}

options parse_args(int argc, char** argv)
{
  options opts;

  for (int index = 1; index < argc; ++index) {
    std::string arg = argv[index];
    if (arg == "--help" || arg == "-h") {
      print_usage(argv[0]);
      std::exit(0);
    }
    if (arg == "--hash" && index + 1 < argc) {
      opts.hash_name = argv[++index];
      continue;
    }
    if (arg == "--items-per-thread" && index + 1 < argc) {
      opts.items_per_thread = std::stoi(argv[++index]);
      continue;
    }
    if (arg == "--threads" && index + 1 < argc) {
      opts.threads = std::stoi(argv[++index]);
      continue;
    }
    if (arg == "--iterations" && index + 1 < argc) {
      opts.iterations = std::stoi(argv[++index]);
      continue;
    }
    if (arg == "--warmup" && index + 1 < argc) {
      opts.warmup = std::stoi(argv[++index]);
      continue;
    }
    if (arg == "--compile-mode" && index + 1 < argc) {
      opts.compile_mode_name = argv[++index];
      continue;
    }
    if (arg == "--lookup-mode" && index + 1 < argc) {
      opts.lookup_mode_name = argv[++index];
      continue;
    }
    if (arg == "--table-load-mode" && index + 1 < argc) {
      opts.table_load_mode_name = argv[++index];
      continue;
    }
    if (arg == "--cpu-backend" && index + 1 < argc) {
      opts.cpu_backend_name = argv[++index];
      continue;
    }
    if (arg == "--cpu-vector-width" && index + 1 < argc) {
      opts.cpu_vector_width = std::stoi(argv[++index]);
      continue;
    }
    if (arg == "--cpu-strict-vector-width" && index + 1 < argc) {
      opts.cpu_strict_vector_width = (std::stoi(argv[++index]) != 0);
      continue;
    }
    if (arg == "--analyze-slot-reuse") {
      opts.analyze_slot_reuse = true;
      continue;
    }
    if (arg == "--analysis-only") {
      opts.analyze_slot_reuse = true;
      opts.analysis_only = true;
      continue;
    }
    if (arg == "--device" && index + 1 < argc) {
      opts.device_ordinal = std::stoi(argv[++index]);
      continue;
    }
    if (arg == "--keys-file" && index + 1 < argc) {
      opts.keys_file = argv[++index];
      continue;
    }
    if (arg == "--probe-keys-file" && index + 1 < argc) {
      opts.probe_keys_file = argv[++index];
      continue;
    }
    if (arg == "--max-keys" && index + 1 < argc) {
      opts.max_keys = std::stoull(argv[++index]);
      continue;
    }
    if (arg == "--max-probe-keys" && index + 1 < argc) {
      opts.max_probe_keys = std::stoull(argv[++index]);
      continue;
    }
    if (arg == "--dump-fragment") {
      opts.dump_fragment = true;
      continue;
    }
    if (arg == "--source-out" && index + 1 < argc) {
      opts.source_out_path = argv[++index];
      continue;
    }
    if (arg == "--csv") {
      opts.csv = true;
      continue;
    }
    if (arg == "--csv-header") {
      opts.csv = true;
      opts.csv_header = true;
      continue;
    }
    if (arg == "--embed-table-data") {
      opts.embed_table_data = true;
      continue;
    }
    if (arg == "--no-verify") {
      opts.verify = false;
      continue;
    }

    std::ostringstream message;
    message << "Unknown argument: " << arg;
    throw std::runtime_error(message.str());
  }

  if (!is_supported_hash_name(opts.hash_name)) {
    throw std::runtime_error("Unsupported hash: " + opts.hash_name);
  }
  if (opts.compile_mode_name != "ptx" && opts.compile_mode_name != "lto") {
    throw std::runtime_error("Unsupported compile mode: " + opts.compile_mode_name);
  }
  if (opts.lookup_mode_name != "direct" &&
      opts.lookup_mode_name != "split" &&
      opts.lookup_mode_name != "warpcache" &&
      opts.lookup_mode_name != "blocksort") {
    throw std::runtime_error("Unsupported lookup mode: " + opts.lookup_mode_name);
  }
  if (opts.table_load_mode_name != "generic" &&
      opts.table_load_mode_name != "readonly") {
    throw std::runtime_error("Unsupported table-load mode: " + opts.table_load_mode_name);
  }
  if (!is_supported_backend_name(opts.cpu_backend_name)) {
    throw std::runtime_error("Unsupported CPU backend: " + opts.cpu_backend_name);
  }
  if (opts.cpu_vector_width != 1 && opts.cpu_vector_width != 2 &&
      opts.cpu_vector_width != 4 && opts.cpu_vector_width != 8 &&
      opts.cpu_vector_width != 16) {
    throw std::runtime_error("cpu-vector-width must be one of 1,2,4,8,16");
  }
  if (opts.items_per_thread <= 0) {
    throw std::runtime_error("items-per-thread must be > 0");
  }
  if (opts.threads <= 0) {
    throw std::runtime_error("threads must be > 0");
  }
  if (opts.iterations <= 0) {
    throw std::runtime_error("iterations must be > 0");
  }
  if (opts.warmup < 0) {
    throw std::runtime_error("warmup must be >= 0");
  }

  return opts;
}

benchmark_result run_benchmark(options const& opts)
{
  benchmark_result result;
  auto const selected_hash = parse_hash_function(opts.hash_name);
  result.hash_name = opts.hash_name;
  result.mode = parse_compile_mode(opts.compile_mode_name);
  result.lookup = parse_lookup_mode(opts.lookup_mode_name);
  result.table_load = parse_table_load_mode(opts.table_load_mode_name);
  result.device_ordinal = opts.device_ordinal;
  result.items_per_thread = opts.items_per_thread;
  result.threads = opts.threads;
  result.warmup = opts.warmup;
  result.iterations = opts.iterations;
  result.embedded_table_data = opts.embed_table_data;
  result.cpu_enabled = (opts.cpu_backend_name != "none");
  result.cpu_backend_requested = opts.cpu_backend_name;
  result.cpu_vector_requested = opts.cpu_vector_width;
  result.cpu_strict_vector_width = opts.cpu_strict_vector_width;
  if (result.cpu_enabled && opts.cpu_backend_name == "rawdog-jit" && opts.cpu_vector_width == 2) {
    throw std::runtime_error("cpu-backend=rawdog-jit does not support cpu-vector-width=2");
  }
  bool const validate_membership = !opts.probe_keys_file.empty();

  if (validate_membership && result.lookup != lookup_mode::direct) {
    throw std::runtime_error(
      "Separate build/probe streams are currently only supported for lookup-mode=direct");
  }

  if (result.lookup == lookup_mode::blocksort) {
    auto const total_requests =
      static_cast<std::size_t>(opts.threads) *
      static_cast<std::size_t>(opts.items_per_thread) * 2u;
    if (total_requests == 0 || (total_requests & (total_requests - 1u)) != 0) {
      throw std::runtime_error(
        "blocksort lookup mode requires threads * items_per_thread * 2 to be a power of two");
    }
  }

  auto build_keys = opts.keys_file.empty() ? make_sample_keys()
                                           : load_keys_file(opts.keys_file, opts.max_keys);
  if (build_keys.empty()) {
    throw std::runtime_error("No keys available to benchmark");
  }
  result.key_source = opts.keys_file.empty() ? std::string{"built-in-sample"} : opts.keys_file;
  result.key_count = build_keys.size();
  result.key_bytes = build_keys.size() * sizeof(std::uint64_t);
  auto probe_keys = opts.probe_keys_file.empty()
    ? build_keys
    : load_keys_file(opts.probe_keys_file, opts.max_probe_keys);
  if (opts.probe_keys_file.empty() && opts.max_probe_keys > 0 &&
      probe_keys.size() > opts.max_probe_keys) {
    probe_keys.resize(static_cast<std::size_t>(opts.max_probe_keys));
  }
  if (probe_keys.empty()) {
    throw std::runtime_error("No probe keys available to benchmark");
  }
  result.probe_source = opts.probe_keys_file.empty() ? result.key_source : opts.probe_keys_file;
  result.probe_key_count = probe_keys.size();
  result.probe_key_bytes = probe_keys.size() * sizeof(std::uint64_t);
  result.host_rss_before_build_bytes = current_rss_bytes();

  ph_context_handle context;
  ph_table_handle table;
  cuda_source_handle fragment;
  table_data_handle table_data;
  PH_ONLINE_JIT_TABLE_INFO cpu_table_info{};

  auto build_start = steady_clock::now();
  throw_if_bad(PhOnlineJitOpen(&context.value), "PhOnlineJitOpen");
  throw_if_bad(PhOnlineJitCreateTable64(context.value,
                                        selected_hash,
                                        build_keys.data(),
                                        static_cast<std::uint64_t>(build_keys.size()),
                                        &table.value),
               "PhOnlineJitCreateTable64");
  throw_if_bad(PhOnlineJitGetTableInfo(table.value, &cpu_table_info),
               "PhOnlineJitGetTableInfo");
  auto build_stop = steady_clock::now();
  result.build_ms = elapsed_ms(build_start, build_stop);
  result.host_rss_after_build_bytes = current_rss_bytes();
  result.host_rss_delta_bytes =
    (result.host_rss_after_build_bytes >= result.host_rss_before_build_bytes)
      ? (result.host_rss_after_build_bytes - result.host_rss_before_build_bytes)
      : 0;
  result.host_peak_rss_bytes = peak_rss_bytes();

  if (opts.analyze_slot_reuse) {
    auto const warp_stats =
      analyze_slot_reuse_for_tile(probe_keys, selected_hash, cpu_table_info, 32u, static_cast<std::size_t>(opts.items_per_thread));
    auto const block_stats =
      analyze_slot_reuse_for_tile(probe_keys,
                                  selected_hash,
                                  cpu_table_info,
                                  static_cast<std::size_t>(opts.threads),
                                  static_cast<std::size_t>(opts.items_per_thread));
    result.slot_reuse_analyzed = true;
    result.warp_unique_requests_avg = warp_stats.unique_requests_avg;
    result.warp_duplicate_ratio = warp_stats.duplicate_ratio;
    result.block_unique_requests_avg = block_stats.unique_requests_avg;
    result.block_duplicate_ratio = block_stats.duplicate_ratio;
  }

  if (opts.analysis_only) {
    result.cpu_enabled = false;
    return result;
  }

  if (result.cpu_enabled) {
    if (cpu_table_info.KeySizeInBytes > 4) {
      throw std::runtime_error(
        "CPU bulk-index benchmark requires a table downsized to 32-bit keys");
    }

    preload_llvm_runtime_library(opts.cpu_backend_name);

    PH_ONLINE_JIT_BACKEND effective_backend = PhOnlineJitBackendAuto;
    std::uint32_t effective_vector_width = 0;
    std::uint32_t compile_flags =
      opts.cpu_strict_vector_width ? PH_ONLINE_JIT_COMPILE_FLAG_STRICT_VECTOR_WIDTH : 0;

    auto cpu_compile_start = steady_clock::now();
    result.cpu_compile_hr = PhOnlineJitCompileTableEx(context.value,
                                                      table.value,
                                                      parse_backend(opts.cpu_backend_name),
                                                      static_cast<std::uint32_t>(opts.cpu_vector_width),
                                                      PhOnlineJitMaxIsaAuto,
                                                      compile_flags,
                                                      &effective_backend,
                                                      &effective_vector_width);
    auto cpu_compile_stop = steady_clock::now();
    result.cpu_compile_ms = elapsed_ms(cpu_compile_start, cpu_compile_stop);
    throw_if_bad(result.cpu_compile_hr, "PhOnlineJitCompileTableEx");

    result.cpu_jit_enabled = true;
    result.cpu_backend_effective = backend_to_string(effective_backend);
    result.cpu_vector_effective = static_cast<int>(effective_vector_width == 0 ? 1 : effective_vector_width);
    result.cpu_key_mode = "downsized32";

    auto const downsized_keys32 = make_downsized_keys32(probe_keys, cpu_table_info);
    std::vector<std::uint32_t> cpu_indexes;
    auto const cpu_stats =
      benchmark_cpu_lookup(table.value,
                           downsized_keys32,
                           result.cpu_vector_effective,
                           opts.warmup,
                           opts.iterations,
                           &cpu_indexes);
    result.cpu_lookup_avg_ms = cpu_stats.avg;
    result.cpu_lookup_min_ms = cpu_stats.min;
    result.cpu_lookup_max_ms = cpu_stats.max;
    result.cpu_lookup_ns_per_key = ns_per_key(result.cpu_lookup_avg_ms, result.probe_key_count);
  }

  if (!opts.embed_table_data || opts.verify) {
    auto table_export_start = steady_clock::now();
    throw_if_bad(PhOnlineJitGetCudaTableData(table.value,
                                             &table_data.value,
                                             &result.table_data_bytes,
                                             &result.table_data_element_size,
                                             &result.table_data_elements),
                 "PhOnlineJitGetCudaTableData");
    auto table_export_stop = steady_clock::now();
    result.table_export_ms = elapsed_ms(table_export_start, table_export_stop);
  } else {
    result.table_data_bytes =
      static_cast<std::size_t>(cpu_table_info.NumberOfTableElements) *
      static_cast<std::size_t>(cpu_table_info.AssignedElementSizeInBytes);
    result.table_data_elements = static_cast<std::size_t>(cpu_table_info.NumberOfTableElements);
    result.table_data_element_size = cpu_table_info.AssignedElementSizeInBytes;
  }

  auto emit_start = steady_clock::now();
  auto source_flags = PH_ONLINE_JIT_CUDA_SOURCE_FLAG_OMIT_KERNELS;
  if (!opts.embed_table_data) {
    source_flags |= PH_ONLINE_JIT_CUDA_SOURCE_FLAG_OMIT_TABLE_DATA;
  }
  throw_if_bad(PhOnlineJitGetCudaSourceEx(table.value,
                                          source_flags,
                                          &fragment.value,
                                          &result.fragment_bytes),
               "PhOnlineJitGetCudaSourceEx");
  auto emit_stop = steady_clock::now();
  result.emit_ms = elapsed_ms(emit_start, emit_stop);

  std::string fragment_source(fragment.value, result.fragment_bytes);
  if (opts.dump_fragment) { std::cout << fragment_source << "\n"; }

  auto compose_start = steady_clock::now();
  std::string combined_source =
    compose_translation_unit(fragment_source,
                             opts.items_per_thread,
                             opts.threads,
                             opts.embed_table_data,
                             validate_membership,
                             result.lookup,
                             result.table_load);
  auto compose_stop = steady_clock::now();
  result.compose_ms = elapsed_ms(compose_start, compose_stop);
  result.combined_bytes = combined_source.size();

  if (!opts.source_out_path.empty()) {
    std::ofstream out(opts.source_out_path, std::ios::binary);
    out.write(combined_source.data(),
              static_cast<std::streamsize>(combined_source.size()));
  }

  throw_if_bad(cuInit(0), "cuInit");

  CUdevice device{};
  throw_if_bad(cuDeviceGet(&device, opts.device_ordinal), "cuDeviceGet");
  char device_name[128] = {};
  throw_if_bad(cuDeviceGetName(device_name, static_cast<int>(sizeof(device_name)), device),
               "cuDeviceGetName");
  result.device_name = device_name;
  throw_if_bad(cuDeviceGetAttribute(&result.major,
                                    CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR,
                                    device),
               "cuDeviceGetAttribute(major)");
  throw_if_bad(cuDeviceGetAttribute(&result.minor,
                                    CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR,
                                    device),
               "cuDeviceGetAttribute(minor)");
  if (result.lookup == lookup_mode::warpcache && result.major < 7) {
    throw std::runtime_error("lookup-mode=warpcache requires compute capability 7.0 or newer");
  }
  if (result.lookup == lookup_mode::blocksort) {
    int shared_limit = 0;
    auto const shared_bytes =
      static_cast<std::size_t>(opts.threads) *
      static_cast<std::size_t>(opts.items_per_thread) * 2u *
      sizeof(std::uint32_t) * 3u;
    throw_if_bad(cuDeviceGetAttribute(&shared_limit,
                                      CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK,
                                      device),
                 "cuDeviceGetAttribute(max_shared_memory_per_block)");
    if (shared_bytes > static_cast<std::size_t>(shared_limit)) {
      throw std::runtime_error(
        "blocksort lookup mode exceeds the device shared-memory-per-block limit");
    }
  }

  primary_context_handle primary_context;
  primary_context.device = device;
  CUcontext cu_context{};
  throw_if_bad(cuDevicePrimaryCtxRetain(&cu_context, device), "cuDevicePrimaryCtxRetain");
  primary_context.retained = true;
  throw_if_bad(cuCtxSetCurrent(cu_context), "cuCtxSetCurrent");

  std::size_t free_before_gpu = 0;
  std::size_t total_gpu_mem = 0;
  throw_if_bad(cuCtxSynchronize(), "cuCtxSynchronize(before meminfo)");
  throw_if_bad(cudaMemGetInfo(&free_before_gpu, &total_gpu_mem), "cudaMemGetInfo(before)");

  std::string ptx;
  std::vector<char> cubin;
  if (result.mode == compile_mode::ptx) {
    auto compile_start = steady_clock::now();
    ptx = compile_to_ptx(combined_source, result.major, result.minor);
    auto compile_stop = steady_clock::now();
    result.compile_ms = elapsed_ms(compile_start, compile_stop);
    result.image_bytes = ptx.size();
  } else {
    auto compile_start = steady_clock::now();
    auto ir = compile_to_ltoir(combined_source, result.major, result.minor);
    auto compile_stop = steady_clock::now();
    result.compile_ms = elapsed_ms(compile_start, compile_stop);

    auto link_start = steady_clock::now();
    cubin = link_ltoir_to_cubin(ir, result.major, result.minor);
    auto link_stop = steady_clock::now();
    result.link_ms = elapsed_ms(link_start, link_stop);
    result.image_bytes = cubin.size();
  }

  module_handle module;
  void const* image = (result.mode == compile_mode::ptx)
    ? static_cast<void const*>(ptx.data())
    : static_cast<void const*>(cubin.data());
  auto module_load_start = steady_clock::now();
  throw_if_bad(cuModuleLoadDataEx(&module.value, image, 0, nullptr, nullptr),
               "cuModuleLoadDataEx");
  auto module_load_stop = steady_clock::now();
  result.module_load_ms = elapsed_ms(module_load_start, module_load_stop);

  CUfunction direct_function{};
  CUfunction compute_slots_function{};
  CUfunction gather_function{};
  CUfunction warpcache_function{};
  CUfunction blocksort_function{};
  if (result.lookup == lookup_mode::direct) {
    throw_if_bad(cuModuleGetFunction(&direct_function, module.value, "probe_kernel"),
                 "cuModuleGetFunction(probe_kernel)");
  } else if (result.lookup == lookup_mode::split) {
    throw_if_bad(cuModuleGetFunction(&compute_slots_function, module.value, "compute_slots_kernel"),
                 "cuModuleGetFunction(compute_slots_kernel)");
    throw_if_bad(cuModuleGetFunction(&gather_function, module.value, "gather_kernel"),
                 "cuModuleGetFunction(gather_kernel)");
  } else if (result.lookup == lookup_mode::warpcache) {
    throw_if_bad(cuModuleGetFunction(&warpcache_function, module.value, "warpcache_probe_kernel"),
                 "cuModuleGetFunction(warpcache_probe_kernel)");
  } else {
    throw_if_bad(cuModuleGetFunction(&blocksort_function, module.value, "blocksort_probe_kernel"),
                 "cuModuleGetFunction(blocksort_probe_kernel)");
  }

  device_memory d_keys;
  device_memory d_build_keys;
  device_memory d_table;
  device_memory d_slot1;
  device_memory d_slot2;
  device_memory d_indexes;
  auto alloc_start = steady_clock::now();
  throw_if_bad(cuMemAlloc(&d_keys.value, result.probe_key_bytes), "cuMemAlloc(keys)");
  if (validate_membership) {
    throw_if_bad(cuMemAlloc(&d_build_keys.value, result.key_bytes), "cuMemAlloc(build_keys)");
  }
  if (!opts.embed_table_data) {
    throw_if_bad(cuMemAlloc(&d_table.value, result.table_data_bytes), "cuMemAlloc(table)");
  }
  if (result.lookup == lookup_mode::split) {
    throw_if_bad(cuMemAlloc(&d_slot1.value,
                            probe_keys.size() * sizeof(std::uint32_t)),
                 "cuMemAlloc(slot1)");
    throw_if_bad(cuMemAlloc(&d_slot2.value,
                            probe_keys.size() * sizeof(std::uint32_t)),
                 "cuMemAlloc(slot2)");
  }
  throw_if_bad(cuMemAlloc(&d_indexes.value,
                          probe_keys.size() * sizeof(std::uint32_t)),
               "cuMemAlloc(indexes)");
  auto alloc_stop = steady_clock::now();
  result.alloc_ms = elapsed_ms(alloc_start, alloc_stop);
  result.vram_explicit_bytes = result.probe_key_bytes +
                               (validate_membership ? result.key_bytes : 0) +
                               (!opts.embed_table_data ? result.table_data_bytes : 0) +
                               (probe_keys.size() * sizeof(std::uint32_t)) +
                               ((result.lookup == lookup_mode::split)
                                  ? (probe_keys.size() * sizeof(std::uint32_t) * 2)
                                  : 0);

  auto h2d_start = steady_clock::now();
  throw_if_bad(cuMemcpyHtoD(d_keys.value, probe_keys.data(), result.probe_key_bytes), "cuMemcpyHtoD");
  if (validate_membership) {
    throw_if_bad(cuMemcpyHtoD(d_build_keys.value, build_keys.data(), result.key_bytes),
                 "cuMemcpyHtoD(build_keys)");
  }
  if (!opts.embed_table_data) {
    auto table_h2d_start = steady_clock::now();
    throw_if_bad(cuMemcpyHtoD(d_table.value,
                              table_data.value,
                              result.table_data_bytes),
                 "cuMemcpyHtoD(table)");
    auto table_h2d_stop = steady_clock::now();
    result.table_h2d_ms = elapsed_ms(table_h2d_start, table_h2d_stop);
  }
  auto h2d_stop = steady_clock::now();
  result.h2d_ms = elapsed_ms(h2d_start, h2d_stop);

  result.blocks = static_cast<int>((probe_keys.size() +
                                    static_cast<std::size_t>(opts.threads * opts.items_per_thread) - 1) /
                                   static_cast<std::size_t>(opts.threads * opts.items_per_thread));
  std::size_t count = probe_keys.size();
  std::size_t build_key_count = build_keys.size();
  if (result.lookup == lookup_mode::direct) {
    void* embedded_params_no_check[] = {&d_keys.value, &d_indexes.value, &count};
    void* embedded_params_check[] = {
      &d_keys.value, &d_build_keys.value, &build_key_count, &d_indexes.value, &count};
    void* detached_params_no_check[] = {&d_keys.value, &d_table.value, &d_indexes.value, &count};
    void* detached_params_check[] = {
      &d_keys.value, &d_build_keys.value, &build_key_count, &d_table.value, &d_indexes.value, &count};
    auto kernel_timings = run_probe_iterations(direct_function,
                                               opts.embed_table_data
                                                 ? (validate_membership ? embedded_params_check : embedded_params_no_check)
                                                 : (validate_membership ? detached_params_check : detached_params_no_check),
                                               result.blocks,
                                               opts.threads,
                                               opts.warmup,
                                               opts.iterations);
    auto const stats = summarize_timings(kernel_timings);
    result.kernel_avg_ms = stats.avg;
    result.kernel_min_ms = stats.min;
    result.kernel_max_ms = stats.max;
  } else if (result.lookup == lookup_mode::split) {
    void* compute_params[] = {&d_keys.value, &d_slot1.value, &d_slot2.value, &count};
    auto compute_timings = run_probe_iterations(compute_slots_function,
                                                compute_params,
                                                result.blocks,
                                                opts.threads,
                                                opts.warmup,
                                                opts.iterations);
    auto const compute_stats = summarize_timings(compute_timings);
    result.slot_compute_avg_ms = compute_stats.avg;

    if (opts.embed_table_data) {
      void* gather_params[] = {&d_slot1.value, &d_slot2.value, &d_indexes.value, &count};
      auto gather_timings = run_probe_iterations(gather_function,
                                                 gather_params,
                                                 result.blocks,
                                                 opts.threads,
                                                 opts.warmup,
                                                 opts.iterations);
      auto const gather_stats = summarize_timings(gather_timings);
      result.slot_gather_avg_ms = gather_stats.avg;
      auto combined_timings = run_split_probe_iterations(compute_slots_function,
                                                         compute_params,
                                                         gather_function,
                                                         gather_params,
                                                         result.blocks,
                                                         opts.threads,
                                                         opts.warmup,
                                                         opts.iterations);
      auto const combined_stats = summarize_timings(combined_timings);
      result.kernel_avg_ms = combined_stats.avg;
      result.kernel_min_ms = combined_stats.min;
      result.kernel_max_ms = combined_stats.max;
    } else {
      void* gather_params[] = {&d_slot1.value, &d_slot2.value, &d_table.value, &d_indexes.value, &count};
      auto gather_timings = run_probe_iterations(gather_function,
                                                 gather_params,
                                                 result.blocks,
                                                 opts.threads,
                                                 opts.warmup,
                                                 opts.iterations);
      auto const gather_stats = summarize_timings(gather_timings);
      result.slot_gather_avg_ms = gather_stats.avg;
      auto combined_timings = run_split_probe_iterations(compute_slots_function,
                                                         compute_params,
                                                         gather_function,
                                                         gather_params,
                                                         result.blocks,
                                                         opts.threads,
                                                         opts.warmup,
                                                         opts.iterations);
      auto const combined_stats = summarize_timings(combined_timings);
      result.kernel_avg_ms = combined_stats.avg;
      result.kernel_min_ms = combined_stats.min;
      result.kernel_max_ms = combined_stats.max;
    }
  } else if (result.lookup == lookup_mode::warpcache) {
    void* embedded_params[] = {&d_keys.value, &d_indexes.value, &count};
    void* detached_params[] = {&d_keys.value, &d_table.value, &d_indexes.value, &count};
    auto kernel_timings = run_probe_iterations(warpcache_function,
                                               opts.embed_table_data ? embedded_params : detached_params,
                                               result.blocks,
                                               opts.threads,
                                               opts.warmup,
                                               opts.iterations);
    auto const stats = summarize_timings(kernel_timings);
    result.kernel_avg_ms = stats.avg;
    result.kernel_min_ms = stats.min;
    result.kernel_max_ms = stats.max;
  } else {
    void* embedded_params[] = {&d_keys.value, &d_indexes.value, &count};
    void* detached_params[] = {&d_keys.value, &d_table.value, &d_indexes.value, &count};
    auto kernel_timings = run_probe_iterations(blocksort_function,
                                               opts.embed_table_data ? embedded_params : detached_params,
                                               result.blocks,
                                               opts.threads,
                                               opts.warmup,
                                               opts.iterations);
    auto const stats = summarize_timings(kernel_timings);
    result.kernel_avg_ms = stats.avg;
    result.kernel_min_ms = stats.min;
    result.kernel_max_ms = stats.max;
  }

  std::vector<std::uint32_t> indexes(probe_keys.size());
  auto d2h_start = steady_clock::now();
  throw_if_bad(cuMemcpyDtoH(indexes.data(),
                            d_indexes.value,
                            indexes.size() * sizeof(std::uint32_t)),
               "cuMemcpyDtoH");
  auto d2h_stop = steady_clock::now();
  result.d2h_ms = elapsed_ms(d2h_start, d2h_stop);

  result.slot_compute_ns_per_key = ns_per_key(result.slot_compute_avg_ms, result.probe_key_count);
  result.slot_gather_ns_per_key = ns_per_key(result.slot_gather_avg_ms, result.probe_key_count);
  result.gpu_lookup_ns_per_key = ns_per_key(result.kernel_avg_ms, result.probe_key_count);
  std::size_t free_after_gpu = 0;
  throw_if_bad(cuCtxSynchronize(), "cuCtxSynchronize(after meminfo)");
  throw_if_bad(cudaMemGetInfo(&free_after_gpu, &total_gpu_mem), "cudaMemGetInfo(after)");
  result.vram_total_bytes =
    (free_before_gpu >= free_after_gpu) ? (free_before_gpu - free_after_gpu) : 0;

  if (opts.verify) {
    auto verify_start = steady_clock::now();
    if (probe_keys == build_keys) {
      verify_indexes(indexes, build_keys, probe_keys, &result);
    } else {
      for (std::size_t i = 0; i < probe_keys.size(); ++i) {
        auto const candidate =
          index_from_key_host(probe_keys[i], selected_hash, cpu_table_info, table_data.value);
        auto const expected =
          (candidate < build_keys.size() && build_keys[candidate] == probe_keys[i])
            ? candidate
            : std::numeric_limits<std::uint32_t>::max();
        if (indexes[i] != expected) {
          throw std::runtime_error("Verification failed at probe index " + std::to_string(i));
        }
      }
      if (!indexes.empty()) {
        std::uint32_t min_index = std::numeric_limits<std::uint32_t>::max();
        std::uint32_t max_index = 0;
        bool saw_hit = false;
        for (auto const index : indexes) {
          if (index == std::numeric_limits<std::uint32_t>::max()) {
            continue;
          }
          min_index = std::min(min_index, index);
          max_index = std::max(max_index, index);
          saw_hit = true;
        }
        if (saw_hit) {
          result.index_min = min_index;
          result.index_max = max_index;
          result.index_span = static_cast<std::size_t>(result.index_max) + 1;
        } else {
          result.index_min = 0;
          result.index_max = 0;
          result.index_span = 0;
        }
      }
    }
    auto verify_stop = steady_clock::now();
    result.verify_ms = elapsed_ms(verify_start, verify_stop);
  }

  if (opts.verify && result.cpu_enabled) {
    auto const downsized_keys32 = make_downsized_keys32(probe_keys, cpu_table_info);
    std::vector<std::uint32_t> cpu_indexes;
    run_cpu_bulk_index_once(table.value,
                            downsized_keys32,
                            result.cpu_vector_effective,
                            &cpu_indexes);
    if (validate_membership) {
      for (std::size_t i = 0; i < cpu_indexes.size(); ++i) {
        auto const candidate = cpu_indexes[i];
        cpu_indexes[i] =
          (candidate < build_keys.size() && build_keys[candidate] == probe_keys[i])
            ? candidate
            : std::numeric_limits<std::uint32_t>::max();
      }
    }
    if (cpu_indexes != indexes) {
      throw std::runtime_error("CPU and GPU index outputs differed");
    }
  }

  return result;
}

}  // namespace

int main(int argc, char** argv)
{
  try {
    auto const opts = parse_args(argc, argv);
    auto const result = run_benchmark(opts);
    if (opts.csv_header) { print_csv_header(); }
    if (opts.csv) { print_csv_row(result); }
    else { print_human_summary(result); }
    return 0;
  } catch (std::exception const& e) {
    std::cerr << e.what() << "\n";
    return 1;
  }
}
