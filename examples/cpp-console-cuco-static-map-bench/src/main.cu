#include <cuco/pair.cuh>
#include <cuco/static_map.cuh>

#include <cuda/stream_ref>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using key_type = std::uint64_t;
using value_type = std::uint32_t;
using pair_type = cuco::pair<key_type, value_type>;

struct options {
  std::string keys_file;
  std::string probe_keys_file;
  std::uint64_t max_keys = 0;
  std::uint64_t max_probe_keys = 0;
  double load_factor = 0.5;
  int device_ordinal = 0;
  int warmup = 2;
  int iterations = 10;
  bool csv = false;
  bool csv_header = false;
  bool verify = true;
};

struct result_row {
  std::string key_source;
  std::string probe_source;
  std::string device_name;
  std::size_t build_key_count = 0;
  std::size_t build_key_bytes = 0;
  std::size_t probe_key_count = 0;
  std::size_t probe_key_bytes = 0;
  std::size_t pair_bytes = 0;
  std::size_t output_bytes = 0;
  std::size_t slot_bytes = 0;
  std::size_t capacity = 0;
  std::size_t size = 0;
  double requested_load_factor = 0.0;
  double actual_load_factor = 0.0;
  double h2d_keys_ms = 0.0;
  double pair_prepare_ms = 0.0;
  double insert_ms = 0.0;
  double build_ms = 0.0;
  double lookup_avg_ms = 0.0;
  double lookup_min_ms = 0.0;
  double lookup_max_ms = 0.0;
  double lookup_ns_per_key = 0.0;
  double verify_ms = 0.0;
  std::size_t vram_buffers_bytes = 0;
  std::size_t vram_total_bytes = 0;
  int device_ordinal = 0;
};

void throw_if_cuda(cudaError_t status, char const* what)
{
  if (status != cudaSuccess) {
    throw std::runtime_error(std::string(what) + " failed: " + cudaGetErrorString(status));
  }
}

double ms_between(cudaEvent_t start, cudaEvent_t stop)
{
  float ms = 0.0f;
  throw_if_cuda(cudaEventElapsedTime(&ms, start, stop), "cudaEventElapsedTime");
  return static_cast<double>(ms);
}

double ns_per_key(double milliseconds, std::size_t key_count)
{
  if (key_count == 0) { return 0.0; }
  return (milliseconds * 1'000'000.0) / static_cast<double>(key_count);
}

std::vector<key_type> load_keys_file(std::string const& path, std::uint64_t max_keys)
{
  std::ifstream input(path, std::ios::binary | std::ios::ate);
  if (!input) { throw std::runtime_error("Unable to open keys file: " + path); }

  auto const size = input.tellg();
  if (size < 0) { throw std::runtime_error("Unable to determine keys file size: " + path); }
  if ((static_cast<std::uint64_t>(size) % sizeof(key_type)) != 0) {
    throw std::runtime_error("Keys file size is not a multiple of 8 bytes: " + path);
  }

  auto const available_keys =
    static_cast<std::uint64_t>(size) / static_cast<std::uint64_t>(sizeof(key_type));
  auto const keys_to_read =
    (max_keys != 0 && max_keys < available_keys) ? max_keys : available_keys;

  std::vector<key_type> keys(static_cast<std::size_t>(keys_to_read));
  input.seekg(0, std::ios::beg);
  input.read(reinterpret_cast<char*>(keys.data()),
             static_cast<std::streamsize>(keys.size() * sizeof(key_type)));
  if (!input) { throw std::runtime_error("Unable to read keys file contents: " + path); }

  return keys;
}

void print_usage(char const* argv0)
{
  std::cout << "Usage: " << argv0
            << " --keys-file <path> [--probe-keys-file <path>] [--max-keys <N>] [--max-probe-keys <N>] [--load-factor <f>]\n"
               "       [--device <ordinal>] [--warmup <N>] [--iterations <N>]\n"
               "       [--csv] [--csv-header] [--no-verify]\n";
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
    if (arg == "--load-factor" && index + 1 < argc) {
      opts.load_factor = std::stod(argv[++index]);
      continue;
    }
    if (arg == "--device" && index + 1 < argc) {
      opts.device_ordinal = std::stoi(argv[++index]);
      continue;
    }
    if (arg == "--warmup" && index + 1 < argc) {
      opts.warmup = std::stoi(argv[++index]);
      continue;
    }
    if (arg == "--iterations" && index + 1 < argc) {
      opts.iterations = std::stoi(argv[++index]);
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
    if (arg == "--no-verify") {
      opts.verify = false;
      continue;
    }
    throw std::runtime_error("Unknown argument: " + arg);
  }

  if (opts.keys_file.empty()) { throw std::runtime_error("--keys-file is required"); }
  if (!(opts.load_factor > 0.0 && opts.load_factor < 1.0)) {
    throw std::runtime_error("--load-factor must be in the open interval (0, 1)");
  }
  if (opts.warmup < 0) { throw std::runtime_error("--warmup must be >= 0"); }
  if (opts.iterations <= 0) { throw std::runtime_error("--iterations must be > 0"); }

  return opts;
}

__global__ void fill_pairs_kernel(key_type const* keys, pair_type* pairs, std::size_t count)
{
  auto const tid = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (tid < count) { pairs[tid] = pair_type{keys[tid], static_cast<value_type>(tid)}; }
}

void print_csv_header()
{
  std::cout << "key_source,build_key_count,build_key_bytes,probe_source,probe_key_count,probe_key_bytes,pair_bytes,output_bytes,slot_bytes,requested_load_factor,"
               "capacity,size,actual_load_factor,h2d_keys_ms,pair_prepare_ms,insert_ms,build_ms,"
               "lookup_avg_ms,lookup_min_ms,lookup_max_ms,lookup_ns_per_key,verify_ms,"
               "vram_buffers_bytes,vram_total_bytes,device_ordinal,device_name\n";
}

void print_csv_row(result_row const& result)
{
  std::cout << std::quoted(result.key_source) << ','
            << result.build_key_count << ','
            << result.build_key_bytes << ','
            << std::quoted(result.probe_source) << ','
            << result.probe_key_count << ','
            << result.probe_key_bytes << ','
            << result.pair_bytes << ','
            << result.output_bytes << ','
            << result.slot_bytes << ','
            << std::fixed << std::setprecision(3)
            << result.requested_load_factor << ','
            << result.capacity << ','
            << result.size << ','
            << result.actual_load_factor << ','
            << result.h2d_keys_ms << ','
            << result.pair_prepare_ms << ','
            << result.insert_ms << ','
            << result.build_ms << ','
            << result.lookup_avg_ms << ','
            << result.lookup_min_ms << ','
            << result.lookup_max_ms << ','
            << result.lookup_ns_per_key << ','
            << result.verify_ms << ','
            << result.vram_buffers_bytes << ','
            << result.vram_total_bytes << ','
            << result.device_ordinal << ','
            << std::quoted(result.device_name) << '\n';
}

void print_human_summary(result_row const& result)
{
  std::cout << "Build: " << result.build_key_count << " from " << result.key_source << "\n";
  std::cout << "Probe: " << result.probe_key_count << " from " << result.probe_source << "\n";
  std::cout << "Device: " << result.device_ordinal << " (" << result.device_name << ")\n";
  std::cout << std::fixed << std::setprecision(3)
            << "Map: requested load factor=" << result.requested_load_factor
            << ", capacity=" << result.capacity
            << ", size=" << result.size
            << ", actual load factor=" << result.actual_load_factor << "\n";
  std::cout << "Memory: build_keys=" << result.build_key_bytes
            << " bytes, probe_keys=" << result.probe_key_bytes
            << " bytes, pairs=" << result.pair_bytes
            << " bytes, output=" << result.output_bytes
            << " bytes, slots=" << result.slot_bytes
            << " bytes, vram_buffers=" << result.vram_buffers_bytes
            << " bytes, vram_total=" << result.vram_total_bytes << " bytes\n";
  std::cout << "Timings (ms): h2d_keys=" << result.h2d_keys_ms
            << ", pair_prepare=" << result.pair_prepare_ms
            << ", insert=" << result.insert_ms
            << ", build=" << result.build_ms
            << ", lookup_avg=" << result.lookup_avg_ms
            << ", lookup_min=" << result.lookup_min_ms
            << ", lookup_max=" << result.lookup_max_ms
            << ", verify=" << result.verify_ms << "\n";
  std::cout << "Throughput: " << result.lookup_ns_per_key << " ns/key\n";
}

}  // namespace

int main(int argc, char** argv)
{
  try {
    auto const opts = parse_args(argc, argv);

    throw_if_cuda(cudaSetDevice(opts.device_ordinal), "cudaSetDevice");

    cudaDeviceProp prop{};
    throw_if_cuda(cudaGetDeviceProperties(&prop, opts.device_ordinal), "cudaGetDeviceProperties");

    auto const host_build_keys = load_keys_file(opts.keys_file, opts.max_keys);
    auto const build_key_count = host_build_keys.size();
    if (build_key_count == 0) {
      throw std::runtime_error("No build keys available to benchmark");
    }
    auto sorted_build_keys = host_build_keys;
    std::sort(sorted_build_keys.begin(), sorted_build_keys.end());
    auto host_probe_keys = opts.probe_keys_file.empty()
      ? host_build_keys
      : load_keys_file(opts.probe_keys_file, opts.max_probe_keys);
    if (opts.probe_keys_file.empty() && opts.max_probe_keys > 0 &&
        host_probe_keys.size() > opts.max_probe_keys) {
      host_probe_keys.resize(static_cast<std::size_t>(opts.max_probe_keys));
    }
    auto const probe_key_count = host_probe_keys.size();

    if (build_key_count > static_cast<std::size_t>(std::numeric_limits<value_type>::max())) {
      throw std::runtime_error("Key count exceeds uint32_t ordinal range");
    }

    std::size_t free_before_buffers = 0;
    std::size_t total_device_mem = 0;
    throw_if_cuda(cudaMemGetInfo(&free_before_buffers, &total_device_mem), "cudaMemGetInfo(before buffers)");

    thrust::device_vector<key_type> d_build_keys(build_key_count);
    thrust::device_vector<pair_type> d_pairs(build_key_count);
    thrust::device_vector<key_type> d_probe_keys(probe_key_count);
    thrust::device_vector<value_type> d_values(probe_key_count);

    std::size_t free_after_buffers = 0;
    throw_if_cuda(cudaMemGetInfo(&free_after_buffers, &total_device_mem), "cudaMemGetInfo(after buffers)");

    auto constexpr empty_key = std::numeric_limits<key_type>::max();
    auto constexpr empty_value = std::numeric_limits<value_type>::max();
    if (std::find(host_build_keys.begin(), host_build_keys.end(), empty_key) != host_build_keys.end()) {
      throw std::runtime_error("Build keys contain the reserved empty-key sentinel");
    }
    if (std::find(host_probe_keys.begin(), host_probe_keys.end(), empty_key) != host_probe_keys.end()) {
      throw std::runtime_error("Probe keys contain the reserved empty-key sentinel");
    }

    auto map = cuco::static_map{build_key_count,
                                opts.load_factor,
                                cuco::empty_key{empty_key},
                                cuco::empty_value{empty_value}};

    std::size_t free_after_map = 0;
    throw_if_cuda(cudaMemGetInfo(&free_after_map, &total_device_mem), "cudaMemGetInfo(after map)");

    cudaStream_t stream{};
    throw_if_cuda(cudaStreamCreate(&stream), "cudaStreamCreate");
    cudaEvent_t start{};
    cudaEvent_t stop{};
    throw_if_cuda(cudaEventCreate(&start), "cudaEventCreate(start)");
    throw_if_cuda(cudaEventCreate(&stop), "cudaEventCreate(stop)");

    auto cleanup = [&]() {
      cudaEventDestroy(stop);
      cudaEventDestroy(start);
      cudaStreamDestroy(stream);
    };

    auto const build_blocks = static_cast<unsigned>((build_key_count + 255) / 256);

    throw_if_cuda(cudaEventRecord(start, stream), "cudaEventRecord(h2d start)");
    throw_if_cuda(cudaMemcpyAsync(thrust::raw_pointer_cast(d_build_keys.data()),
                                  host_build_keys.data(),
                                  build_key_count * sizeof(key_type),
                                  cudaMemcpyHostToDevice,
                                  stream),
                  "cudaMemcpyAsync(build keys)");
    throw_if_cuda(cudaEventRecord(stop, stream), "cudaEventRecord(h2d stop)");
    throw_if_cuda(cudaEventSynchronize(stop), "cudaEventSynchronize(h2d)");
    auto const h2d_keys_ms = ms_between(start, stop);

    throw_if_cuda(cudaEventRecord(start, stream), "cudaEventRecord(pairs start)");
    fill_pairs_kernel<<<build_blocks, 256, 0, stream>>>(
      thrust::raw_pointer_cast(d_build_keys.data()),
      thrust::raw_pointer_cast(d_pairs.data()),
      build_key_count);
    throw_if_cuda(cudaGetLastError(), "fill_pairs_kernel launch");
    throw_if_cuda(cudaEventRecord(stop, stream), "cudaEventRecord(pairs stop)");
    throw_if_cuda(cudaEventSynchronize(stop), "cudaEventSynchronize(pairs)");
    auto const pair_prepare_ms = ms_between(start, stop);

    throw_if_cuda(cudaEventRecord(start, stream), "cudaEventRecord(insert start)");
    map.insert_async(d_pairs.begin(), d_pairs.end(), cuda::stream_ref{stream});
    throw_if_cuda(cudaEventRecord(stop, stream), "cudaEventRecord(insert stop)");
    throw_if_cuda(cudaEventSynchronize(stop), "cudaEventSynchronize(insert)");
    auto const insert_ms = ms_between(start, stop);
    auto const build_ms = h2d_keys_ms + pair_prepare_ms + insert_ms;

    throw_if_cuda(cudaMemcpyAsync(thrust::raw_pointer_cast(d_probe_keys.data()),
                                  host_probe_keys.data(),
                                  probe_key_count * sizeof(key_type),
                                  cudaMemcpyHostToDevice,
                                  stream),
                  "cudaMemcpyAsync(probe keys)");
    throw_if_cuda(cudaStreamSynchronize(stream), "cudaStreamSynchronize(probe copy)");

    for (int i = 0; i < opts.warmup; ++i) {
      map.find_async(d_probe_keys.begin(), d_probe_keys.end(), d_values.begin(), cuda::stream_ref{stream});
      throw_if_cuda(cudaStreamSynchronize(stream), "cudaStreamSynchronize(warmup)");
    }

    std::vector<double> lookup_timings;
    lookup_timings.reserve(static_cast<std::size_t>(opts.iterations));
    for (int i = 0; i < opts.iterations; ++i) {
      throw_if_cuda(cudaEventRecord(start, stream), "cudaEventRecord(find start)");
      map.find_async(d_probe_keys.begin(), d_probe_keys.end(), d_values.begin(), cuda::stream_ref{stream});
      throw_if_cuda(cudaEventRecord(stop, stream), "cudaEventRecord(find stop)");
      throw_if_cuda(cudaEventSynchronize(stop), "cudaEventSynchronize(find)");
      lookup_timings.push_back(ms_between(start, stop));
    }

    auto const [min_it, max_it] = std::minmax_element(lookup_timings.begin(), lookup_timings.end());
    double lookup_total = 0.0;
    for (double t : lookup_timings) { lookup_total += t; }
    auto const lookup_avg_ms = lookup_total / static_cast<double>(lookup_timings.size());

    double verify_ms = 0.0;
    if (opts.verify) {
      auto const verify_start = std::chrono::steady_clock::now();
      thrust::host_vector<value_type> h_values = d_values;
      bool const same_probe_stream = (host_probe_keys == host_build_keys);
      for (std::size_t i = 0; i < probe_key_count; ++i) {
        bool const in_build =
          std::binary_search(sorted_build_keys.begin(), sorted_build_keys.end(), host_probe_keys[i]);
        if (in_build) {
          if (h_values[i] == empty_value) {
            throw std::runtime_error("Verification failed at index " + std::to_string(i));
          }
          if (same_probe_stream && h_values[i] != static_cast<value_type>(i)) {
            throw std::runtime_error("Verification failed at index " + std::to_string(i));
          }
          if (static_cast<std::size_t>(h_values[i]) >= build_key_count ||
              host_build_keys[static_cast<std::size_t>(h_values[i])] != host_probe_keys[i]) {
            throw std::runtime_error("Verification failed at index " + std::to_string(i));
          }
        } else if (h_values[i] != empty_value) {
          throw std::runtime_error("Verification failed at index " + std::to_string(i));
        }
      }
      auto const verify_stop = std::chrono::steady_clock::now();
      verify_ms = std::chrono::duration<double, std::milli>(verify_stop - verify_start).count();
    }

    result_row result;
    result.key_source = opts.keys_file;
    result.probe_source = opts.probe_keys_file.empty() ? opts.keys_file : opts.probe_keys_file;
    result.device_name = prop.name;
    result.build_key_count = build_key_count;
    result.build_key_bytes = build_key_count * sizeof(key_type);
    result.probe_key_count = probe_key_count;
    result.probe_key_bytes = probe_key_count * sizeof(key_type);
    result.pair_bytes = build_key_count * sizeof(pair_type);
    result.output_bytes = probe_key_count * sizeof(value_type);
    result.capacity = map.capacity();
    result.size = map.size(cuda::stream_ref{stream});
    result.slot_bytes = result.capacity * sizeof(pair_type);
    result.requested_load_factor = opts.load_factor;
    result.actual_load_factor =
      static_cast<double>(result.size) / static_cast<double>(result.capacity);
    result.h2d_keys_ms = h2d_keys_ms;
    result.pair_prepare_ms = pair_prepare_ms;
    result.insert_ms = insert_ms;
    result.build_ms = build_ms;
    result.lookup_avg_ms = lookup_avg_ms;
    result.lookup_min_ms = *min_it;
    result.lookup_max_ms = *max_it;
    result.lookup_ns_per_key = ns_per_key(lookup_avg_ms, probe_key_count);
    result.verify_ms = verify_ms;
    result.vram_buffers_bytes = free_before_buffers - free_after_buffers;
    result.vram_total_bytes = free_before_buffers - free_after_map;
    result.device_ordinal = opts.device_ordinal;

    cleanup();

    if (opts.csv_header) { print_csv_header(); }
    if (opts.csv) { print_csv_row(result); }
    else { print_human_summary(result); }
    return 0;
  } catch (std::exception const& e) {
    std::cerr << e.what() << "\n";
    return 1;
  }
}
