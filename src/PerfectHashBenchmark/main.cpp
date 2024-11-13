#include <benchmark/benchmark.h>

// Define a simple benchmark
static void BM_SimpleFunction(benchmark::State& state) {
    for (auto _ : state) {
        // This is the code you want to measure the performance of
    }
}
// Register the function as a benchmark
BENCHMARK(BM_SimpleFunction);

// Main function to run the benchmarks
BENCHMARK_MAIN();