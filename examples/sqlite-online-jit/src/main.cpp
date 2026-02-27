#include "perfecthash_vtab.h"

#include <sqlite3.h>

#include <algorithm>
#include <chrono>
#include <cerrno>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <random>
#include <sstream>
#include <string>
#include <vector>

#if !defined(_WIN32) && defined(PH_ONLINE_JIT_LLVM_LIBRARY_PATH)
#include <dlfcn.h>
#endif

namespace {

enum class RunMode {
  Single,
  Matrix,
};

struct Options {
  RunMode mode = RunMode::Matrix;
  bool mode_explicit = false;

  std::string backend = "rawdog-jit";
  std::string hash = "mulshrolate2rx";
  uint32_t vector_width = 16;
  bool strict_vector_width = false;

  bool backend_set = false;
  bool hash_set = false;
  bool vector_width_set = false;
  bool strict_vector_width_set = false;

  uint32_t dim_size = 5'000;
  uint32_t fact_size = 100'000;
  uint32_t iterations = 2;
  uint32_t seed = 1;
  uint32_t build_runs = 1;

  std::string output_detailed_csv;
  std::string output_summary_csv;
};

struct BenchmarkCase {
  std::string backend;
  std::string hash;
  uint32_t vector_width = 16;
  bool strict_vector_width = false;
};

struct BenchmarkRun {
  BenchmarkCase config;
  uint32_t run_index = 0;

  bool create_ok = false;
  bool query_ok = false;
  bool result_match = false;
  bool have_build_status = false;

  PerfectHashBuildStatus build_status;

  sqlite3_int64 sum = 0;
  double create_ms = 0.0;
  double query_ms = 0.0;
  double end_to_end_ms = 0.0;

  double source_extract_ms = 0.0;
  double table_create_ms = 0.0;
  double compile_ms = 0.0;
  double materialize_ms = 0.0;
  double internal_create_ms = 0.0;

  std::string error;
};

struct BenchmarkResult {
  BenchmarkCase config;
  std::vector<BenchmarkRun> runs;

  bool any_create_ok = false;
  bool any_query_ok = false;
  bool all_result_match = true;

  uint32_t total_runs = 0;
  uint32_t create_success_runs = 0;
  uint32_t query_success_runs = 0;
  uint32_t jit_success_runs = 0;
  uint32_t jit_fallback_runs = 0;

  double create_avg_ms = 0.0;
  double create_min_ms = 0.0;
  double create_max_ms = 0.0;

  double query_avg_ms = 0.0;
  double query_min_ms = 0.0;
  double query_max_ms = 0.0;

  double end_to_end_avg_ms = 0.0;
  double end_to_end_min_ms = 0.0;
  double end_to_end_max_ms = 0.0;

  double source_extract_avg_ms = 0.0;
  double source_extract_min_ms = 0.0;
  double source_extract_max_ms = 0.0;

  double table_create_avg_ms = 0.0;
  double table_create_min_ms = 0.0;
  double table_create_max_ms = 0.0;

  double compile_avg_ms = 0.0;
  double compile_min_ms = 0.0;
  double compile_max_ms = 0.0;

  double materialize_avg_ms = 0.0;
  double materialize_min_ms = 0.0;
  double materialize_max_ms = 0.0;

  double internal_create_avg_ms = 0.0;
  double internal_create_min_ms = 0.0;
  double internal_create_max_ms = 0.0;

  sqlite3_int64 representative_sum = 0;
  std::string outcome = "ok";
  std::string error;
};

std::string HrToHex(int32_t hr) {
  std::ostringstream stream;
  stream << "0x" << std::hex << std::uppercase << static_cast<uint32_t>(hr);
  return stream.str();
}

std::string CsvEscape(const std::string &text) {
  bool needs_quotes = false;
  for (const char c : text) {
    if (c == ',' || c == '"' || c == '\n' || c == '\r') {
      needs_quotes = true;
      break;
    }
  }

  if (!needs_quotes) {
    return text;
  }

  std::string escaped;
  escaped.reserve(text.size() + 2);
  escaped.push_back('"');
  for (const char c : text) {
    if (c == '"') {
      escaped.push_back('"');
      escaped.push_back('"');
    } else {
      escaped.push_back(c);
    }
  }
  escaped.push_back('"');
  return escaped;
}

void PrintUsage(const char *argv0) {
  std::cout
      << "Usage: " << argv0
      << " [--matrix|--single]"
         " [--backend <rawdog-jit|llvm-jit|auto>]"
         " [--hash <name>]"
         " [--vector-width <0|1|2|4|8|16>]"
         " [--strict-vector-width <0|1>]"
         " [--build-runs <count>]"
         " [--output-detailed-csv <path>]"
         " [--output-summary-csv <path>]"
         " [--dim-size <count>]"
         " [--fact-size <count>]"
         " [--iterations <count>]"
         " [--seed <value>]\n"
      << "\n"
      << "Default behavior: run full matrix across RawDog-JIT + LLVM-JIT,\n"
      << "all curated hash functions, vector widths 1/2/4/8/16.\n"
      << "Pass --build-runs N to measure creation-time distributions per permutation.\n";
}

bool ParseUint32(const char *text, uint32_t *value) {
  if (!text || !value || !*text) {
    return false;
  }

  char *end = nullptr;
  errno = 0;
  const unsigned long parsed = strtoul(text, &end, 10);
  if (errno != 0 || !end || *end != '\0' ||
      parsed > std::numeric_limits<uint32_t>::max()) {
    return false;
  }

  *value = static_cast<uint32_t>(parsed);
  return true;
}

bool ParseArgs(int argc, char **argv, Options *options) {
  if (!options) {
    return false;
  }

  for (int index = 1; index < argc; ++index) {
    const char *arg = argv[index];

    if (std::strcmp(arg, "--help") == 0 || std::strcmp(arg, "-h") == 0) {
      PrintUsage(argv[0]);
      std::exit(0);
    }

    auto require_value = [&](const char *name) -> const char * {
      if (index + 1 >= argc) {
        std::cerr << "Missing value for " << name << "\n";
        return nullptr;
      }
      return argv[++index];
    };

    if (std::strcmp(arg, "--matrix") == 0) {
      options->mode = RunMode::Matrix;
      options->mode_explicit = true;
      continue;
    }

    if (std::strcmp(arg, "--single") == 0) {
      options->mode = RunMode::Single;
      options->mode_explicit = true;
      continue;
    }

    if (std::strcmp(arg, "--backend") == 0) {
      const char *value = require_value("--backend");
      if (!value) {
        return false;
      }
      options->backend = value;
      options->backend_set = true;
      continue;
    }

    if (std::strcmp(arg, "--hash") == 0) {
      const char *value = require_value("--hash");
      if (!value) {
        return false;
      }
      options->hash = value;
      options->hash_set = true;
      continue;
    }

    if (std::strcmp(arg, "--vector-width") == 0) {
      const char *value = require_value("--vector-width");
      if (!value || !ParseUint32(value, &options->vector_width)) {
        std::cerr << "Invalid vector width.\n";
        return false;
      }
      options->vector_width_set = true;
      continue;
    }

    if (std::strcmp(arg, "--strict-vector-width") == 0) {
      uint32_t strict_value = 0;
      const char *value = require_value("--strict-vector-width");
      if (!value || !ParseUint32(value, &strict_value) || strict_value > 1) {
        std::cerr << "strict-vector-width must be 0 or 1.\n";
        return false;
      }
      options->strict_vector_width = (strict_value != 0);
      options->strict_vector_width_set = true;
      continue;
    }

    if (std::strcmp(arg, "--build-runs") == 0) {
      const char *value = require_value("--build-runs");
      if (!value || !ParseUint32(value, &options->build_runs) ||
          options->build_runs == 0) {
        std::cerr << "Invalid build-runs value.\n";
        return false;
      }
      continue;
    }

    if (std::strcmp(arg, "--output-detailed-csv") == 0) {
      const char *value = require_value("--output-detailed-csv");
      if (!value) {
        return false;
      }
      options->output_detailed_csv = value;
      continue;
    }

    if (std::strcmp(arg, "--output-summary-csv") == 0) {
      const char *value = require_value("--output-summary-csv");
      if (!value) {
        return false;
      }
      options->output_summary_csv = value;
      continue;
    }

    if (std::strcmp(arg, "--dim-size") == 0) {
      const char *value = require_value("--dim-size");
      if (!value || !ParseUint32(value, &options->dim_size) ||
          options->dim_size == 0) {
        std::cerr << "Invalid dim size.\n";
        return false;
      }
      continue;
    }

    if (std::strcmp(arg, "--fact-size") == 0) {
      const char *value = require_value("--fact-size");
      if (!value || !ParseUint32(value, &options->fact_size) ||
          options->fact_size == 0) {
        std::cerr << "Invalid fact size.\n";
        return false;
      }
      continue;
    }

    if (std::strcmp(arg, "--iterations") == 0) {
      const char *value = require_value("--iterations");
      if (!value || !ParseUint32(value, &options->iterations) ||
          options->iterations == 0) {
        std::cerr << "Invalid iterations count.\n";
        return false;
      }
      continue;
    }

    if (std::strcmp(arg, "--seed") == 0) {
      const char *value = require_value("--seed");
      if (!value || !ParseUint32(value, &options->seed)) {
        std::cerr << "Invalid seed.\n";
        return false;
      }
      continue;
    }

    std::cerr << "Unknown argument: " << arg << "\n";
    return false;
  }

  if (!options->mode_explicit &&
      (options->backend_set || options->hash_set || options->vector_width_set ||
       options->strict_vector_width_set)) {
    options->mode = RunMode::Single;
  }

  if (options->backend != "rawdog-jit" && options->backend != "llvm-jit" &&
      options->backend != "auto") {
    std::cerr << "Unsupported backend: " << options->backend << "\n";
    return false;
  }

  switch (options->vector_width) {
    case 0:
    case 1:
    case 2:
    case 4:
    case 8:
    case 16:
      break;
    default:
      std::cerr << "Vector width must be one of 0,1,2,4,8,16.\n";
      return false;
  }

  return true;
}

void PreloadLlvmRuntimeLibrary(const std::string &backend) {
#if !defined(_WIN32) && defined(PH_ONLINE_JIT_LLVM_LIBRARY_PATH)
  if (backend == "llvm-jit" || backend == "auto") {
    void *handle = dlopen(PH_ONLINE_JIT_LLVM_LIBRARY_PATH, RTLD_NOW | RTLD_GLOBAL);
    if (!handle) {
      std::cerr << "Warning: unable to preload LLVM runtime from "
                << PH_ONLINE_JIT_LLVM_LIBRARY_PATH << ": " << dlerror()
                << "\n";
    }
  }
#else
  (void)backend;
#endif
}

bool ExecSql(sqlite3 *db, const std::string &sql, std::string *error = nullptr) {
  char *raw_error = nullptr;
  const int rc = sqlite3_exec(db, sql.c_str(), nullptr, nullptr, &raw_error);
  if (rc != SQLITE_OK) {
    const std::string sqlite_error =
        raw_error ? raw_error : std::string(sqlite3_errmsg(db));
    sqlite3_free(raw_error);

    if (error) {
      *error = sqlite_error;
    }

    std::cerr << "SQL failed: " << sql << "\n";
    std::cerr << "sqlite error: " << sqlite_error << "\n";
    return false;
  }

  return true;
}

bool PopulateTables(sqlite3 *db, const Options &options) {
  if (!ExecSql(db, "CREATE TABLE dim(key INTEGER PRIMARY KEY, value INTEGER NOT NULL);") ||
      !ExecSql(db, "CREATE TABLE fact(key INTEGER NOT NULL, measure INTEGER NOT NULL);") ||
      !ExecSql(db, "BEGIN TRANSACTION;")) {
    return false;
  }

  sqlite3_stmt *dim_insert = nullptr;
  int rc = sqlite3_prepare_v2(db,
                              "INSERT INTO dim(key, value) VALUES(?, ?);",
                              -1,
                              &dim_insert,
                              nullptr);
  if (rc != SQLITE_OK) {
    std::cerr << "Failed to prepare dim insert: " << sqlite3_errmsg(db) << "\n";
    return false;
  }

  for (uint32_t i = 1; i <= options.dim_size; ++i) {
    sqlite3_bind_int64(dim_insert, 1, static_cast<sqlite3_int64>(i));
    sqlite3_bind_int64(dim_insert,
                       2,
                       static_cast<sqlite3_int64>(i) * 3 + 7);
    rc = sqlite3_step(dim_insert);
    if (rc != SQLITE_DONE) {
      std::cerr << "dim insert failed: " << sqlite3_errmsg(db) << "\n";
      sqlite3_finalize(dim_insert);
      return false;
    }
    sqlite3_reset(dim_insert);
    sqlite3_clear_bindings(dim_insert);
  }

  sqlite3_finalize(dim_insert);
  dim_insert = nullptr;

  sqlite3_stmt *fact_insert = nullptr;
  rc = sqlite3_prepare_v2(db,
                          "INSERT INTO fact(key, measure) VALUES(?, ?);",
                          -1,
                          &fact_insert,
                          nullptr);
  if (rc != SQLITE_OK) {
    std::cerr << "Failed to prepare fact insert: " << sqlite3_errmsg(db) << "\n";
    return false;
  }

  std::mt19937 rng(options.seed);
  std::uniform_int_distribution<uint32_t> key_dist(1, options.dim_size);
  std::uniform_int_distribution<uint32_t> measure_dist(1, 9);

  for (uint32_t i = 0; i < options.fact_size; ++i) {
    const uint32_t key = key_dist(rng);
    const uint32_t measure = measure_dist(rng);

    sqlite3_bind_int64(fact_insert, 1, static_cast<sqlite3_int64>(key));
    sqlite3_bind_int64(fact_insert, 2, static_cast<sqlite3_int64>(measure));
    rc = sqlite3_step(fact_insert);
    if (rc != SQLITE_DONE) {
      std::cerr << "fact insert failed: " << sqlite3_errmsg(db) << "\n";
      sqlite3_finalize(fact_insert);
      return false;
    }
    sqlite3_reset(fact_insert);
    sqlite3_clear_bindings(fact_insert);
  }

  sqlite3_finalize(fact_insert);

  if (!ExecSql(db, "COMMIT;") ||
      !ExecSql(db, "CREATE INDEX idx_fact_key ON fact(key);") ||
      !ExecSql(db, "ANALYZE;")) {
    return false;
  }

  return true;
}

bool PrintExplainQueryPlan(sqlite3 *db,
                           const std::string &label,
                           const std::string &sql) {
  const std::string explain_sql = "EXPLAIN QUERY PLAN " + sql;
  sqlite3_stmt *statement = nullptr;
  int rc = sqlite3_prepare_v2(db, explain_sql.c_str(), -1, &statement, nullptr);
  if (rc != SQLITE_OK) {
    std::cerr << "Failed to prepare explain query: " << sqlite3_errmsg(db)
              << "\n";
    return false;
  }

  std::cout << "Plan (" << label << "):\n";
  while ((rc = sqlite3_step(statement)) == SQLITE_ROW) {
    const char *detail =
        reinterpret_cast<const char *>(sqlite3_column_text(statement, 3));
    std::cout << "  - " << (detail ? detail : "<null>") << "\n";
  }

  sqlite3_finalize(statement);

  if (rc != SQLITE_DONE) {
    std::cerr << "Explain query failed: " << sqlite3_errmsg(db) << "\n";
    return false;
  }

  return true;
}

bool RunBenchmark(sqlite3 *db,
                  const std::string &sql,
                  uint32_t iterations,
                  double *avg_ms,
                  sqlite3_int64 *result) {
  if (!avg_ms || !result) {
    return false;
  }

  sqlite3_stmt *statement = nullptr;
  int rc = sqlite3_prepare_v2(db, sql.c_str(), -1, &statement, nullptr);
  if (rc != SQLITE_OK) {
    std::cerr << "Failed to prepare benchmark query: " << sqlite3_errmsg(db)
              << "\n";
    return false;
  }

  auto execute_once = [&](sqlite3_int64 *sum, double *elapsed_ms) -> bool {
    const auto start = std::chrono::steady_clock::now();

    rc = sqlite3_step(statement);
    if (rc != SQLITE_ROW) {
      std::cerr << "Expected row result, got rc=" << rc << " ("
                << sqlite3_errmsg(db) << ")\n";
      return false;
    }

    *sum = sqlite3_column_int64(statement, 0);

    rc = sqlite3_step(statement);
    if (rc != SQLITE_DONE) {
      std::cerr << "Expected SQLITE_DONE, got rc=" << rc << " ("
                << sqlite3_errmsg(db) << ")\n";
      return false;
    }

    const auto stop = std::chrono::steady_clock::now();
    *elapsed_ms = std::chrono::duration<double, std::milli>(stop - start).count();

    sqlite3_reset(statement);
    sqlite3_clear_bindings(statement);
    return true;
  };

  sqlite3_int64 warmup_sum = 0;
  double warmup_ms = 0.0;
  if (!execute_once(&warmup_sum, &warmup_ms)) {
    sqlite3_finalize(statement);
    return false;
  }

  double total_ms = 0.0;
  sqlite3_int64 last_sum = warmup_sum;

  for (uint32_t i = 0; i < iterations; ++i) {
    double elapsed_ms = 0.0;
    if (!execute_once(&last_sum, &elapsed_ms)) {
      sqlite3_finalize(statement);
      return false;
    }
    total_ms += elapsed_ms;
  }

  sqlite3_finalize(statement);

  *avg_ms = total_ms / static_cast<double>(iterations);
  *result = last_sum;
  return true;
}

bool CreatePerfectHashVtab(sqlite3 *db,
                           const BenchmarkCase &benchmark_case,
                           std::string *error) {
  if (!ExecSql(db, "DROP TABLE IF EXISTS temp.dim_ph;", error)) {
    return false;
  }

  const std::string create_vtab_sql =
      "CREATE VIRTUAL TABLE temp.dim_ph USING perfecthash(" +
      std::string("'dim','key','value','") + benchmark_case.backend + "','" +
      benchmark_case.hash + "'," + std::to_string(benchmark_case.vector_width) +
      "," + (benchmark_case.strict_vector_width ? "1" : "0") + ");";

  return ExecSql(db, create_vtab_sql, error);
}

void AggregateResult(BenchmarkResult *result, sqlite3_int64 baseline_sum) {
  if (!result) {
    return;
  }

  result->total_runs = static_cast<uint32_t>(result->runs.size());
  result->all_result_match = true;
  result->outcome = "ok";

  std::vector<double> create_samples;
  std::vector<double> query_samples;
  std::vector<double> end_to_end_samples;
  std::vector<double> source_extract_samples;
  std::vector<double> table_create_samples;
  std::vector<double> compile_samples;
  std::vector<double> materialize_samples;
  std::vector<double> internal_create_samples;

  for (const auto &run : result->runs) {
    if (run.create_ok) {
      result->create_success_runs += 1;
      result->any_create_ok = true;
      create_samples.push_back(run.create_ms);
    }

    if (run.have_build_status) {
      if (run.build_status.jit_enabled) {
        result->jit_success_runs += 1;
      } else {
        result->jit_fallback_runs += 1;
      }

      source_extract_samples.push_back(run.source_extract_ms);
      table_create_samples.push_back(run.table_create_ms);
      compile_samples.push_back(run.compile_ms);
      materialize_samples.push_back(run.materialize_ms);
      internal_create_samples.push_back(run.internal_create_ms);
    }

    if (run.query_ok) {
      result->query_success_runs += 1;
      result->any_query_ok = true;
      query_samples.push_back(run.query_ms);
      end_to_end_samples.push_back(run.end_to_end_ms);
      result->representative_sum = run.sum;
    }

    if (!run.result_match && run.query_ok) {
      result->all_result_match = false;
    }

    if (!run.error.empty() && result->error.empty()) {
      result->error = run.error;
    }
  }

  auto populate_stats = [](std::vector<double> *samples,
                           double *avg,
                           double *min,
                           double *max) {
    if (!samples || !avg || !min || !max || samples->empty()) {
      return;
    }

    std::sort(samples->begin(), samples->end());
    *min = samples->front();
    *max = samples->back();
    double sum = 0.0;
    for (double sample : *samples) {
      sum += sample;
    }
    *avg = sum / static_cast<double>(samples->size());
  };

  populate_stats(&create_samples,
                 &result->create_avg_ms,
                 &result->create_min_ms,
                 &result->create_max_ms);
  populate_stats(&query_samples,
                 &result->query_avg_ms,
                 &result->query_min_ms,
                 &result->query_max_ms);
  populate_stats(&end_to_end_samples,
                 &result->end_to_end_avg_ms,
                 &result->end_to_end_min_ms,
                 &result->end_to_end_max_ms);
  populate_stats(&source_extract_samples,
                 &result->source_extract_avg_ms,
                 &result->source_extract_min_ms,
                 &result->source_extract_max_ms);
  populate_stats(&table_create_samples,
                 &result->table_create_avg_ms,
                 &result->table_create_min_ms,
                 &result->table_create_max_ms);
  populate_stats(&compile_samples,
                 &result->compile_avg_ms,
                 &result->compile_min_ms,
                 &result->compile_max_ms);
  populate_stats(&materialize_samples,
                 &result->materialize_avg_ms,
                 &result->materialize_min_ms,
                 &result->materialize_max_ms);
  populate_stats(&internal_create_samples,
                 &result->internal_create_avg_ms,
                 &result->internal_create_min_ms,
                 &result->internal_create_max_ms);

  if (!result->any_create_ok) {
    result->outcome = "create_failed";
  } else if (!result->any_query_ok) {
    result->outcome = "query_failed";
  } else if (!result->all_result_match || result->representative_sum != baseline_sum) {
    result->outcome = "sum_mismatch";
  }
}

BenchmarkResult RunPerfectHashCase(sqlite3 *db,
                                   const BenchmarkCase &benchmark_case,
                                   const std::string &query,
                                   sqlite3_int64 baseline_sum,
                                   uint32_t iterations,
                                   uint32_t build_runs,
                                   bool print_query_plan) {
  BenchmarkResult result;
  result.config = benchmark_case;

  bool printed_plan = false;

  for (uint32_t run_index = 0; run_index < build_runs; ++run_index) {
    BenchmarkRun run;
    run.config = benchmark_case;
    run.run_index = run_index;

    const auto create_start = std::chrono::steady_clock::now();
    if (!CreatePerfectHashVtab(db, benchmark_case, &run.error)) {
      const auto create_end = std::chrono::steady_clock::now();
      run.create_ms =
          std::chrono::duration<double, std::milli>(create_end - create_start)
              .count();
      run.create_ok = false;
      result.runs.push_back(std::move(run));
      continue;
    }

    const auto create_end = std::chrono::steady_clock::now();
    run.create_ms =
        std::chrono::duration<double, std::milli>(create_end - create_start)
            .count();
    run.create_ok = true;

    if (GetLastPerfectHashBuildStatus(&run.build_status)) {
      run.have_build_status = true;
      run.source_extract_ms = run.build_status.source_extract_ms;
      run.table_create_ms = run.build_status.table_create_ms;
      run.compile_ms = run.build_status.compile_ms;
      run.materialize_ms = run.build_status.materialize_ms;
      run.internal_create_ms = run.build_status.internal_total_create_ms;
    }

    if (print_query_plan && !printed_plan) {
      if (!PrintExplainQueryPlan(db,
                                 "perfecthash-" + benchmark_case.backend +
                                     "-" + benchmark_case.hash + "-v" +
                                     std::to_string(benchmark_case.vector_width),
                                 query)) {
        run.error = "Failed to print query plan.";
        result.runs.push_back(std::move(run));
        continue;
      }
      printed_plan = true;
    }

    if (!RunBenchmark(db, query, iterations, &run.query_ms, &run.sum)) {
      run.query_ok = false;
      run.error = "Failed to run benchmark query.";
      result.runs.push_back(std::move(run));
      continue;
    }

    run.query_ok = true;
    run.end_to_end_ms = run.create_ms + run.query_ms;
    run.result_match = (run.sum == baseline_sum);
    if (!run.result_match) {
      run.error = "Result mismatch versus baseline.";
    }

    result.runs.push_back(std::move(run));
  }

  AggregateResult(&result, baseline_sum);
  return result;
}

std::vector<BenchmarkCase> BuildMatrixCases() {
  static const char *kBackends[] = {
      "rawdog-jit",
      "llvm-jit",
  };

  static const char *kHashes[] = {
      "multiplyshiftr",
      "multiplyshiftlr",
      "multiplyshiftrmultiply",
      "multiplyshiftr2",
      "multiplyshiftrx",
      "mulshrolate1rx",
      "mulshrolate2rx",
      "mulshrolate3rx",
      "mulshrolate4rx",
  };

  static const uint32_t kVectorWidths[] = {
      1,
      2,
      4,
      8,
      16,
  };

  std::vector<BenchmarkCase> cases;
  for (const char *backend : kBackends) {
    for (const char *hash : kHashes) {
      for (uint32_t vector_width : kVectorWidths) {
        BenchmarkCase benchmark_case;
        benchmark_case.backend = backend;
        benchmark_case.hash = hash;
        benchmark_case.vector_width = vector_width;
        benchmark_case.strict_vector_width = true;
        cases.push_back(benchmark_case);
      }
    }
  }

  return cases;
}

double QuerySpeedup(double baseline_avg_ms, double query_avg_ms) {
  if (baseline_avg_ms <= 0.0 || query_avg_ms <= 0.0) {
    return 0.0;
  }
  return baseline_avg_ms / query_avg_ms;
}

double EndToEndSpeedup(double baseline_avg_ms, double end_to_end_avg_ms) {
  if (baseline_avg_ms <= 0.0 || end_to_end_avg_ms <= 0.0) {
    return 0.0;
  }
  return baseline_avg_ms / end_to_end_avg_ms;
}

double BreakEvenQueryCount(double baseline_avg_ms,
                           double query_avg_ms,
                           double create_avg_ms) {
  const double per_query_delta_ms = baseline_avg_ms - query_avg_ms;
  if (per_query_delta_ms <= 0.0 || create_avg_ms <= 0.0) {
    return 0.0;
  }
  return create_avg_ms / per_query_delta_ms;
}

void WriteDetailedCsv(const std::string &path,
                      const std::vector<BenchmarkResult> &results) {
  if (path.empty()) {
    return;
  }

  std::ofstream file(path);
  if (!file.is_open()) {
    std::cerr << "Warning: unable to open detailed CSV for writing: " << path
              << "\n";
    return;
  }

  file << "backend,hash,req_vec,strict_vec,run_index,create_ok,query_ok,result_match,"
          "create_ms,query_ms,end_to_end_ms,source_extract_ms,table_create_ms,"
          "compile_ms,materialize_ms,internal_create_ms,sum,key_count,requested_backend,"
          "effective_backend,requested_vector,effective_vector,compile_hr,jit,error\n";

  for (const auto &result : results) {
    for (const auto &run : result.runs) {
      file << run.config.backend << ","
           << run.config.hash << ","
           << run.config.vector_width << ","
           << (run.config.strict_vector_width ? 1 : 0) << ","
           << run.run_index << ","
           << (run.create_ok ? 1 : 0) << ","
           << (run.query_ok ? 1 : 0) << ","
           << (run.result_match ? 1 : 0) << ","
           << std::fixed << std::setprecision(6) << run.create_ms << ","
           << std::fixed << std::setprecision(6) << run.query_ms << ","
           << std::fixed << std::setprecision(6) << run.end_to_end_ms << ","
           << std::fixed << std::setprecision(6) << run.source_extract_ms << ","
           << std::fixed << std::setprecision(6) << run.table_create_ms << ","
           << std::fixed << std::setprecision(6) << run.compile_ms << ","
           << std::fixed << std::setprecision(6) << run.materialize_ms << ","
           << std::fixed << std::setprecision(6) << run.internal_create_ms << ","
           << run.sum << ",";

      if (run.have_build_status) {
        file << run.build_status.key_count << ","
             << CsvEscape(run.build_status.requested_backend) << ","
             << CsvEscape(run.build_status.effective_backend) << ","
             << run.build_status.requested_vector_width << ","
             << run.build_status.effective_vector_width << ","
             << HrToHex(run.build_status.compile_result) << ","
             << (run.build_status.jit_enabled ? 1 : 0) << ",";
      } else {
        file << "0,n/a,n/a,0,0,n/a,0,";
      }

      file << CsvEscape(run.error) << "\n";
    }
  }
}

void WriteSummaryCsv(const std::string &path,
                     const std::vector<BenchmarkResult> &results,
                     double baseline_avg_ms,
                     sqlite3_int64 baseline_sum) {
  if (path.empty()) {
    return;
  }

  std::ofstream file(path);
  if (!file.is_open()) {
    std::cerr << "Warning: unable to open summary CSV for writing: " << path
              << "\n";
    return;
  }

  file << "baseline_avg_ms,baseline_sum,backend,hash,req_vec,strict_vec,"
          "build_runs,create_success_runs,query_success_runs,jit_success_runs,"
          "jit_fallback_runs,source_extract_avg_ms,source_extract_min_ms,source_extract_max_ms,"
          "table_create_avg_ms,table_create_min_ms,table_create_max_ms,"
          "compile_avg_ms,compile_min_ms,compile_max_ms,"
          "materialize_avg_ms,materialize_min_ms,materialize_max_ms,"
          "internal_create_avg_ms,internal_create_min_ms,internal_create_max_ms,"
          "create_avg_ms,create_min_ms,create_max_ms,"
          "query_avg_ms,query_min_ms,query_max_ms,"
          "end_to_end_avg_ms,end_to_end_min_ms,end_to_end_max_ms,"
          "query_speedup,end_to_end_speedup,break_even_queries,outcome,error\n";

  for (const auto &result : results) {
    const double query_speedup = QuerySpeedup(baseline_avg_ms, result.query_avg_ms);
    const double end_to_end_speedup =
        EndToEndSpeedup(baseline_avg_ms, result.end_to_end_avg_ms);
    const double break_even_queries =
        BreakEvenQueryCount(baseline_avg_ms,
                            result.query_avg_ms,
                            result.create_avg_ms);

    file << std::fixed << std::setprecision(6)
         << baseline_avg_ms << ","
         << baseline_sum << ","
         << result.config.backend << ","
         << result.config.hash << ","
         << result.config.vector_width << ","
         << (result.config.strict_vector_width ? 1 : 0) << ","
         << result.total_runs << ","
         << result.create_success_runs << ","
         << result.query_success_runs << ","
         << result.jit_success_runs << ","
         << result.jit_fallback_runs << ","
         << result.source_extract_avg_ms << ","
         << result.source_extract_min_ms << ","
         << result.source_extract_max_ms << ","
         << result.table_create_avg_ms << ","
         << result.table_create_min_ms << ","
         << result.table_create_max_ms << ","
         << result.compile_avg_ms << ","
         << result.compile_min_ms << ","
         << result.compile_max_ms << ","
         << result.materialize_avg_ms << ","
         << result.materialize_min_ms << ","
         << result.materialize_max_ms << ","
         << result.internal_create_avg_ms << ","
         << result.internal_create_min_ms << ","
         << result.internal_create_max_ms << ","
         << result.create_avg_ms << ","
         << result.create_min_ms << ","
         << result.create_max_ms << ","
         << result.query_avg_ms << ","
         << result.query_min_ms << ","
         << result.query_max_ms << ","
         << result.end_to_end_avg_ms << ","
         << result.end_to_end_min_ms << ","
         << result.end_to_end_max_ms << ","
         << query_speedup << ","
         << end_to_end_speedup << ","
         << break_even_queries << ","
         << result.outcome << ","
         << CsvEscape(result.error)
         << "\n";
  }
}

void PrintSingleResult(const Options &options,
                       const BenchmarkResult &result,
                       double baseline_avg_ms,
                       sqlite3_int64 baseline_sum) {
  std::cout << std::fixed << std::setprecision(3);
  std::cout << "\nBenchmark configuration:\n";
  std::cout << "  mode=single\n";
  std::cout << "  backend=" << options.backend << "\n";
  std::cout << "  hash=" << options.hash << "\n";
  std::cout << "  vector-width=" << options.vector_width << "\n";
  std::cout << "  strict-vector-width=" << (options.strict_vector_width ? 1 : 0)
            << "\n";
  std::cout << "  build-runs=" << options.build_runs << "\n";
  std::cout << "  dim-size=" << options.dim_size << "\n";
  std::cout << "  fact-size=" << options.fact_size << "\n";
  std::cout << "  iterations=" << options.iterations << "\n";

  std::cout << "\nResults:\n";
  std::cout << "  baseline avg ms:          " << baseline_avg_ms << "\n";
  std::cout << "  create avg/min/max ms:    " << result.create_avg_ms << " / "
            << result.create_min_ms << " / " << result.create_max_ms << "\n";
  std::cout << "  table-create avg/min/max: " << result.table_create_avg_ms
            << " / " << result.table_create_min_ms << " / "
            << result.table_create_max_ms << "\n";
  std::cout << "  compile avg/min/max ms:   " << result.compile_avg_ms << " / "
            << result.compile_min_ms << " / " << result.compile_max_ms << "\n";
  std::cout << "  query avg/min/max ms:     " << result.query_avg_ms << " / "
            << result.query_min_ms << " / " << result.query_max_ms << "\n";
  std::cout << "  end-to-end avg/min/max:   " << result.end_to_end_avg_ms
            << " / " << result.end_to_end_min_ms << " / "
            << result.end_to_end_max_ms << "\n";

  if (result.outcome != "ok") {
    std::cout << "  perfecthash status:       " << result.outcome << "\n";
    if (!result.error.empty()) {
      std::cout << "  error:                    " << result.error << "\n";
    }
    return;
  }

  const double query_speedup = QuerySpeedup(baseline_avg_ms, result.query_avg_ms);
  const double end_to_end_speedup =
      EndToEndSpeedup(baseline_avg_ms, result.end_to_end_avg_ms);
  const double break_even_queries =
      BreakEvenQueryCount(baseline_avg_ms,
                          result.query_avg_ms,
                          result.create_avg_ms);

  std::cout << "  speedup (query-only):     " << query_speedup << "x\n";
  std::cout << "  speedup (with create):    " << end_to_end_speedup << "x\n";
  if (break_even_queries > 0.0) {
    std::cout << "  break-even queries:       " << break_even_queries << "\n";
  } else {
    std::cout << "  break-even queries:       n/a\n";
  }
  std::cout << "  jit success/fallback:     " << result.jit_success_runs << " / "
            << result.jit_fallback_runs << "\n";
  std::cout << "  baseline sum:             " << baseline_sum << "\n";
  std::cout << "  perfecthash sum:          " << result.representative_sum << "\n";
}

bool PrintMatrixSummary(const std::vector<BenchmarkResult> &results,
                        double baseline_avg_ms,
                        sqlite3_int64 baseline_sum) {
  bool all_ok = true;

  std::cout << std::fixed << std::setprecision(3);
  std::cout << "\nMatrix Summary (baseline avg ms=" << baseline_avg_ms
            << ", baseline sum=" << baseline_sum << ")\n";
  std::cout << "backend,hash,req_vec,build_runs,create_avg_ms,table_create_avg_ms,"
               "compile_avg_ms,query_avg_ms,end_to_end_avg_ms,query_speedup,"
               "end_to_end_speedup,break_even_queries,jit_success,jit_fallback,result\n";

  for (const auto &result : results) {
    const double query_speedup = QuerySpeedup(baseline_avg_ms, result.query_avg_ms);
    const double end_to_end_speedup =
        EndToEndSpeedup(baseline_avg_ms, result.end_to_end_avg_ms);
    const double break_even_queries =
        BreakEvenQueryCount(baseline_avg_ms,
                            result.query_avg_ms,
                            result.create_avg_ms);

    if (result.outcome != "ok") {
      all_ok = false;
    }

    std::cout << result.config.backend << ","
              << result.config.hash << ","
              << result.config.vector_width << ","
              << result.total_runs << ","
              << result.create_avg_ms << ","
              << result.table_create_avg_ms << ","
              << result.compile_avg_ms << ","
              << result.query_avg_ms << ","
              << result.end_to_end_avg_ms << ","
              << query_speedup << ","
              << end_to_end_speedup << ","
              << break_even_queries << ","
              << result.jit_success_runs << ","
              << result.jit_fallback_runs << ","
              << result.outcome << "\n";

    if (!result.error.empty()) {
      std::cout << "  note," << result.config.backend << ","
                << result.config.hash << ","
                << result.config.vector_width << ","
                << result.error << "\n";
    }
  }

  return all_ok;
}

}  // namespace

int main(int argc, char **argv) {
  Options options;
  if (!ParseArgs(argc, argv, &options)) {
    PrintUsage(argv[0]);
    return 2;
  }

  if (options.mode == RunMode::Matrix) {
    PreloadLlvmRuntimeLibrary("llvm-jit");
  } else {
    PreloadLlvmRuntimeLibrary(options.backend);
  }

  sqlite3 *db = nullptr;
  int rc = sqlite3_open(":memory:", &db);
  if (rc != SQLITE_OK) {
    std::cerr << "Failed to open sqlite memory db: "
              << (db ? sqlite3_errmsg(db) : "unknown") << "\n";
    if (db) {
      sqlite3_close(db);
    }
    return 1;
  }

  if (!ExecSql(db, "PRAGMA journal_mode=OFF;") ||
      !ExecSql(db, "PRAGMA synchronous=OFF;") ||
      !ExecSql(db, "PRAGMA temp_store=MEMORY;") ||
      !PopulateTables(db, options)) {
    sqlite3_close(db);
    return 1;
  }

  rc = RegisterPerfectHashModule(db);
  if (rc != SQLITE_OK) {
    std::cerr << "Failed to register perfecthash sqlite module: "
              << sqlite3_errmsg(db) << "\n";
    sqlite3_close(db);
    return 1;
  }

  const std::string baseline_query =
      "SELECT SUM(f.measure * d.value) "
      "FROM fact AS f "
      "JOIN dim AS d ON d.key = f.key;";

  const std::string perfecthash_query =
      "SELECT SUM(f.measure * p.value) "
      "FROM fact AS f "
      "JOIN dim_ph AS p ON p.key = f.key;";

  if (!PrintExplainQueryPlan(db, "baseline-btree", baseline_query)) {
    sqlite3_close(db);
    return 1;
  }

  sqlite3_int64 baseline_sum = 0;
  double baseline_avg_ms = 0.0;
  if (!RunBenchmark(db,
                    baseline_query,
                    options.iterations,
                    &baseline_avg_ms,
                    &baseline_sum)) {
    sqlite3_close(db);
    return 1;
  }

  bool ok = true;
  std::vector<BenchmarkResult> all_results;

  if (options.mode == RunMode::Single) {
    BenchmarkCase benchmark_case;
    benchmark_case.backend = options.backend;
    benchmark_case.hash = options.hash;
    benchmark_case.vector_width = options.vector_width;
    benchmark_case.strict_vector_width = options.strict_vector_width;

    BenchmarkResult result = RunPerfectHashCase(db,
                                                benchmark_case,
                                                perfecthash_query,
                                                baseline_sum,
                                                options.iterations,
                                                options.build_runs,
                                                true);

    PrintSingleResult(options, result, baseline_avg_ms, baseline_sum);
    all_results.push_back(result);

    ok = (result.outcome == "ok");
  } else {
    std::vector<BenchmarkCase> cases = BuildMatrixCases();
    all_results.reserve(cases.size());

    bool printed_first_plan = false;

    for (const auto &benchmark_case : cases) {
      BenchmarkResult result = RunPerfectHashCase(db,
                                                  benchmark_case,
                                                  perfecthash_query,
                                                  baseline_sum,
                                                  options.iterations,
                                                  options.build_runs,
                                                  !printed_first_plan);
      printed_first_plan = true;
      all_results.push_back(result);
    }

    ok = PrintMatrixSummary(all_results, baseline_avg_ms, baseline_sum);
  }

  WriteDetailedCsv(options.output_detailed_csv, all_results);
  WriteSummaryCsv(options.output_summary_csv,
                  all_results,
                  baseline_avg_ms,
                  baseline_sum);

  sqlite3_close(db);
  return ok ? 0 : 1;
}
