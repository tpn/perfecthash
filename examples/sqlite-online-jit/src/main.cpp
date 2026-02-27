#include "perfecthash_vtab.h"

#include <sqlite3.h>

#include <chrono>
#include <cerrno>
#include <cstdint>
#include <cstdlib>
#include <cstring>
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
};

struct BenchmarkCase {
  std::string backend;
  std::string hash;
  uint32_t vector_width = 16;
  bool strict_vector_width = false;
};

struct BenchmarkResult {
  BenchmarkCase config;
  bool create_ok = false;
  bool query_ok = false;
  bool result_match = false;
  bool have_build_status = false;
  PerfectHashBuildStatus build_status;
  sqlite3_int64 sum = 0;
  double avg_ms = 0.0;
  std::string error;
};

std::string HrToHex(int32_t hr) {
  std::ostringstream stream;
  stream << "0x" << std::hex << std::uppercase << static_cast<uint32_t>(hr);
  return stream.str();
}

void PrintUsage(const char *argv0) {
  std::cout
      << "Usage: " << argv0
      << " [--matrix|--single]"
         " [--backend <rawdog-jit|llvm-jit|auto>]"
         " [--hash <name>]"
         " [--vector-width <0|1|2|4|8|16>]"
         " [--strict-vector-width <0|1>]"
         " [--dim-size <count>]"
         " [--fact-size <count>]"
         " [--iterations <count>]"
         " [--seed <value>]\n"
      << "\n"
      << "Default behavior: run a full matrix across RawDog-JIT + LLVM-JIT,\n"
      << "all curated hash functions, and vector widths 1/2/4/8/16.\n"
      << "Passing --backend/--hash/--vector-width without --matrix forces single mode.\n";
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

BenchmarkResult RunPerfectHashCase(sqlite3 *db,
                                   const BenchmarkCase &benchmark_case,
                                   const std::string &query,
                                   sqlite3_int64 baseline_sum,
                                   uint32_t iterations,
                                   bool print_query_plan) {
  BenchmarkResult result;
  result.config = benchmark_case;

  if (!CreatePerfectHashVtab(db, benchmark_case, &result.error)) {
    result.create_ok = false;
    return result;
  }

  result.create_ok = true;

  if (GetLastPerfectHashBuildStatus(&result.build_status)) {
    result.have_build_status = true;
  }

  if (print_query_plan) {
    if (!PrintExplainQueryPlan(db,
                               "perfecthash-" + benchmark_case.backend +
                                   "-" + benchmark_case.hash + "-v" +
                                   std::to_string(benchmark_case.vector_width),
                               query)) {
      result.query_ok = false;
      result.error = "Failed to print query plan.";
      return result;
    }
  }

  if (!RunBenchmark(db, query, iterations, &result.avg_ms, &result.sum)) {
    result.query_ok = false;
    result.error = "Failed to run benchmark query.";
    return result;
  }

  result.query_ok = true;
  result.result_match = (result.sum == baseline_sum);
  if (!result.result_match) {
    result.error = "Result mismatch versus baseline.";
  }

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
  std::cout << "  dim-size=" << options.dim_size << "\n";
  std::cout << "  fact-size=" << options.fact_size << "\n";
  std::cout << "  iterations=" << options.iterations << "\n";

  std::cout << "\nResults:\n";
  std::cout << "  baseline avg ms:     " << baseline_avg_ms << "\n";

  if (!result.create_ok || !result.query_ok || !result.result_match) {
    std::cout << "  perfecthash status:  failed\n";
    if (!result.error.empty()) {
      std::cout << "  error:               " << result.error << "\n";
    }
    return;
  }

  const double speedup = baseline_avg_ms / result.avg_ms;
  std::cout << "  perfecthash avg ms:  " << result.avg_ms << "\n";
  std::cout << "  speedup (baseline/perfecthash): " << speedup << "x\n";

  if (result.have_build_status) {
    std::cout << "  compile hr:          "
              << HrToHex(result.build_status.compile_result) << "\n";
    std::cout << "  requested backend:   "
              << result.build_status.requested_backend << "\n";
    std::cout << "  effective backend:   "
              << result.build_status.effective_backend << "\n";
    std::cout << "  requested vector:    "
              << result.build_status.requested_vector_width << "\n";
    std::cout << "  effective vector:    "
              << result.build_status.effective_vector_width << "\n";
    std::cout << "  jit enabled:         "
              << (result.build_status.jit_enabled ? 1 : 0) << "\n";
  }

  std::cout << "  baseline sum:        " << baseline_sum << "\n";
  std::cout << "  perfecthash sum:     " << result.sum << "\n";
}

bool PrintMatrixSummary(const std::vector<BenchmarkResult> &results,
                        double baseline_avg_ms,
                        sqlite3_int64 baseline_sum) {
  bool all_ok = true;

  std::cout << std::fixed << std::setprecision(3);
  std::cout << "\nMatrix Summary (baseline avg ms=" << baseline_avg_ms
            << ", baseline sum=" << baseline_sum << ")\n";
  std::cout << "backend,hash,req_vec,eff_backend,eff_vec,jit,compile_hr,avg_ms,speedup,result\n";

  for (const auto &result : results) {
    std::string outcome = "ok";
    std::string compile_hr = "n/a";
    std::string effective_backend = "n/a";
    uint32_t effective_vector = 0;
    int jit_enabled = 0;

    if (!result.create_ok) {
      outcome = "create_failed";
      all_ok = false;
    } else if (!result.query_ok) {
      outcome = "query_failed";
      all_ok = false;
    } else if (!result.result_match) {
      outcome = "sum_mismatch";
      all_ok = false;
    }

    if (result.have_build_status) {
      compile_hr = HrToHex(result.build_status.compile_result);
      effective_backend = result.build_status.effective_backend;
      effective_vector = result.build_status.effective_vector_width;
      jit_enabled = result.build_status.jit_enabled ? 1 : 0;
    }

    const double speedup =
        (result.avg_ms > 0.0) ? (baseline_avg_ms / result.avg_ms) : 0.0;

    std::cout << result.config.backend << ","
              << result.config.hash << ","
              << result.config.vector_width << ","
              << effective_backend << ","
              << effective_vector << ","
              << jit_enabled << ","
              << compile_hr << ","
              << result.avg_ms << ","
              << speedup << ","
              << outcome << "\n";

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

  if (options.mode == RunMode::Single) {
    BenchmarkCase benchmark_case;
    benchmark_case.backend = options.backend;
    benchmark_case.hash = options.hash;
    benchmark_case.vector_width = options.vector_width;
    benchmark_case.strict_vector_width = options.strict_vector_width;

    const BenchmarkResult result = RunPerfectHashCase(db,
                                                      benchmark_case,
                                                      perfecthash_query,
                                                      baseline_sum,
                                                      options.iterations,
                                                      true);

    PrintSingleResult(options, result, baseline_avg_ms, baseline_sum);

    ok = result.create_ok && result.query_ok && result.result_match;
  } else {
    std::vector<BenchmarkCase> cases = BuildMatrixCases();
    std::vector<BenchmarkResult> results;
    results.reserve(cases.size());

    bool printed_first_plan = false;

    for (const auto &benchmark_case : cases) {
      const BenchmarkResult result = RunPerfectHashCase(db,
                                                        benchmark_case,
                                                        perfecthash_query,
                                                        baseline_sum,
                                                        options.iterations,
                                                        !printed_first_plan);
      printed_first_plan = true;
      results.push_back(result);
    }

    ok = PrintMatrixSummary(results, baseline_avg_ms, baseline_sum);
  }

  sqlite3_close(db);
  return ok ? 0 : 1;
}
