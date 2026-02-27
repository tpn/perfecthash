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
#include <string>

#if !defined(_WIN32) && defined(PH_ONLINE_JIT_LLVM_LIBRARY_PATH)
#include <dlfcn.h>
#endif

namespace {

struct Options {
  std::string backend = "rawdog-jit";
  std::string hash = "mulshrolate2rx";
  uint32_t vector_width = 16;
  uint32_t dim_size = 50'000;
  uint32_t fact_size = 1'000'000;
  uint32_t iterations = 5;
  uint32_t seed = 1;
};

void PrintUsage(const char *argv0) {
  std::cout
      << "Usage: " << argv0
      << " [--backend <rawdog-jit|llvm-jit|auto>]"
         " [--hash <name>]"
         " [--vector-width <0|1|2|4|8|16>]"
         " [--dim-size <count>]"
         " [--fact-size <count>]"
         " [--iterations <count>]"
         " [--seed <value>]\n";
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

    if (std::strcmp(arg, "--backend") == 0) {
      const char *value = require_value("--backend");
      if (!value) {
        return false;
      }
      options->backend = value;
      continue;
    }

    if (std::strcmp(arg, "--hash") == 0) {
      const char *value = require_value("--hash");
      if (!value) {
        return false;
      }
      options->hash = value;
      continue;
    }

    if (std::strcmp(arg, "--vector-width") == 0) {
      const char *value = require_value("--vector-width");
      if (!value || !ParseUint32(value, &options->vector_width)) {
        std::cerr << "Invalid vector width.\n";
        return false;
      }
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

bool ExecSql(sqlite3 *db, const std::string &sql) {
  char *error = nullptr;
  const int rc = sqlite3_exec(db, sql.c_str(), nullptr, nullptr, &error);
  if (rc != SQLITE_OK) {
    std::cerr << "SQL failed: " << sql << "\n";
    std::cerr << "sqlite error: " << (error ? error : sqlite3_errmsg(db))
              << "\n";
    sqlite3_free(error);
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

}  // namespace

int main(int argc, char **argv) {
  Options options;
  if (!ParseArgs(argc, argv, &options)) {
    PrintUsage(argv[0]);
    return 2;
  }

  PreloadLlvmRuntimeLibrary(options.backend);

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

  const std::string create_vtab_sql =
      "CREATE VIRTUAL TABLE temp.dim_ph USING perfecthash(" +
      std::string("'dim','key','value','") + options.backend + "','" +
      options.hash + "'," + std::to_string(options.vector_width) + ");";

  if (!ExecSql(db, create_vtab_sql)) {
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

  if (!PrintExplainQueryPlan(db, "baseline-btree", baseline_query) ||
      !PrintExplainQueryPlan(db,
                             "perfecthash-" + options.backend,
                             perfecthash_query)) {
    sqlite3_close(db);
    return 1;
  }

  sqlite3_int64 baseline_sum = 0;
  sqlite3_int64 perfecthash_sum = 0;
  double baseline_avg_ms = 0.0;
  double perfecthash_avg_ms = 0.0;

  if (!RunBenchmark(db,
                    baseline_query,
                    options.iterations,
                    &baseline_avg_ms,
                    &baseline_sum) ||
      !RunBenchmark(db,
                    perfecthash_query,
                    options.iterations,
                    &perfecthash_avg_ms,
                    &perfecthash_sum)) {
    sqlite3_close(db);
    return 1;
  }

  if (baseline_sum != perfecthash_sum) {
    std::cerr << "Result mismatch: baseline=" << baseline_sum
              << ", perfecthash=" << perfecthash_sum << "\n";
    sqlite3_close(db);
    return 1;
  }

  const double speedup = baseline_avg_ms / perfecthash_avg_ms;

  std::cout << std::fixed << std::setprecision(3);
  std::cout << "\nBenchmark configuration:\n";
  std::cout << "  backend=" << options.backend << "\n";
  std::cout << "  hash=" << options.hash << "\n";
  std::cout << "  vector-width=" << options.vector_width << "\n";
  std::cout << "  dim-size=" << options.dim_size << "\n";
  std::cout << "  fact-size=" << options.fact_size << "\n";
  std::cout << "  iterations=" << options.iterations << "\n";

  std::cout << "\nResults:\n";
  std::cout << "  baseline avg ms:     " << baseline_avg_ms << "\n";
  std::cout << "  perfecthash avg ms:  " << perfecthash_avg_ms << "\n";
  std::cout << "  speedup (baseline/perfecthash): " << speedup << "x\n";

  sqlite3_close(db);
  return 0;
}
