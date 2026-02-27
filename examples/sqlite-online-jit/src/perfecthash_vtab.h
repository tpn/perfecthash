#pragma once

#include <sqlite3.h>

#include <cstdint>
#include <string>

struct PerfectHashBuildStatus {
  std::string requested_backend;
  std::string effective_backend;
  std::string hash;
  uint32_t requested_vector_width = 0;
  uint32_t effective_vector_width = 0;
  uint32_t key_count = 0;
  int32_t compile_result = 0;
  bool strict_vector_width = false;
  bool jit_enabled = false;
  double source_extract_ms = 0.0;
  double table_create_ms = 0.0;
  double compile_ms = 0.0;
  double materialize_ms = 0.0;
  double internal_total_create_ms = 0.0;
};

int RegisterPerfectHashModule(sqlite3 *db);
bool GetLastPerfectHashBuildStatus(PerfectHashBuildStatus *status);
