#include "perfecthash_vtab.h"

#include <PerfectHash/PerfectHashOnlineJit.h>

#include <algorithm>
#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <new>
#include <sstream>
#include <string>
#include <vector>

namespace {

constexpr int32_t kPhNotImplemented = static_cast<int32_t>(0xE0040230u);
constexpr int32_t kPhLlvmBackendNotFound = static_cast<int32_t>(0xE004041Cu);

PerfectHashBuildStatus g_last_build_status;
bool g_has_last_build_status = false;

struct PerfectHashVtab {
  sqlite3_vtab base{};
  sqlite3 *db = nullptr;

  std::string source_table;
  std::string key_column;
  std::string value_column;
  std::string backend_name = "rawdog-jit";
  std::string hash_name = "mulshrolate2rx";

  PH_ONLINE_JIT_BACKEND backend = PhOnlineJitBackendRawDogJit;
  PH_ONLINE_JIT_HASH_FUNCTION hash = PhOnlineJitHashMulshrolate2RX;
  uint32_t vector_width = 16;
  bool strict_vector_width = false;

  bool jit_enabled = false;
  int32_t compile_result = 0;
  PH_ONLINE_JIT_BACKEND effective_backend = PhOnlineJitBackendRawDogJit;
  uint32_t effective_vector_width = 16;

  PH_ONLINE_JIT_CONTEXT *context = nullptr;
  PH_ONLINE_JIT_TABLE *table = nullptr;

  std::vector<uint32_t> scan_keys;
  std::vector<sqlite3_int64> scan_values;

  std::vector<uint32_t> keys_by_index;
  std::vector<sqlite3_int64> values_by_index;
};

struct PerfectHashCursor {
  sqlite3_vtab_cursor base{};
  PerfectHashVtab *vtab = nullptr;

  bool eof = true;
  bool point_lookup = false;
  sqlite3_int64 scan_position = 0;

  uint32_t current_key = 0;
  sqlite3_int64 current_value = 0;
};

std::string HrToHex(int32_t hr) {
  std::ostringstream stream;
  stream << "0x" << std::hex << std::uppercase
         << static_cast<uint32_t>(hr);
  return stream.str();
}

const char *BackendToString(PH_ONLINE_JIT_BACKEND backend) {
  switch (backend) {
    case PhOnlineJitBackendAuto:
      return "auto";
    case PhOnlineJitBackendRawDogJit:
      return "rawdog-jit";
    case PhOnlineJitBackendLlvmJit:
      return "llvm-jit";
    default:
      return "unknown";
  }
}

const char *HashToString(PH_ONLINE_JIT_HASH_FUNCTION hash) {
  switch (hash) {
    case PhOnlineJitHashMultiplyShiftR:
      return "multiplyshiftr";
    case PhOnlineJitHashMultiplyShiftLR:
      return "multiplyshiftlr";
    case PhOnlineJitHashMultiplyShiftRMultiply:
      return "multiplyshiftrmultiply";
    case PhOnlineJitHashMultiplyShiftR2:
      return "multiplyshiftr2";
    case PhOnlineJitHashMultiplyShiftRX:
      return "multiplyshiftrx";
    case PhOnlineJitHashMulshrolate1RX:
      return "mulshrolate1rx";
    case PhOnlineJitHashMulshrolate2RX:
      return "mulshrolate2rx";
    case PhOnlineJitHashMulshrolate3RX:
      return "mulshrolate3rx";
    case PhOnlineJitHashMulshrolate4RX:
      return "mulshrolate4rx";
    default:
      return "unknown";
  }
}

std::string NormalizeArgument(const char *arg) {
  if (!arg) {
    return {};
  }

  std::string text(arg);
  if (text.size() >= 2) {
    const char first = text.front();
    const char last = text.back();

    if ((first == '\'' && last == '\'') ||
        (first == '"' && last == '"') ||
        (first == '`' && last == '`')) {
      text = text.substr(1, text.size() - 2);
    }
  }

  return text;
}

bool ParseUInt32(const std::string &text, uint32_t *value) {
  if (!value) {
    return false;
  }

  if (text.empty()) {
    return false;
  }

  char *end = nullptr;
  errno = 0;
  const unsigned long parsed = strtoul(text.c_str(), &end, 10);
  if (errno != 0 || !end || *end != '\0' ||
      parsed > std::numeric_limits<uint32_t>::max()) {
    return false;
  }

  *value = static_cast<uint32_t>(parsed);
  return true;
}

PH_ONLINE_JIT_BACKEND ParseBackend(const std::string &name) {
  if (name == "rawdog-jit") {
    return PhOnlineJitBackendRawDogJit;
  }
  if (name == "llvm-jit") {
    return PhOnlineJitBackendLlvmJit;
  }
  if (name == "auto") {
    return PhOnlineJitBackendAuto;
  }

  return PhOnlineJitBackendRawDogJit;
}

bool ParseHashFunction(const std::string &name,
                       PH_ONLINE_JIT_HASH_FUNCTION *hash) {
  if (!hash) {
    return false;
  }

  if (name == "multiplyshiftr") {
    *hash = PhOnlineJitHashMultiplyShiftR;
    return true;
  }
  if (name == "multiplyshiftrx") {
    *hash = PhOnlineJitHashMultiplyShiftRX;
    return true;
  }
  if (name == "mulshrolate1rx") {
    *hash = PhOnlineJitHashMulshrolate1RX;
    return true;
  }
  if (name == "mulshrolate2rx") {
    *hash = PhOnlineJitHashMulshrolate2RX;
    return true;
  }
  if (name == "mulshrolate3rx") {
    *hash = PhOnlineJitHashMulshrolate3RX;
    return true;
  }
  if (name == "mulshrolate4rx") {
    *hash = PhOnlineJitHashMulshrolate4RX;
    return true;
  }

  return false;
}

void SetVtabError(sqlite3_vtab *vtab, const std::string &message) {
  if (!vtab) {
    return;
  }

  sqlite3_free(vtab->zErrMsg);
  vtab->zErrMsg = sqlite3_mprintf("%s", message.c_str());
}

void ReleasePerfectHashResources(PerfectHashVtab *vtab) {
  if (!vtab) {
    return;
  }

  if (vtab->table) {
    PhOnlineJitReleaseTable(vtab->table);
    vtab->table = nullptr;
  }

  if (vtab->context) {
    PhOnlineJitClose(vtab->context);
    vtab->context = nullptr;
  }

  vtab->scan_keys.clear();
  vtab->scan_values.clear();
  vtab->keys_by_index.clear();
  vtab->values_by_index.clear();
  vtab->jit_enabled = false;
  vtab->compile_result = 0;
  vtab->effective_backend = vtab->backend;
  vtab->effective_vector_width = vtab->vector_width;
}

void UpdateLastBuildStatus(const PerfectHashVtab *vtab) {
  if (!vtab) {
    return;
  }

  g_last_build_status.requested_backend = vtab->backend_name;
  g_last_build_status.effective_backend = BackendToString(vtab->effective_backend);
  g_last_build_status.hash = vtab->hash_name;
  g_last_build_status.requested_vector_width = vtab->vector_width;
  g_last_build_status.effective_vector_width = vtab->effective_vector_width;
  g_last_build_status.compile_result = vtab->compile_result;
  g_last_build_status.strict_vector_width = vtab->strict_vector_width;
  g_last_build_status.jit_enabled = vtab->jit_enabled;

  g_has_last_build_status = true;
}

bool TryLookup(PerfectHashVtab *vtab, uint32_t key, sqlite3_int64 *value) {
  if (!vtab || !vtab->table || !value) {
    return false;
  }

  uint32_t index = 0;
  const int32_t hr = PhOnlineJitIndex32(vtab->table, key, &index);
  if (hr < 0) {
    return false;
  }

  if (index >= vtab->keys_by_index.size()) {
    return false;
  }

  if (vtab->keys_by_index[index] != key) {
    return false;
  }

  *value = vtab->values_by_index[index];
  return true;
}

bool TryConvertSqlValueToU32(sqlite3_value *value, uint32_t *result) {
  if (!value || !result) {
    return false;
  }

  if (sqlite3_value_type(value) == SQLITE_NULL) {
    return false;
  }

  const sqlite3_int64 signed_value = sqlite3_value_int64(value);
  if (signed_value < 0 ||
      signed_value > static_cast<sqlite3_int64>(
                         std::numeric_limits<uint32_t>::max())) {
    return false;
  }

  *result = static_cast<uint32_t>(signed_value);
  return true;
}

int BuildPerfectHashIndex(PerfectHashVtab *vtab, sqlite3_vtab *base_vtab) {
  if (!vtab || !base_vtab) {
    return SQLITE_ERROR;
  }

  ReleasePerfectHashResources(vtab);

  char *query = sqlite3_mprintf(
      "SELECT \"%w\", \"%w\" FROM \"%w\" ORDER BY \"%w\";",
      vtab->key_column.c_str(),
      vtab->value_column.c_str(),
      vtab->source_table.c_str(),
      vtab->key_column.c_str());
  if (!query) {
    SetVtabError(base_vtab, "Unable to allocate sqlite source query.");
    return SQLITE_NOMEM;
  }

  sqlite3_stmt *statement = nullptr;
  int rc = sqlite3_prepare_v2(vtab->db, query, -1, &statement, nullptr);
  sqlite3_free(query);

  if (rc != SQLITE_OK) {
    SetVtabError(base_vtab,
                 "Failed to prepare sqlite source query: " +
                     std::string(sqlite3_errmsg(vtab->db)));
    return rc;
  }

  std::vector<uint32_t> keys;
  std::vector<sqlite3_int64> values;
  bool have_previous_key = false;
  uint32_t previous_key = 0;

  while ((rc = sqlite3_step(statement)) == SQLITE_ROW) {
    const sqlite3_int64 key_value = sqlite3_column_int64(statement, 0);
    if (key_value < 0 ||
        key_value >
            static_cast<sqlite3_int64>(std::numeric_limits<uint32_t>::max())) {
      sqlite3_finalize(statement);
      SetVtabError(base_vtab,
                   "Source key is outside uint32 range: " +
                       std::to_string(key_value));
      return SQLITE_MISMATCH;
    }

    const uint32_t key = static_cast<uint32_t>(key_value);
    if (have_previous_key && key == previous_key) {
      sqlite3_finalize(statement);
      SetVtabError(base_vtab,
                   "Source key column contains duplicates: " +
                       std::to_string(key));
      return SQLITE_CONSTRAINT;
    }

    have_previous_key = true;
    previous_key = key;

    keys.push_back(key);
    values.push_back(sqlite3_column_int64(statement, 1));
  }

  if (rc != SQLITE_DONE) {
    sqlite3_finalize(statement);
    SetVtabError(base_vtab,
                 "Failed while reading sqlite source rows: " +
                     std::string(sqlite3_errmsg(vtab->db)));
    return rc;
  }

  sqlite3_finalize(statement);
  statement = nullptr;

  if (keys.empty()) {
    SetVtabError(base_vtab,
                 "Source dimension query returned no rows; cannot build "
                 "perfect hash table.");
    return SQLITE_EMPTY;
  }

  int32_t hr = PhOnlineJitOpen(&vtab->context);
  if (hr < 0) {
    SetVtabError(base_vtab,
                 "PhOnlineJitOpen() failed: " + HrToHex(hr));
    return SQLITE_ERROR;
  }

  hr = PhOnlineJitCreateTable32(vtab->context,
                                vtab->hash,
                                keys.data(),
                                static_cast<uint64_t>(keys.size()),
                                &vtab->table);
  if (hr < 0) {
    SetVtabError(base_vtab,
                 "PhOnlineJitCreateTable32() failed: " + HrToHex(hr));
    ReleasePerfectHashResources(vtab);
    return SQLITE_ERROR;
  }

  uint32_t compile_flags = 0;
  if (vtab->strict_vector_width) {
    compile_flags |= PH_ONLINE_JIT_COMPILE_FLAG_STRICT_VECTOR_WIDTH;
  }

  hr = PhOnlineJitCompileTableEx(vtab->context,
                                 vtab->table,
                                 vtab->backend,
                                 vtab->vector_width,
                                 PhOnlineJitMaxIsaAuto,
                                 compile_flags,
                                 &vtab->effective_backend,
                                 &vtab->effective_vector_width);
  vtab->compile_result = hr;

  if (hr < 0) {
    const bool nonfatal_compile_failure =
        (hr == kPhNotImplemented ||
         hr == kPhLlvmBackendNotFound ||
         vtab->strict_vector_width);
    if (nonfatal_compile_failure) {
      vtab->jit_enabled = false;
      if (!vtab->strict_vector_width) {
        std::fprintf(stderr,
                     "[sqlite-online-jit] Warning: JIT backend unavailable "
                     "(%s), continuing with non-JIT index path.\n",
                     HrToHex(hr).c_str());
      }
    } else {
      SetVtabError(base_vtab,
                   "PhOnlineJitCompileTableEx() failed: " + HrToHex(hr));
      UpdateLastBuildStatus(vtab);
      ReleasePerfectHashResources(vtab);
      return SQLITE_ERROR;
    }
  } else {
    vtab->jit_enabled = true;
  }

  vtab->scan_keys = keys;
  vtab->scan_values = values;
  vtab->keys_by_index.assign(keys.size(), std::numeric_limits<uint32_t>::max());
  vtab->values_by_index.assign(values.size(), 0);

  for (size_t index = 0; index < keys.size(); ++index) {
    uint32_t mapped_index = 0;
    hr = PhOnlineJitIndex32(vtab->table, keys[index], &mapped_index);
    if (hr < 0) {
      SetVtabError(base_vtab,
                   "PhOnlineJitIndex32() failed while materializing map: " +
                       HrToHex(hr));
      ReleasePerfectHashResources(vtab);
      return SQLITE_ERROR;
    }

    if (mapped_index >= vtab->keys_by_index.size()) {
      SetVtabError(base_vtab,
                   "PerfectHash index out of bounds while materializing map.");
      ReleasePerfectHashResources(vtab);
      return SQLITE_ERROR;
    }

    if (vtab->keys_by_index[mapped_index] !=
        std::numeric_limits<uint32_t>::max()) {
      SetVtabError(base_vtab,
                   "Detected duplicate PerfectHash output index while "
                   "materializing map.");
      ReleasePerfectHashResources(vtab);
      return SQLITE_CONSTRAINT;
    }

    vtab->keys_by_index[mapped_index] = keys[index];
    vtab->values_by_index[mapped_index] = values[index];
  }

  UpdateLastBuildStatus(vtab);

  return SQLITE_OK;
}

int ParseVtabArguments(int argc,
                       const char *const *argv,
                       PerfectHashVtab *vtab,
                       sqlite3_vtab *base_vtab) {
  if (!vtab || !base_vtab) {
    return SQLITE_ERROR;
  }

  if (argc < 6) {
    SetVtabError(base_vtab,
                 "Usage: perfecthash(source_table,key_column,value_column,"
                 "backend,hash,vector_width,strict_vector_width)");
    return SQLITE_MISUSE;
  }

  vtab->source_table = NormalizeArgument(argv[3]);
  vtab->key_column = NormalizeArgument(argv[4]);
  vtab->value_column = NormalizeArgument(argv[5]);

  const std::string backend =
      (argc >= 7) ? NormalizeArgument(argv[6]) : "rawdog-jit";
  const std::string hash =
      (argc >= 8) ? NormalizeArgument(argv[7]) : "mulshrolate2rx";
  const std::string vector_width =
      (argc >= 9) ? NormalizeArgument(argv[8]) : "16";
  const std::string strict_vector_width =
      (argc >= 10) ? NormalizeArgument(argv[9]) : "0";

  if (vtab->source_table.empty() || vtab->key_column.empty() ||
      vtab->value_column.empty()) {
    SetVtabError(base_vtab,
                 "Source table, key column, and value column are required.");
    return SQLITE_MISUSE;
  }

  vtab->backend = ParseBackend(backend);
  vtab->backend_name = BackendToString(vtab->backend);
  if (!ParseHashFunction(hash, &vtab->hash)) {
    SetVtabError(base_vtab, "Unsupported hash argument: " + hash);
    return SQLITE_MISUSE;
  }
  vtab->hash_name = HashToString(vtab->hash);

  if (!ParseUInt32(vector_width, &vtab->vector_width)) {
    SetVtabError(base_vtab,
                 "Invalid vector width argument: " + vector_width);
    return SQLITE_MISUSE;
  }

  uint32_t strict_width = 0;
  if (!ParseUInt32(strict_vector_width, &strict_width) || strict_width > 1) {
    SetVtabError(base_vtab,
                 "strict_vector_width must be 0 or 1.");
    return SQLITE_MISUSE;
  }
  vtab->strict_vector_width = (strict_width != 0);

  switch (vtab->vector_width) {
    case 0:
    case 1:
    case 2:
    case 4:
    case 8:
    case 16:
      break;
    default:
      SetVtabError(base_vtab,
                   "Vector width must be one of 0,1,2,4,8,16.");
      return SQLITE_MISUSE;
  }

  return SQLITE_OK;
}

int PerfectHashCreateOrConnect(sqlite3 *db,
                               void * /*p_aux*/,
                               int argc,
                               const char *const *argv,
                               sqlite3_vtab **pp_vtab,
                               char ** /*pz_err*/) {
  if (!db || !pp_vtab) {
    return SQLITE_MISUSE;
  }

  auto *vtab = new (std::nothrow) PerfectHashVtab();
  if (!vtab) {
    return SQLITE_NOMEM;
  }

  vtab->db = db;

  int rc = sqlite3_declare_vtab(db, "CREATE TABLE x(key INTEGER, value INTEGER)");
  if (rc != SQLITE_OK) {
    delete vtab;
    return rc;
  }

  rc = ParseVtabArguments(argc, argv, vtab, &vtab->base);
  if (rc != SQLITE_OK) {
    delete vtab;
    return rc;
  }

  rc = BuildPerfectHashIndex(vtab, &vtab->base);
  if (rc != SQLITE_OK) {
    delete vtab;
    return rc;
  }

  *pp_vtab = &vtab->base;
  return SQLITE_OK;
}

int PerfectHashBestIndex(sqlite3_vtab *p_vtab, sqlite3_index_info *p_info) {
  auto *vtab = reinterpret_cast<PerfectHashVtab *>(p_vtab);
  if (!vtab || !p_info) {
    return SQLITE_ERROR;
  }

  int key_eq_constraint = -1;
  for (int index = 0; index < p_info->nConstraint; ++index) {
    const auto &constraint = p_info->aConstraint[index];
    if (constraint.usable && constraint.iColumn == 0 &&
        constraint.op == SQLITE_INDEX_CONSTRAINT_EQ) {
      key_eq_constraint = index;
      break;
    }
  }

  if (key_eq_constraint >= 0) {
    p_info->idxNum = 1;
    p_info->aConstraintUsage[key_eq_constraint].argvIndex = 1;
    p_info->aConstraintUsage[key_eq_constraint].omit = 1;
    p_info->estimatedCost = 1.0;
    p_info->estimatedRows = 1;
    p_info->idxFlags = SQLITE_INDEX_SCAN_UNIQUE;
  } else {
    p_info->idxNum = 0;
    p_info->estimatedCost = static_cast<double>(
        std::max<size_t>(1, vtab->scan_keys.size()));
    p_info->estimatedRows = static_cast<sqlite3_int64>(
        std::max<size_t>(1, vtab->scan_keys.size()));
  }

  return SQLITE_OK;
}

int PerfectHashDisconnect(sqlite3_vtab *p_vtab) {
  auto *vtab = reinterpret_cast<PerfectHashVtab *>(p_vtab);
  if (!vtab) {
    return SQLITE_OK;
  }

  ReleasePerfectHashResources(vtab);
  delete vtab;
  return SQLITE_OK;
}

int PerfectHashOpen(sqlite3_vtab *p_vtab, sqlite3_vtab_cursor **pp_cursor) {
  if (!p_vtab || !pp_cursor) {
    return SQLITE_MISUSE;
  }

  auto *cursor = new (std::nothrow) PerfectHashCursor();
  if (!cursor) {
    return SQLITE_NOMEM;
  }
  cursor->vtab = reinterpret_cast<PerfectHashVtab *>(p_vtab);

  *pp_cursor = &cursor->base;
  return SQLITE_OK;
}

int PerfectHashClose(sqlite3_vtab_cursor *p_cursor) {
  auto *cursor = reinterpret_cast<PerfectHashCursor *>(p_cursor);
  delete cursor;
  return SQLITE_OK;
}

void SetCursorFromScanPosition(PerfectHashCursor *cursor) {
  if (!cursor || !cursor->vtab) {
    return;
  }

  const auto position = static_cast<size_t>(cursor->scan_position);
  if (position >= cursor->vtab->scan_keys.size()) {
    cursor->eof = true;
    return;
  }

  cursor->current_key = cursor->vtab->scan_keys[position];
  cursor->current_value = cursor->vtab->scan_values[position];
  cursor->eof = false;
}

int PerfectHashFilter(sqlite3_vtab_cursor *p_cursor,
                      int idx_num,
                      const char * /*idx_str*/,
                      int argc,
                      sqlite3_value **argv) {
  auto *cursor = reinterpret_cast<PerfectHashCursor *>(p_cursor);
  if (!cursor || !cursor->vtab) {
    return SQLITE_MISUSE;
  }

  cursor->scan_position = 0;
  cursor->point_lookup = false;
  cursor->eof = true;
  cursor->current_key = 0;
  cursor->current_value = 0;

  if (idx_num == 1 && argc == 1) {
    uint32_t key = 0;
    if (!TryConvertSqlValueToU32(argv[0], &key)) {
      cursor->eof = true;
      return SQLITE_OK;
    }

    sqlite3_int64 value = 0;
    if (!TryLookup(cursor->vtab, key, &value)) {
      cursor->eof = true;
      return SQLITE_OK;
    }

    cursor->point_lookup = true;
    cursor->current_key = key;
    cursor->current_value = value;
    cursor->eof = false;
    return SQLITE_OK;
  }

  cursor->scan_position = 0;
  cursor->point_lookup = false;
  SetCursorFromScanPosition(cursor);
  return SQLITE_OK;
}

int PerfectHashNext(sqlite3_vtab_cursor *p_cursor) {
  auto *cursor = reinterpret_cast<PerfectHashCursor *>(p_cursor);
  if (!cursor || !cursor->vtab) {
    return SQLITE_MISUSE;
  }

  if (cursor->point_lookup) {
    cursor->eof = true;
    return SQLITE_OK;
  }

  cursor->scan_position += 1;
  SetCursorFromScanPosition(cursor);
  return SQLITE_OK;
}

int PerfectHashEof(sqlite3_vtab_cursor *p_cursor) {
  auto *cursor = reinterpret_cast<PerfectHashCursor *>(p_cursor);
  return cursor && !cursor->eof ? 0 : 1;
}

int PerfectHashColumn(sqlite3_vtab_cursor *p_cursor,
                      sqlite3_context *context,
                      int column) {
  auto *cursor = reinterpret_cast<PerfectHashCursor *>(p_cursor);
  if (!cursor) {
    return SQLITE_MISUSE;
  }

  if (column == 0) {
    sqlite3_result_int64(context, static_cast<sqlite3_int64>(cursor->current_key));
  } else if (column == 1) {
    sqlite3_result_int64(context, cursor->current_value);
  } else {
    sqlite3_result_null(context);
  }

  return SQLITE_OK;
}

int PerfectHashRowId(sqlite3_vtab_cursor *p_cursor, sqlite3_int64 *row_id) {
  auto *cursor = reinterpret_cast<PerfectHashCursor *>(p_cursor);
  if (!cursor || !row_id) {
    return SQLITE_MISUSE;
  }

  if (cursor->point_lookup) {
    *row_id = static_cast<sqlite3_int64>(cursor->current_key);
  } else {
    *row_id = cursor->scan_position + 1;
  }

  return SQLITE_OK;
}

sqlite3_module kPerfectHashModule = {
    4,                      /* iVersion */
    PerfectHashCreateOrConnect,
    PerfectHashCreateOrConnect,
    PerfectHashBestIndex,
    PerfectHashDisconnect,
    PerfectHashDisconnect,
    PerfectHashOpen,
    PerfectHashClose,
    PerfectHashFilter,
    PerfectHashNext,
    PerfectHashEof,
    PerfectHashColumn,
    PerfectHashRowId,
    nullptr,
    nullptr,
    nullptr,
    nullptr,
    nullptr,
    nullptr,
    nullptr,
    nullptr,
    nullptr,
    nullptr,
    nullptr,
};

}  // namespace

int RegisterPerfectHashModule(sqlite3 *db) {
  if (!db) {
    return SQLITE_MISUSE;
  }

  return sqlite3_create_module_v2(db,
                                  "perfecthash",
                                  &kPerfectHashModule,
                                  nullptr,
                                  nullptr);
}

bool GetLastPerfectHashBuildStatus(PerfectHashBuildStatus *status) {
  if (!status || !g_has_last_build_status) {
    return false;
  }

  *status = g_last_build_status;
  return true;
}
