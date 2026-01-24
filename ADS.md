# ADS Notes

Context
- Table-size ADS (keys:Algorithm_Hash_Mask.TableSize) is only used to seed
  RequestedNumberOfTableElements for the current run. If missing, creation
  still works; the requested size stays 0 and the solver proceeds normally.
- Table-info ADS (table.pht1:Info) is required for Table->Vtbl->Load() and
  for downstream generators/tests that read Table->TableInfoOnDisk.

Key locations
- Table-size load/save: `src/PerfectHash/PerfectHashKeysLoadTableSize.c`
- Table-info stream save: `src/PerfectHash/Chm01FileWorkTableInfoStream.c`
- Table-info load: `src/PerfectHash/PerfectHashTableLoad.c`

Implications
- Dropping table-size ADS is low risk if we accept losing the "reuse previous
  table size" hint. It is already optional from a correctness standpoint.
- Dropping table-info ADS breaks load-from-disk and any tools that read
  TableInfoOnDisk (CSV outputs, generators, tests). A replacement on-disk
  metadata format would be required first.

Notes
- ReFS/ADS issues observed: zero-length TableSize streams caused
  PH_E_INVALID_END_OF_FILE. Fixed by extending empty existing streams when
  NoTruncate is set.
- If we want to avoid ADS entirely, table-size is a good first candidate to
  move to a sidecar file (e.g., .TableSize) without affecting core table load.
