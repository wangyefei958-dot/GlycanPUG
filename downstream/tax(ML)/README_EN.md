# tax(ML) Data Pipeline Reproduction Guide

This directory converts the raw glycosylation classification table into per-rank training data for downstream GNN training.

## Directory Layout

```
tax(ML)/
├── allstata/                     ← Stage 0: Backfill WURCS from raw data
│   ├── glycan_classification.csv     Raw input (no wurcs column)
│   ├── 0add_wurcs_to_csv.py          Batch API call to convert IUPAC → WURCS
│   ├── test_single_glycan.py         Utility library
│   ├── 1batch_process_errors.py      Retry & repair API-failed rows
│   ├── 2merge_fixed.py               Merge repaired rows back into the main table
│   └── ...intermediate products...
├── glycan_classification_merged.csv  Final Stage 0 artifact (read by 分拆.py at repo root)
├── 分拆.py                       ← Stage 1: Split into 8 taxonomic ranks
└── split_files/                  ← Stage 1 final output
    ├── species.csv
    ├── genus.csv
    └── ... (8 rank CSVs + split_log.txt)
```

## Dependencies

| Stage | Third-party packages | Notes |
|---|---|---|
| Stage 0 (allstata) | `requests` | Calls the glycosmos API |
| Stage 1 (split) | `pandas` | The only analysis dependency |

All scripts target Python 3.13.

---

## Reproduction Steps

### Stage 0: Backfill WURCS (run inside `allstata/`)

**Input**: `allstata/glycan_classification.csv`  
**Header**: `target, species, genus, family, order, class, phylum, kingdom, domain`

#### 0.1 Call the API to convert IUPAC → WURCS

```bash
cd allstata
python add_wurcs_to_csv.py
```

- API endpoint: `https://api.glycosmos.org/glycanformatconverter/2.7.0/iupaccondensed2wurcs`
- 6 concurrent threads, 0.1s per request, 2 retries on failure
- **Produces**:
  - `glycan_classification_with_wurcs.csv` (successful rows, new `wurcs` column)
  - `glycan_classification_errors.csv` (failed rows, `wurcs=ERROR` + `error_reason`)
  - `wurcs_progress.json` (resume checkpoint)

#### 0.2 Repair failed rows

```bash
python batch_process_errors.py
```

- Imports helpers from `test_single_glycan.py` (normalization, protecting substituents, IUPAC↔WURCS round-trip)
- Resumable: already-processed rows are skipped
- **Produces**: `glycan_classification_errors_processed.csv`

#### 0.3 Merge repaired rows back

```bash
python merge_fixed.py
```

- Uses `(target, species)` as the join key to replace `ERROR` entries with repaired WURCS values
- **Produces**: `allstata/glycan_classification_merged.csv`
- **Final header**: `glycan, species, genus, family, order, class, phylum, kingdom, domain, wurcs`  
  (note the first column is renamed from `target` to `glycan`)

#### 0.4 Copy merged.csv to the repo root (required by Stage 1)

```powershell
Copy-Item "allstata\glycan_classification_merged.csv" "."
```

`分拆.py` uses a relative path and looks for `glycan_classification_merged.csv` in its own directory (the repo root), so this copy step is required before running Stage 1.

### Stage 1: Split by taxonomic rank (run at repo root)

```bash
cd ..
python 分拆.py
```

- Reads `glycan_classification_merged.csv` in the current directory
- Eight ranks processed independently: `species, genus, family, order, class, phylum, kingdom, domain`
- For each rank: drop empty → drop low-frequency labels (count < 30) → aggregate by WURCS
- **Produces**:
  - `split_files/{species,genus,...domain}.csv` (header: `wurcs, {rank}, glycan`)
  - `split_log.txt` (list of discarded low-frequency labels)

> `split_files/` already exists with the full set of outputs. If you do not want to rerun Stage 0, you can inspect this directory directly. To reproduce from scratch, run Stage 0 → Stage 1 in order.

---

## FAQ

**Q: The API times out or gets rate-limited?**  
The endpoint is hosted at [glycosmos.org](https://api.glycosmos.org/glycanformatconverter/). The current script already uses a 30s timeout, 2 retries, and a 0.1s request interval. If failures persist, lower `MAX_WORKERS` or increase `REQUEST_INTERVAL`.

**Q: Why is there also a glycan_classification_merged.csv inside allstata/?**  
`allstata/merge_fixed.py` writes its output into `allstata/`. `分拆.py` at the repo root reads the copy at the repo root via a relative path, so after Stage 0 you need to copy the file over (Stage 0.4). The copy inside allstata/ can be kept or deleted — it does not affect Stage 1.
