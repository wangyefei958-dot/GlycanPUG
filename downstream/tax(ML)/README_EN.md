# tax(ML) Data Pipeline Reproduction Guide

This directory converts the raw glycosylation classification table into per-rank training data for downstream GNN training.

## Directory Layout

```
tax(ML)/
├── allstata/                     ← Stage 0: Backfill WURCS from raw data
│   ├── glycan_classification.csv     Raw input (no wurcs column)
│   ├── 0add_wurcs_to_csv.py          Batch API call to convert IUPAC → WURCS
│   ├── test_single_glycan.py         Utility library
│   ├── iupac_tree.py                 Tree-structure validation (residue tree parsing + isomorphism check)
│   ├── 1batch_process_errors.py      Retry & repair API-failed rows
│   ├── 2merge_fixed.py               Merge repaired rows back into the main table
│   └── ...intermediate products...
├── allstata/glycan_classification_final.csv  Final Stage 0 artifact (read directly by 3split.py)
├── 3split.py                     ← Stage 1: Split into 8 taxonomic ranks
├── split_log.txt                 ← Stage 1 output: discarded low-frequency labels
└── split_files/                  ← Stage 1 final output
    ├── species.csv
    ├── genus.csv
    └── ... (8 rank CSVs)
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
- **Tree validation (final step)**: Performed by `iupac_tree.py`, replacing the previous character-frequency comparison. `modified_iupac` (expected) and `verified_iupac` (API round-trip) are parsed into residue trees and compared by canonical signature (isomorphism), checking residue types, linkage endpoints, branch placement, stereochemistry (α/β) and modifications. Identical character counts with different topology are flagged as MISMATCH.
- **Pre-API interception**: Illegal input (unbalanced brackets etc.) is marked `INVALID_INPUT`; co-occurring uronic acid A and OAc (both using `?F`) is marked `COLLISION`. Neither calls the API.
- Resumable: already-processed rows are skipped
- **Produces**: `glycan_classification_errors_processed.csv` (with `status` / `modified_iupac` / `verified_iupac` columns)

#### 0.3 Merge repaired rows back

```bash
python merge_fixed.py
```

- Uses `(target, species)` as the join key to replace `ERROR` entries with repaired WURCS values
- **Produces**: `allstata/glycan_classification_final.csv`
- **Final header**: `glycan, species, genus, family, order, class, phylum, kingdom, domain, wurcs`  
  (note the first column is renamed from `target` to `glycan`)

#### 0.4 No file copy required

`3split.py` now resolves an absolute path and reads `allstata/glycan_classification_final.csv` directly, independent of the current working directory. The old `glycan_classification_merged.csv` is a historical artifact and should no longer be used.

### Stage 1: Split by taxonomic rank (run at repo root)

```bash
cd ..
python 3split.py
```

- Reads `allstata/glycan_classification_final.csv` (absolute path, independent of the working directory)
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
That is an artifact from an older 2merge script and is now stale. The current 2merge writes `glycan_classification_final.csv`, which `3split.py` reads directly — no copy step is needed.
