# Glycan Immunogenicity Dataset — WURCS Conversion Pipeline

This project converts glycan strings in **IUPAC Condensed** format to **WURCS** format for the glycan immunogenicity dataset, producing computer-readable standardized representations for downstream modeling.

## Data Sources

- **Input**: `glycan_immunogenicity.csv` (1320 glycan records)
- **Output**: `imm_stata.csv` (1106 clean records after filtering)

### Input Columns

| Column | Description |
|--------|-------------|
| `glycan_id` | Glycan ID |
| `glycan` | IUPAC Condensed glycan string (conversion input) |
| `species` | Source organism |
| `immunogenicity` | Immunogenicity label (0.0 / 1.0) |
| `inferred_origin` | Inferred origin |
| `link` | Annotation column |
| `train` / `valid` / `test` | Dataset split flags |

---

## Pipeline Execution Order

Scripts are prefixed with numbers indicating execution order. **Run them sequentially**:

```
0add_wurcs_to_csv.py      ← Step 1
        │
        ▼
1batch_process_errors.py  ← Step 2 (retry failed records from Step 1)
        │
        ▼
2merge_fixed.py           ← Step 3 (merge fixed results back)
        │
        ▼
3create_imm_stata.py      ← Step 4 (generate final deliverable)
```

### Step Details

#### Step 1 — Initial Conversion: `0add_wurcs_to_csv.py`

- **Input**: `glycan_immunogenicity.csv`
- **API**: `https://api.glycosmos.org/glycanformatconverter/2.7.0/iupaccondensed2wurcs`
- **Strategy**: 6-thread concurrent calls to the Glycosmos API for direct IUPAC Condensed → WURCS conversion
- **Output**:
  - `glycan_immunogenicity_with_wurcs.csv` — all records (successes have WURCS; failures marked `wurcs=ERROR`)
  - `glycan_immunogenicity_errors.csv` — failed records only (~691)
  - `wurcs_immunogenicity_progress.json` — checkpoint file for resume capability

#### Step 2 — Error Recovery: `1batch_process_errors.py`

- **Input**: `glycan_immunogenicity_errors.csv`
- **Core Logic**: Imports utility functions from `test_single_glycan.py`. Uses a **placeholder substitution strategy** to work around API limitations with uncommon modifications:

  | Modification | Placeholder | Note |
  |--------------|-------------|------|
  | OMe          | `?Me`       | Methoxy |
  | Sugar + A    | `?F`        | Uronic acid (e.g., GalA) |
  | OP           | `?P`        | Phosphorylation |
  | OS           | `?S`        | Sulfation |
  | OAc          | `?F` → replaced in WURCS as `_?*OCC/3=O` | Acetyl |
  | NeuNAc       | Neu5Ac      | Naming normalization |

- **Validation flow**: Substitution → API call → WURCS cleanup → reverse WURCS→IUPAC → composition (character-frequency) comparison
- **Output**: `glycan_immunogenicity_errors_processed.csv` (contains SUCCESS / MISMATCH / ERROR status)

#### Step 3 — Merge Fixed Results: `2merge_fixed.py`

- **Input**:
  - `glycan_immunogenicity_with_wurcs.csv` (initial conversion)
  - `glycan_immunogenicity_errors_processed.csv` (repaired results)
- **Strategy**: Keyed by `(glycan, species)` — patches successful WURCS back into the initial results
- **Output**: `glycan_immunogenicity_final.csv` (some ERROR rows may remain)

#### Step 4 — Final Filtering: `3create_imm_stata.py`

- **Input**: `glycan_immunogenicity_final.csv`
- **Strategy**: Removes all rows where `wurcs == "ERROR"`
- **Output**: `imm_stata.csv` (final deliverable for downstream modeling)

---

## File Overview

| File | Type | Description |
|------|------|-------------|
| `glycan_immunogenicity.csv` | Input | Original glycan immunogenicity data |
| `glycan_immunogenicity_with_wurcs.csv` | Intermediate | Initial API conversion results |
| `glycan_immunogenicity_errors.csv` | Intermediate | Records that failed initial conversion |
| `glycan_immunogenicity_errors_processed.csv` | Intermediate | Error recovery results |
| `glycan_immunogenicity_final.csv` | Intermediate | Data after merging repaired records |
| `imm_stata.csv` | **Final** | Clean dataset with WURCS annotations |
| `test_single_glycan.py` | Utility | Core library for placeholder substitution, bidirectional conversion, and validation (used by Step 2) |

---

## Dependencies

```
requests    # HTTP client for Glycosmos API calls
```

Python standard library: `csv`, `os`, `re`, `json`, `threading`, `concurrent.futures`, `collections`

## Notes

- **API Rate Limiting**: The Glycosmos public API requires no authentication, but concurrency must be controlled (this project uses `MAX_WORKERS=6/28` and `REQUEST_INTERVAL=0.1~0.5s`) to avoid throttling
- **Checkpoint Resume**: Step 1 saves progress to `wurcs_immunogenicity_progress.json` and can resume after interruption
- **Placeholder Limitations**: The substitution logic in Step 2 is empirically derived — not all modifications are recognized by the Glycosmos API, and some records will remain ERROR after recovery
