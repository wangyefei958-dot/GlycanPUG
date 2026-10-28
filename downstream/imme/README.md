# Glycan Immunogenicity Dataset — WURCS Conversion Pipeline

This project converts glycan strings in **IUPAC Condensed** format to **WURCS** format for the glycan immunogenicity dataset, producing computer-readable standardized representations for downstream modeling.

## Data Sources

- **Input**: `glycan_immunogenicity.csv` (1320 glycan records)
- **Output**: `imm_stata.csv` (1106 clean records after filtering)

The input file comes from the immunogenicity task of the **GlycanML** benchmark (Xu et al., *GlycanML: A Multi-Task and Multi-Structure Benchmark for Glycan Machine Learning*, ICLR 2025) — the 1,320 SugarBase glycans with literature-annotated immunogenicity; its columns include GlycanML's train/valid/test splits, **which downstream training uses as-is**.

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

- **Validation flow**: Substitution → API call → WURCS cleanup → reverse WURCS→IUPAC → **tree-structure validation**
- **Tree validation (final step)**: Handled locally by `iupac_tree.py`, replacing the previous character-frequency comparison. Both `modified_iupac` (expected structure) and `verified_iupac` (API round-trip result) are parsed into residue trees, then compared by canonical signature (isomorphism) — checking residue types, linkage endpoints (donor anomeric carbon + acceptor position), branch placement, stereochemistry (α/β) and modification count/position. Structures with identical character counts but different topology (changed linkage position, anomer, or linear-vs-branched) are correctly flagged as MISMATCH.
- **Pre-API interception**: Illegal input (unbalanced brackets etc.) is marked `INVALID_INPUT`; co-occurring uronic acid A and OAc (both using the `?F` placeholder) is marked `COLLISION`. Neither calls the API.
- **Output**: `glycan_immunogenicity_errors_processed.csv` with `status`, `modified_iupac`, `verified_iupac` columns; status is one of SUCCESS / MISMATCH / COLLISION / INVALID_INPUT / PARSE_FAIL / ERROR

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
| `test_single_glycan.py` | Utility | Core library for placeholder substitution and bidirectional conversion (used by Step 2) |
| `iupac_tree.py` | Utility | Parses IUPAC into residue trees and performs isomorphism-based tree validation, including input legality checks (used by Step 2) |

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
- **Tree Validation**: Step 2's final decision is based on structural isomorphism rather than character counting — identical character frequency does not prove identical structure. MISMATCH rows carry the specific differences in `status` (e.g. `残基@Galf: Galf!=Gal`)
