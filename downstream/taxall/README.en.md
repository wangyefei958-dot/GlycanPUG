# Glycan Taxonomy Pipeline

A pipeline that builds multi-label glycan datasets organized by NCBI taxonomic rank (8 levels: superkingdom → species). It crawls [Glycosmos](https://glycosmos.org/organisms) for organisms per rank and their glycan lists, joins GlyTouCan IDs to WURCS structures, cleans invalid structures, and merges by WURCS — concatenating taxon names with `_` when a glycan spans multiple taxa.

## Directory layout

```
taxall/
├── preprocessall.py              # main preprocessing script (loops over each rank folder)
├── 1superkingdom/
│   ├── dic.csv                   # GlyTouCan ID → WURCS dictionary (~57 MB)
│   ├── superkingdom.csv          # final merged output
│   └── data/
│       ├── glycosmos_crawler.py  # rank-specific crawler
│       └── <taxon>.csv ...       # one CSV per taxon (GlyTouCan ID column)
├── 2phylum/        …  same structure, output = phylum.csv
├── 3class/         …  class.csv
├── 4order/         …  order.csv
├── 5family/        …  family.csv
├── 6genus/         …  genus.csv
├── 7kingdom/       …  kingdom.csv
├── 8species/       …  species.csv
├── split_files/    # manually created; copies of the 8 final CSVs
│   ├── *.csv
│   ├── 数量统计.py    # row / label-count statistics
│   └── result.txt
└── __pycache__/     # stale bytecode: remove_tilde.py / 合并_带标签.py / creature_to_graph.py (sources deleted; first two refactored into preprocessall.py)
```

Each `<N><rank>/` folder is self-contained: the same `dic.csv`, its own `data/` subfolder (rank-specific crawler + raw per-taxon CSVs), and the rank's final output CSV.

## Pipeline

### 1. Crawl — `data/glycosmos_crawler.py`

Each rank folder ships its own copy, templated for that rank. The crawler:

- Pages through `https://glycosmos.org/organisms/index?q[rank_in]=<RANK>&q[s]=glycan_count+desc`.
- Collects organisms with `glycans_count > 20` (superkingdom uses `> 30`).
- Stops at the first organism with `glycans_count <= 20` (the list is sorted by descending glycan count).
- For each organism, downloads its glycans CSV from `https://glycosmos.org/organisms/<id>/glycans.csv`, keeps only the `GlyTouCan ID` column, and writes it as `data/<organism_name>.csv`.

### 2. Clean + merge — `preprocessall.py`

Run from the project root. It walks every `<N><rank>/` subfolder and, for each:

1. Loads `dic.csv` to build a `GlyTouCan ID → WURCS` map.
2. For every CSV in `data/`:
   - Reads it and **adds the `WURCS` column** by joining on GlyTouCan ID (writes the file back in place).
   - **Cleaning rules (hard constraints)**:
     - Drops rows where **any column contains `~`** (`~` is a WURCS placeholder for incomplete structures).
     - Drops rows where `GlyTouCan ID == G18907LN` (a specific bad entry, hardcoded).
     - If fewer than **30 rows** remain after cleaning, the whole file is skipped.
   - Adds a temporary `_temp_label` column with the file stem (the taxon name).
3. Concatenates all kept files.
4. **Groups by `WURCS`** (deduplication); for groups spanning multiple files, joins the sorted taxon names with `_` to form a multi-label string.
5. Drops the `_temp_label` column, normalizes the `WURCS` column name, removes unnamed columns.
6. The label column is named after the rank (folder name with the leading number prefix stripped, e.g. `1superkingdom` → `superkingdom`).
7. Writes the final merged file as `<rank>.csv` at the folder root (e.g. `1superkingdom/superkingdom.csv`).

### 3. Split folder — `split_files/`

Manually created folder. The 8 final `<rank>.csv` files were manually copied here for downstream convenience. No script generates this folder.

### 4. Statistics — `split_files/数量统计.py`

Scans its own directory for CSVs and, for each, counts:

- Valid data rows (skipping header, requiring ≥ 3 columns).
- Unique labels (3rd column split by `_`, deduplicated across all files for the grand total).

Writes both to console and `result.txt`.

## Output

The 8 final files (one per rank), columns: `GlyTouCan ID, WURCS, <rank>`, where `<rank>` holds one or more taxon names joined by `_` when multi-label.

### Counts (from `split_files/result.txt`)

| File              | Rows   | Unique labels |
|-------------------|--------|---------------|
| superkingdom.csv  | 15,431 | 4             |
| kingdom.csv       | 11,720 | 5             |
| phylum.csv        | 15,271 | 19            |
| class.csv         | 15,089 | 31            |
| order.csv         | 14,571 | 65            |
| family.csv        | 14,054 | 76            |
| genus.csv         | 13,179 | 85            |
| species.csv       | 11,877 | 79            |
| **Total**         | 111,192 | 364 (dedup)   |

## Dependencies

- Python 3.13 (tested)
- `pandas`, `numpy`
- `requests`, `beautifulsoup4` (crawler only)

## Notes

- **The crawler is time-sensitive**: [Glycosmos](https://glycosmos.org/organisms) may change at any time (list-page URL patterns, HTML element classes, glycan CSV export format, etc.). If re-crawling returns empty data or errors, `data/glycosmos_crawler.py` will need adjustment to match the current site structure — URL, CSS selectors, and column names.
- The 8 `dic.csv` files are byte-identical copies of the same GlyTouCan WURCS dictionary (~57 MB each).
- `__pycache__` holds stale bytecode for `remove_tilde.py`, `合并_带标签.py`, and `creature_to_graph.py`; the `.py` sources no longer exist. The first two have been refactored into `preprocessall.py`.
- All cleaning operations modify files in place under `data/` and are irreversible. Back up the raw data before the first run.
