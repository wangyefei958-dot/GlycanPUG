# GlyConnect Glycan Structure Processing Pipeline

A three-stage pipeline that scrapes glycan structures from the [GlyConnect](https://glyconnect.expasy.org/) database, enriches them with WURCS identifiers, and produces a cleaned dataset for downstream tasks.

## Pipeline Overview

```
GlyConnect website
      │
      ▼  0glyconnect_crawler.py
glyconnect_structures.csv  (3 cols: GlyTouCan ID, glycantype, IUPAC Condensed)
      │
      ▼  1process_glyconnect.py  +  dic.csv
glyconnect_structures.csv  (4 cols: + WURCS)
      │
      ▼  2remove_tilde.py
creature_0wave.csv  (polysaccharide structures removed)
```

## Files

| File | Description |
| --- | --- |
| `0glyconnect_crawler.py` | Crawls all pages of GlyConnect, extracts GlyTouCan ID, glycan type, and IUPAC condensed notation |
| `1process_glyconnect.py` | Joins WURCS identifiers from `dic.csv` and drops empty/invalid rows |
| `2remove_tilde.py` | Removes structures whose WURCS contains `~` (polymer repeating unit) and one known bad entry |
| `dic.csv` | External mapping of GlyTouCan ID → WURCS |
| `glyconnect_structures.csv` | Final 4-column dataset after crawling and WURCS join |

## Requirements

- Python 3.10+
- `requests`
- `beautifulsoup4`
- `pandas`

```bash
pip install requests beautifulsoup4 pandas
```

## Usage

Run the scripts in numerical order:

```bash
# Step 1: crawl GlyConnect (~57 pages, a few minutes)
python 0glyconnect_crawler.py

# Step 2: enrich with WURCS (requires dic.csv to be present)
python 1process_glyconnect.py

# Step 3: filter out polysaccharides
python 2remove_tilde.py
```

Each step **overwrites** `glyconnect_structures.csv`. The final step writes to `creature_0wave.csv`. Back up manually if you need to preserve intermediate outputs.

## Output Format

`glyconnect_structures.csv` has 4 columns:

| Column | Example | Description |
| --- | --- | --- |
| `GlyTouCan ID` | `G51562OR` | Unique structure identifier from [GlyTouCan](https://glytoucan.org/) |
| `glycantype` | `N-Linked`, `O-Linked`, ... | Glycan class |
| `IUPAC Condensed` | `Gal(b1-4)GlcNAc(b1-3)...` | Structure in condensed IUPAC notation |
| `WURCS` | `WURCS=2.0/5,18,17/...` | Structure in [WURCS 2.0](https://www.wurcs.org/) format |
