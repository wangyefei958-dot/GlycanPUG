import csv
import requests
import time
import json
import os
import re
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

# 获取脚本所在目录
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# 使用相对路径
CSV_PATH = os.path.join(SCRIPT_DIR, "glycan_immunogenicity.csv")
OUTPUT_CSV_PATH = os.path.join(SCRIPT_DIR, "glycan_immunogenicity_with_wurcs.csv")
ERROR_CSV_PATH = os.path.join(SCRIPT_DIR, "glycan_immunogenicity_errors.csv")
PROGRESS_FILE = os.path.join(SCRIPT_DIR, "wurcs_immunogenicity_progress.json")
API_BASE = "https://api.glycosmos.org/glycanformatconverter/2.7.0/iupaccondensed2wurcs"
MAX_RETRIES = 2
RETRY_DELAY = 3
REQUEST_TIMEOUT = 30
BATCH_SAVE = 100
REQUEST_INTERVAL = 0.1
MAX_WORKERS = 6

lock = threading.Lock()
stats_lock = threading.Lock()


def extract_wurcs(response_text):
    try:
        data = json.loads(response_text)
        msg = data.get("message", "")
        match = re.search(r"wurcs=(WURCS=.+)$", msg)
        if match:
            return match.group(1)
        for key in ["WURCS", "IUPACcondensed", "IUPACextended", "GlycoCT", "LinearCode"]:
            if key in data and data[key]:
                return str(data[key])
        if msg and "FileNotFoundException" not in msg and "Error" not in msg:
            return msg
        return ""
    except json.JSONDecodeError:
        return response_text.strip()


def load_progress():
    if os.path.exists(PROGRESS_FILE):
        with open(PROGRESS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"last_index": 0, "completed_indices": []}


def save_progress(progress):
    with open(PROGRESS_FILE, "w", encoding="utf-8") as f:
        json.dump(progress, f, ensure_ascii=False, indent=2)


def convert_iupac_to_wurcs(iupac_str):
    for attempt in range(MAX_RETRIES):
        try:
            r = requests.post(API_BASE, data={"str": iupac_str}, timeout=REQUEST_TIMEOUT)
            if r.status_code == 200:
                wurcs = extract_wurcs(r.text)
                if wurcs:
                    return wurcs, None
                else:
                    error_msg = f"Cannot extract WURCS: {r.text[:200]}"
                    if attempt < MAX_RETRIES - 1:
                        time.sleep(RETRY_DELAY)
                    else:
                        return "ERROR", error_msg
            else:
                error_msg = f"HTTP {r.status_code}: {r.text[:200]}"
                if attempt < MAX_RETRIES - 1:
                    time.sleep(RETRY_DELAY)
                else:
                    return "ERROR", error_msg
        except requests.exceptions.RequestException as e:
            error_msg = str(e)
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_DELAY)
            else:
                return "ERROR", error_msg
    return "ERROR", "Max retries exceeded"


def save_csv(header, rows):
    with open(OUTPUT_CSV_PATH, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)


def save_error_csv(header, error_rows):
    with open(ERROR_CSV_PATH, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header + ["error_reason"])
        writer.writerows(error_rows)


def process_row(args):
    idx, glycan, header, rows, wurcs_idx = args

    if not glycan.strip():
        return idx, "", None

    time.sleep(REQUEST_INTERVAL)
    wurcs, error = convert_iupac_to_wurcs(glycan)

    if wurcs and wurcs.startswith("WURCS="):
        return idx, wurcs, None
    else:
        return idx, "ERROR", error


def main():
    print("=" * 60)
    print("IUPAC Condensed -> WURCS Converter (Multi-threaded)")
    print("=" * 60)
    print(f"Reading CSV: {CSV_PATH}")
    
    with open(CSV_PATH, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)
        rows = list(reader)
    
    total = len(rows)
    print(f"Total: {total} rows")
    
    # 当前CSV的第二列是'glycan'，包含glycan的IUPAC名称
    glycan_idx = header.index("glycan")
    print(f"Glycan column: 'glycan' at index {glycan_idx}")
    
    if "wurcs" in header:
        wurcs_idx = header.index("wurcs")
    else:
        header.append("wurcs")
        for row in rows:
            row.append("")
        wurcs_idx = len(header) - 1
        print("Added 'wurcs' column")
    
    print(f"wurcs column index: {wurcs_idx}")
    print(f"Max workers: {MAX_WORKERS}")
    
    progress = load_progress()
    start_index = progress.get("last_index", 0)
    completed_indices = set(progress.get("completed_indices", []))
    error_records = []
    error_indices = set()

    if os.path.exists(ERROR_CSV_PATH) and os.path.getsize(ERROR_CSV_PATH) > 0:
        with open(ERROR_CSV_PATH, "r", encoding="utf-8") as f:
            reader_err = csv.reader(f)
            next(reader_err)
            error_records = list(reader_err)
        for err_row in error_records:
            if len(err_row) > 0:
                error_indices.add(err_row[0] if err_row[0].isdigit() else None)
        print(f"Loaded {len(error_records)} existing error records")

    if start_index > 0 or completed_indices:
        print(f"Resuming from row {start_index + 1} / {total}")
        with open(OUTPUT_CSV_PATH, "r", encoding="utf-8") as f:
            reader2 = csv.reader(f)
            next(reader2)
            saved_rows = list(reader2)
            for i in range(min(start_index, len(saved_rows))):
                if i < len(rows):
                    rows[i][wurcs_idx] = saved_rows[i][wurcs_idx]
    else:
        print("Starting from beginning")

    success_count = sum(1 for r in rows[:start_index] if r[wurcs_idx] and r[wurcs_idx].startswith("WURCS="))
    fail_count = sum(1 for r in rows[:start_index] if r[wurcs_idx] == "ERROR")
    print(f"Success: {success_count}, Failed: {fail_count}")
    print("-" * 60)

    tasks = []
    for i in range(start_index, total):
        glycan = rows[i][glycan_idx]
        if i not in completed_indices and glycan.strip():
            tasks.append((i, glycan, header, rows, wurcs_idx))

    print(f"Tasks to process: {len(tasks)}")

    try:
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = {executor.submit(process_row, task): task for task in tasks}

            batch_count = 0
            for future in as_completed(futures):
                idx, result, error = future.result()
                rows[idx][wurcs_idx] = result

                with stats_lock:
                    if result.startswith("WURCS="):
                        success_count += 1
                    else:
                        fail_count += 1
                        error_records.append(rows[idx] + [error or "unknown"])

                completed_indices.add(idx)
                batch_count += 1

                if batch_count % BATCH_SAVE == 0:
                    with lock:
                        save_csv(header, rows)
                        save_error_csv(header, error_records)
                        save_progress({
                            "last_index": idx + 1,
                            "completed_indices": list(completed_indices)
                        })
                    print(f"  [Saved] {batch_count}/{len(tasks)} (OK:{success_count} FAIL:{fail_count})")

    except KeyboardInterrupt:
        print("\nUser interrupted! Saving progress...")
        save_csv(header, rows)
        save_error_csv(header, error_records)
        save_progress({
            "last_index": max(completed_indices) + 1 if completed_indices else 0,
            "completed_indices": list(completed_indices)
        })
        print(f"Saved up to row {max(completed_indices) + 1 if completed_indices else 0}.")
        sys.exit(0)
    except Exception as e:
        print(f"\nException: {e}")
        save_csv(header, rows)
        save_error_csv(header, error_records)
        save_progress({
            "last_index": max(completed_indices) + 1 if completed_indices else 0,
            "completed_indices": list(completed_indices)
        })
        sys.exit(1)

    save_csv(header, rows)
    save_error_csv(header, error_records)
    print("\n" + "=" * 60)
    print("Done!")
    print(f"Total: {total}, Success: {success_count}, Failed: {fail_count}")
    print(f"Result: {OUTPUT_CSV_PATH}")
    print(f"Errors: {ERROR_CSV_PATH}")
    print("=" * 60)
    if os.path.exists(PROGRESS_FILE):
        os.remove(PROGRESS_FILE)
        print("Progress file cleaned up")


if __name__ == "__main__":
    main()