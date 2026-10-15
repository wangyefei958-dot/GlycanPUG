import requests
from bs4 import BeautifulSoup
import csv
import time
import os

headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
base_url = "https://glyconnect.expasy.org/all/structures"

all_data = []
total_pages = 57  # p=0 to p=56

for page in range(total_pages):
    url = f"{base_url}?p={page}&n=100&s=id&o=asc"
    print(f"Fetching page {page+1}/{total_pages}...", end=" ", flush=True)
    try:
        r = requests.get(url, headers=headers, timeout=30)
        soup = BeautifulSoup(r.text, "html.parser")
        table = soup.find_all("table")[1]
        rows = table.find_all("tr")
        count = 0
        for row in rows[1:]:
            cells = row.find_all("td")
            if len(cells) >= 6:
                glytoucan = cells[5].get_text(strip=True)
                glycan_type = cells[4].get_text(strip=True)
                iupac = cells[1].get_text(strip=True)
                all_data.append([glytoucan, glycan_type, iupac])
                count += 1
        print(f"got {count} rows, total: {len(all_data)}")
    except Exception as e:
        print(f"Error: {e}")
        # Retry once
        time.sleep(5)
        try:
            r = requests.get(url, headers=headers, timeout=30)
            soup = BeautifulSoup(r.text, "html.parser")
            table = soup.find_all("table")[1]
            rows = table.find_all("tr")
            count = 0
            for row in rows[1:]:
                cells = row.find_all("td")
                if len(cells) >= 6:
                    glytoucan = cells[5].get_text(strip=True)
                    glycan_type = cells[4].get_text(strip=True)
                    iupac = cells[1].get_text(strip=True)
                    all_data.append([glytoucan, glycan_type, iupac])
                    count += 1
            print(f"Retry got {count} rows, total: {len(all_data)}")
        except Exception as e2:
            print(f"Retry also failed: {e2}")
    time.sleep(1)  # Be polite

# Write to CSV with iupac quoted
script_dir = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(script_dir, "glyconnect_structures.csv")

print(f"\nTotal rows: {len(all_data)}")
with open(csv_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f, quoting=csv.QUOTE_NONNUMERIC)
    writer.writerow(["GlyTouCan ID", "glycantype", "IUPAC Condensed"])
    for row in all_data:
        writer.writerow([row[0], row[1], row[2]])

print("Saved to glyconnect_structures.csv")

