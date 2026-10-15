import requests
import re
import os
from bs4 import BeautifulSoup

def get_species_list():
    url = "https://glycosmos.org/organisms/index?q%5Brank_in%5D=Species&q%5Bs%5D=glycan_count+desc"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    }

    species_data = []
    page = 1

    while True:
        print(f"正在获取第 {page} 页...")
        page_url = f"{url}&page={page}" if page > 1 else url
        response = requests.get(page_url, headers=headers)
        response.raise_for_status()

        soup = BeautifulSoup(response.text, 'html.parser')

        table = soup.find('table', class_='table')
        if not table:
            break

        tbody = table.find('tbody')
        if not tbody:
            break

        rows = tbody.find_all('tr')
        if not rows:
            break

        page_has_data = False
        for row in rows:
            cells = row.find_all('td')
            if len(cells) >= 3:
                name_cell = cells[0]
                name_link = name_cell.find('a')
                if name_link:
                    species_name = name_link.get_text(strip=True)
                    species_url = name_link['href']
                    species_id = species_url.split('/')[-1]

                    glycans_cell = cells[3]
                    glycans_text = glycans_cell.get_text(strip=True)
                    if glycans_text.isdigit():
                        glycans_count = int(glycans_text)
                    else:
                        glycans_count = 0

                    if glycans_count > 20:
                        species_data.append({
                            'name': species_name,
                            'id': species_id,
                            'glycans_count': glycans_count,
                            'url': species_url
                        })
                        page_has_data = True
                    else:
                        print(f"遇到glycans <= 20的物种，停止爬取: {species_name} ({glycans_count})")
                        print(f"共找到 {len(species_data)} 个glycans>20的物种")
                        return species_data

        if not page_has_data:
            break

        pagination = soup.find('ul', class_='pagination')
        if pagination:
            next_link = pagination.find('li', class_='next')
            if not next_link or 'disabled' in next_link.get('class', []):
                break

        page += 1

    print(f"共找到 {len(species_data)} 个glycans>20的物种")
    return species_data

def download_glycans_csv(species_id, species_name):
    safe_name = re.sub(r'[\\/*?:"<>|]', '_', species_name)
    filename = f"{safe_name}.csv"

    if os.path.exists(filename):
        print(f"文件 {filename} 已存在，跳过")
        return False

    download_url = f"https://glycosmos.org/organisms/{species_id}/glycans.csv?action=show&controller=organisms&id={species_id}&table=glycans"

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    }

    try:
        response = requests.get(download_url, headers=headers)
        response.raise_for_status()

        content = response.text

        lines = content.split('\n')
        if len(lines) > 0 and ',' in lines[0]:
            header_parts = [h.strip() for h in lines[0].split(',')]
            if 'GlyTouCan ID' in header_parts:
                gtc_index = header_parts.index('GlyTouCan ID')
                filtered_lines = [header_parts[gtc_index]]
                for line in lines[1:]:
                    if line.strip():
                        parts = line.split(',')
                        if len(parts) > gtc_index:
                            filtered_lines.append(parts[gtc_index])
                content = '\n'.join(filtered_lines)

        with open(filename, 'w', encoding='utf-8') as f:
            f.write(content)

        print(f"已下载: {filename}")
        return True
    except Exception as e:
        print(f"下载 {species_name} 时出错: {e}")
        return False

def main():
    print("正在获取glycans>20的物种列表（按glycans数量降序）...")
    species_list = get_species_list()

    print(f"\n开始下载 {len(species_list)} 个物种的glycans数据...\n")

    downloaded_count = 0
    for species in species_list:
        print(f"处理: {species['name']} (glycans: {species['glycans_count']})")
        if download_glycans_csv(species['id'], species['name']):
            downloaded_count += 1

    print(f"\n下载完成！共下载 {downloaded_count} 个文件")

if __name__ == "__main__":
    main()