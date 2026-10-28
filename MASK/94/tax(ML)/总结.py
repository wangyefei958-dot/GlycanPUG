import os
import csv
import sys

LEVELS = ['domain', 'kingdom', 'phylum', 'class', 'order', 'family', 'genus', 'species']

def extract_metrics(log_path):
    """支持多种日志格式"""
    f1 = auc = None
    try:
        with open(log_path, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                line = line.strip()
                
                # 匹配 F1
                if line.startswith('F1 (macro):') or line.startswith('Macro F1:'):
                    parts = line.split()
                    if parts:
                        f1 = float(parts[-1])
                
                # 匹配 AUC（支持两种格式）
                if line.startswith('AUC (macro):') or line.startswith('Macro AUC:'):
                    parts = line.split()
                    if parts:
                        auc = float(parts[-1])
                        
    except Exception as e:
        print(f"⚠️  读取失败 {log_path}: {e}")
    
    return f1, auc

def find_training_logs(script_dir):
    results = {}
    finaltrain_dir = os.path.join(script_dir, 'finaltrain')
    print(f"🔍 脚本目录: {script_dir}")
    print(f"📂 训练目录: {finaltrain_dir}")
    
    seeds = sorted(
        int(d) for d in os.listdir(finaltrain_dir)
        if d.isdigit() and os.path.isdir(os.path.join(finaltrain_dir, d))
    )
    print(f"🌱 发现种子: {seeds}")

    for seed in seeds:
        seed_dir = os.path.join(finaltrain_dir, str(seed))
        
        if not os.path.exists(seed_dir):
            print(f"⚠️  文件夹不存在: {seed_dir}")
            continue
        
        processed_dir = os.path.join(seed_dir, 'split_files_processed_wurcs')
        if not os.path.exists(processed_dir):
            print(f"⚠️  split_files_processed_wurcs不存在: {processed_dir}")
            continue
        
        print(f"\n✅ 处理 seed={seed}")
        results[seed] = {}
        
        for level in LEVELS:
            log_dir = os.path.join(processed_dir, f'training_results_{level}')
            log_path = os.path.join(log_dir, 'training_log.txt')
            
            if os.path.exists(log_path):
                f1, auc = extract_metrics(log_path)
                results[seed][level] = (f1, auc)
                print(f"  ✅ {level}: F1={f1}, AUC={auc}")
            else:
                print(f"  ❌ {level}: 文件不存在")
                results[seed][level] = (None, None)
    
    return results

def save_csv(data, output_dir):
    levels_cap = [l.capitalize() for l in LEVELS]
    seeds_sorted = sorted(data.keys())
    
    csv_path = os.path.join(output_dir, 'GlycanW0_metrics.csv')
    
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        
        w.writerow(['macro f1 of GlycanW0'])
        w.writerow(['random seed'] + levels_cap)
        for seed in seeds_sorted:
            row = [seed] + [
                data[seed].get(lvl, (None,))[0] or '' 
                for lvl in LEVELS
            ]
            w.writerow(row)
        
        w.writerow([])
        
        w.writerow(['macro AUC of GlycanW0'])
        w.writerow(['random seed'] + levels_cap)
        for seed in seeds_sorted:
            row = [seed] + [
                data[seed].get(lvl, (None, None))[1] or '' 
                for lvl in LEVELS
            ]
            w.writerow(row)
    
    print(f"\n🎉 CSV已保存至：{csv_path}")

if __name__ == '__main__':
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data = find_training_logs(script_dir)
    save_csv(data, script_dir)