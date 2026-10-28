import subprocess
import sys
from pathlib import Path

def find_classifier_scripts(root: Path):
    """
    递归查找所有包含 'classifier' 的 .py 文件
    """
    return [
        p for p in root.rglob("*.py")
        if "classifier" in p.name.lower()
    ]


def run_script(script_path: Path):
    print(f"\n{'='*60}")
    print(f"[RUN] {script_path}")
    print(f"[CWD] {script_path.parent}")
    print(f"{'='*60}")

    # 关键：cwd 设为脚本所在目录
    result = subprocess.run(
        [sys.executable, script_path.name],
        cwd=script_path.parent,
        stdout=None,
        stderr=None,
    )

    if result.returncode != 0:
        print(f"\n❌ 失败: {script_path}")
    else:
        print(f"\n✅ 成功: {script_path}")


def main():
    root = Path(__file__).resolve().parent
    scripts = find_classifier_scripts(root)

    if not scripts:
        print("❌ 没找到任何 classifier 脚本")
        return

    scripts.sort()

    print(f"🔍 共找到 {len(scripts)} 个脚本")

    for script in scripts:
        run_script(script)


if __name__ == "__main__":
    main()