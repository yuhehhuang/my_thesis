import os
import shutil

# 建立資料夾（如果不存在）
os.makedirs("data", exist_ok=True)
os.makedirs("results", exist_ok=True)
os.makedirs("src", exist_ok=True)

# 不要搬這些
EXCLUDE = {"main.py", "organize_project.py", "README.md", ".gitignore", "__pycache__"}

# 根據副檔名分類
for fname in os.listdir("."):
    if fname in EXCLUDE or os.path.isdir(fname):
        continue

    # 搬到 data/
    if fname.endswith((".csv", ".pkl", ".txt", ".sc", ".php")):
        if "result" not in fname and "success_rate" not in fname:
            shutil.move(fname, os.path.join("data", fname))
    # 搬到 results/
    elif "result" in fname or "success_rate" in fname:
        shutil.move(fname, os.path.join("results", fname))
    # 搬到 src/（非 main.py）
    elif fname.endswith(".py"):
        shutil.move(fname, os.path.join("src", fname))

print("✅ 檔案已重新整理完成！請手動檢查以下程式碼中的路徑：\n")
print("- pd.read_csv(...)")
print("- df.to_csv(...)")
print("- open(...)")
print("- pickle.load(...)")
print("\n🔧 建議改成例如：pd.read_csv('data/xxx.csv') or 'results/xxx.csv'\n")