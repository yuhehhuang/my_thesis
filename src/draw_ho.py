import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# 建立資料夾保險
os.makedirs("results", exist_ok=True)

# baseline 檔案與對應方法名稱
method_to_file = {
    "Yens": "results/baseline_yens_results.csv",
    "DP": "results/baseline_dp_results.csv",
    "PROPOSED": "results/baseline_proposed_results.csv",
    "TPB": "results/baseline_tpb_results.csv",
    "LBB": "results/baseline_lbb_results.csv",
    "SDB": "results/baseline_dura_results.csv"
}

K_list = ["K0", "K1", "K2","K3", "K4"]
ho_summary = pd.DataFrame(index=K_list)

# 讀取每個方法資料
for method, file_path in method_to_file.items():
    try:
        df = pd.read_csv(file_path)
        avg_ho_by_K = df.groupby("K")["handover_count"].mean().reindex(K_list)
        ho_summary[method] = avg_ho_by_K
    except Exception as e:
        print(f"⚠️ {method} 無法載入: {e}")
        ho_summary[method] = [None] * len(K_list)

# 畫 bar chart
bar_width = 0.13
x = np.arange(len(K_list))  # K0, K1, K2 對應的位置

plt.figure(figsize=(10, 6))

for i, method in enumerate(ho_summary.columns):
    if ho_summary[method].notna().sum() == 0:
        continue
    plt.bar(x + i * bar_width, ho_summary[method], width=bar_width, label=method)

plt.xticks(x + bar_width * (len(ho_summary.columns) - 1) / 2, K_list)
plt.xlabel("K")
plt.ylabel("Average Handover Count")
plt.title("Average Handover Count vs K (Bar Chart)")
plt.legend()
plt.grid(True, axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig("results/avg_handover_bar_chart.png")
plt.show()
