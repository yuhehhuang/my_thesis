import json
import pandas as pd
import matplotlib.pyplot as plt
import os
from collections import defaultdict

def compute_avg_per_K(json_path, method_name):
    with open(json_path, "r") as f:
        data = json.load(f)

    stats = defaultdict(lambda: {"total_throughput": 0, "total_slots": 0})

    for user in data:
        k = user["K"]
        per_slot = user["throughput_per_slot"]
        slot_values = list(per_slot.values())
        stats[k]["total_throughput"] += sum(slot_values)
        stats[k]["total_slots"] += len(slot_values)  # 只加上實際這個 user 使用過的時間長度

    result = []
    for k, val in stats.items():
        total_slots = val["total_slots"]
        if total_slots > 0:
            avg_tp = val["total_throughput"] / total_slots
        else:
            avg_tp = 0
        result.append({
            "K": k,
            "method": method_name,
            "avg_throughput": avg_tp
        })

    return pd.DataFrame(result)

# 所有方法與 JSON 檔案對應
method_files = {
    "Yens": "results/yens_user_throughput.json",
    "DP": "results/dp_user_throughput.json",
    "PROPOSED": "results/proposed_user_throughput.json",
    "TPB": "results/tpb_user_throughput.json",
    "LBB": "results/lbb_user_throughput.json",
    "SDB": "results/dura_user_throughput.json",
    "MSLB": "results/mslb_user_throughput.json"
}

# 統整各方法資料
all_df = []
for method, path in method_files.items():
    if os.path.exists(path):
        df = compute_avg_per_K(path, method)
        all_df.append(df)
    else:
        print(f"⚠️ 檔案不存在: {path}")

df_all = pd.concat(all_df, ignore_index=True)
# 為每個方法指定不同的 marker 形狀
method_markers = {
    "Yens": "o",       # 圓形
    "DP": "s",         # 方形
    "PROPOSED": "D",   # 菱形
    "TPB": "^",        # 上三角形
    "LBB": "v",        # 下三角形
    "SDB": "P",         # 五邊形
    "MSLB": "X"      # X 形狀
}
plt.figure(figsize=(9, 6))
for method in df_all["method"].unique():
    df_plot = df_all[df_all["method"] == method].sort_values("K")
    marker = method_markers.get(method, "o")  # 預設為圓形
    plt.plot(df_plot["K"], df_plot["avg_throughput"], marker=marker, label=method)

plt.xlabel("K")
plt.ylabel("Global Avg Throughput (Mbps)")
plt.title("Average Throughput vs. K for Each Method")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig("results/global_avg_throughput_vs_k.png")
plt.show()
# 存結果
df_all.to_csv("results/global_avg_throughput_summary.csv", index=False)