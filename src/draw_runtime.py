import pandas as pd
import matplotlib.pyplot as plt

# 讀取資料
df = pd.read_csv("results/running_time_per_k.csv")

# 確保 K 有正確排序
K_list = sorted(df["K"].unique(), key=lambda k: int(k[1:]))
methods = df["method"].unique()

# 指定 marker
method_markers = {
    "Yens": "o", "DP": "s", "PROPOSED": "D", "TPB": "^", "LBB": "v", "DURA": "P","MSLB": "X"
}

plt.figure(figsize=(10, 6))

for method in methods:
    df_method = df[df["method"] == method].set_index("K").reindex(K_list)
    plt.plot(
        K_list,
        df_method["time_sec"],
        label=method,
        marker=method_markers.get(method, 'o')
    )

plt.xlabel("K")
plt.ylabel("Running Time (seconds)")
plt.title("Running Time vs K for Different Methods")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("results/running_time_vs_K_backup.png")
plt.show()