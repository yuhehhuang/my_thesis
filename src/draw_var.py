import pandas as pd
import matplotlib.pyplot as plt

# 讀取正確的 variance 結果
df = pd.read_csv("results/variance_by_method.csv")

# 確保排序一致
K_list = ["K0", "K1", "K2","K3", "K4"]
methods = df["method"].unique()

# 給每個方法一個 marker
method_markers = {
    "Yens": "o", "DP": "s", "PROPOSED": "D",
    "TPB": "^", "LBB": "v", "SDB": "P","MSLB": "X" 
}

plt.figure(figsize=(10, 6))

# 依方法畫折線圖
for method in methods:
    df_method = df[df["method"] == method]
    df_method = df_method.set_index("K").reindex(K_list)  # 保證順序為 K0, K1, K2
    plt.plot(
        K_list,
        df_method["avg_variance"],
        label=method,
        marker=method_markers.get(method, 'o')
    )

plt.xlabel("K")
plt.ylabel("Average Load Variance")
plt.title("Load Variance vs K for Different Methods")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("results/load_variance_vs_K.png")
plt.show()
