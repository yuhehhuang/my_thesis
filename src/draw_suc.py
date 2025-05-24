import pandas as pd
import matplotlib.pyplot as plt

# 設定方法與檔案對應
method_to_file = {
    "Yens": "results/yens_success_rate.csv",
    "DP": "results/dp_success_rate.csv",
    "PROPOSED": "results/proposed_success_rate.csv",
    "TPB": "results/tpb_success_rate.csv",
    "LBB": "results/lbb_success_rate.csv",
    "SDB": "results/dura_success_rate.csv",
    "MSLB": "results/mslb_success_rate.csv"
}

# 指定 marker 形狀
method_markers = {
    "Yens": "o",
    "DP": "s",
    "PROPOSED": "D",
    "TPB": "^",
    "LBB": "v",
    "SDB": "P",
    "MSLB": "X"
}

# 畫圖
plt.figure(figsize=(10, 6))
for method, file_path in method_to_file.items():
    df = pd.read_csv(file_path)
    plt.plot(df["K"], df["success_rate"], label=method, marker=method_markers.get(method, 'o'))

plt.xlabel("K")
plt.ylabel("Success Rate (%)")
plt.title("Success Rate vs K for Different Methods")
plt.ylim(0, 105)  # 限制在 0~100+空間
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig("results/success_rate_vs_K.png")
plt.show()
