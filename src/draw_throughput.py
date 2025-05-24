import pandas as pd
import matplotlib.pyplot as plt
import os

def plot_throughput_per_K(method_to_file: dict, k_list: list, save_dir="results"):
    for k in k_list:
        plt.figure(figsize=(10, 5))
        xmin, xmax = float("inf"), float("-inf")

        for method, filepath in method_to_file.items():
            if not os.path.exists(filepath):
                print(f"⚠️ 找不到檔案：{filepath}")
                continue

            df = pd.read_csv(filepath)
            df_k = df[df["K"] == k].sort_values("user_id")

            if df_k.empty:
                print(f"⚠️ {method} 沒有 {k} 的資料")
                continue

            x = df_k["user_id"]
            y = df_k["avg_throughput"]
            plt.plot(x, y, label=method)

            xmin = min(xmin, x.min())
            xmax = max(xmax, x.max())

        plt.xlabel("User ID")
        plt.ylabel("Average Throughput (Mbps)")
        plt.title(f"Avg Throughput per User - {k}")
        plt.xlim(xmin, xmax)  # 讓 X 軸貼齊左右邊
        plt.grid(True, alpha=0.3)

        plt.legend(
            loc='upper right',
            frameon=True,
            fancybox=True,
            shadow=False,
            fontsize=10
        )

        # 減少圖邊界空間
        plt.tight_layout()
        plt.subplots_adjust(left=0.08, right=0.98)

        output_path = os.path.join(save_dir, f"avg_throughput_comparison_{k}.png")
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.show()

method_to_prefix = {
    "DP": "results/dp_avg_throughput_per_user.csv",
    "Yens": "results/yens_avg_throughput_per_user.csv",
    "Proposed": "results/proposed_avg_throughput_per_user.csv",
    "TPB": "results/tpb_avg_throughput_per_user.csv",
    "LBB": "results/lbb_avg_throughput_per_user.csv",
    "SDB" : "results/dura_avg_throughput_per_user.csv",
    "MSLB" : "results/mslb_avg_throughput_per_user.csv"
}
K_list = ["K0", "K1", "K2","K3", "K4"]
plot_throughput_per_K(method_to_prefix, K_list)        