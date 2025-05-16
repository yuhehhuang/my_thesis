import pandas as pd
import matplotlib.pyplot as plt
import ast

# 假設你有一個 access_matrix 是從 CSV 讀來的
df = pd.read_csv("data/access_matrix.csv")  # or 你的實際路徑
df["visible_sats"] = df["visible_sats"].apply(ast.literal_eval)  # 轉成 list

# 計算每個 time slot 的衛星數量
num_visible_sats = df["visible_sats"].apply(len)

# 畫圖
plt.figure(figsize=(8, 5))
plt.bar(df.index, num_visible_sats)
plt.xlabel("Time slot")
plt.ylabel("Number of Accessible Satellites")
plt.title("Number of Accessible Satellites per Time Slot")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("results/num_accessible_sats_per_timeslot.png")  # 若要存圖
plt.show()