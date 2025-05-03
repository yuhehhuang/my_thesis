import pandas as pd
import ast  # 為了安全轉換字串成 list
import random
import yens_algo
import pickle
# === access_matrix 載入 ===
df = pd.read_csv("access_matrix.csv")
df["visible_sats"] = df["visible_sats"].apply(ast.literal_eval)
access_matrix = df.to_dict("records")  # list of dicts 格式
#==========

# === user info 載入 ===
user_df = pd.read_csv("user_info_with_Ks.csv")
#================================================
with open("data_rate_dict.pkl", "rb") as f:
    data_rate_dict = pickle.load(f)
# === 建立所有衛星初始負載（只初始化 t=0） ===
def load_satellite_names_from_csv(csv_path):
    df = pd.read_csv(csv_path)
    df["visible_sats"] = df["visible_sats"].apply(ast.literal_eval)
    all_sats = set()
    for sats in df["visible_sats"]:
        all_sats.update(sats)
    return sorted(list(all_sats))

def init_sat_load_t0(satellites, max_channels_per_sat=200, randomize=False, max_background_users=0):
    sat_load_dict = {}
    for sat in satellites:
        usage = 0
        if randomize:
            usage = random.randint(0, max_background_users)
        L = usage / max_channels_per_sat
        sat_load_dict[(sat, 0)] = min(L, 1.0)
    return sat_load_dict
#######################################################
satellites = load_satellite_names_from_csv("access_matrix.csv")
sat_load_dict = init_sat_load_t0(satellites, randomize=True, max_background_users=100)
#######################################################################
# === baseline: Yen's algorithm ===
K_columns = ["K0", "K1", "K2", "K3", "K4", "K5"]
results = []
def count_handovers(path):
    return sum(1 for (s1, _), (s2, _) in zip(path[:-1], path[1:]) if s1 != s2)
###############################################################################################
for _, user in user_df.iterrows():
    user_id = int(user["user_id"])
    t_start = int(user["t_start"])
    t_end = int(user["t_end"])
        # 邊界檢查
    if t_start >= len(access_matrix) or t_end >= len(access_matrix):
        continue
    # Step 1: 只建一次 G（跟 K 無關）
    G = yens_algo.build_graph_for_yens(
        access_matrix, user_id, data_rate_dict, sat_load_dict,
        t_start, t_end, LAMBDA=10000
    )
    G_nodes = set(G.nodes)
    # Step 2: source / target nodes
    source_nodes = [(sat, t_start) for sat in access_matrix[t_start]["visible_sats"] if (sat, t_start) in G_nodes]
    target_nodes = [(sat, t_end) for sat in access_matrix[t_end]["visible_sats"] if (sat, t_end) in G_nodes]

    # Step 3: 跑 Yen's algorithm
    path, reward =yens_algo.run_yens_algorithm(G, source_nodes, target_nodes, k=2)
    handover_count = count_handovers(path) if path else None

    # Step 4: 對每個 Kx 判斷成功與否
    for k_col in K_columns:
        max_handover = int(user[k_col])
        success = (path is not None) and (handover_count <= max_handover)

        results.append({
            "user_id": user_id,
            "K": k_col,
            "K_limit": max_handover, 
            "reward": reward if success else None,
            "handover_count": handover_count,
            "success": success
        })

df_result = pd.DataFrame(results)
df_result.to_csv("baseline_yens_results.csv", index=False)
# 統計成功率
success_rate = (
    df_result.groupby("K")["success"]
    .mean()
    .reset_index()
    .rename(columns={"success": "success_rate"})
)
success_rate["success_rate"] = (success_rate["success_rate"] * 100).round(2)

# 輸出統計結果
print("\n✅ 每個 K 值的成功率 (%):")
print(success_rate)

# 另存成 CSV
success_rate.to_csv("yens_success_rate.csv", index=False)
######