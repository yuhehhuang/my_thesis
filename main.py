import sys
sys.path.append("src")
import pandas as pd
import ast  # 為了安全轉換字串成 list,Abstract Syntax Trees
import random
import yens_algo
import pickle
from collections import defaultdict
import performance_calculate
from copy import deepcopy
from dp_algo import build_full_time_expanded_graph, dp_k_handover_path_dp_style
import os
import json
# 確保 src 資料夾有 __init__.py
init_file = os.path.join("src", "__init__.py")
if not os.path.exists(init_file):
    with open(init_file, "w"):
        pass  # 建立一個空檔
# === access_matrix ，all satellite_name 載入 ===
df_names = pd.read_csv("data/all_satellite_names.csv")
all_satellite_names = df_names["sat_name"].tolist()
df = pd.read_csv("data/access_matrix.csv")
#.apply() 是 pandas 中用來「逐列或逐欄套用一個函數」的工具，是處理 DataFrame 非常常見、非常強大的方法。
df["visible_sats"] = df["visible_sats"].apply(ast.literal_eval)
access_matrix = df.to_dict("records")  # list of dicts 格式
'''''[
  {"time_slot": 0, "visible_sats": [...]},
  {"time_slot": 1, "visible_sats": [...]},
  ...
]'''
#==========

# === user info 載入 ===
user_df = pd.read_csv("data/user_info_with_Ks.csv")
#================================================
with open("data/data_rate_dict_user.pkl", "rb") as f:
    data_rate_dict_user = pickle.load(f)

with open("data/data_rate_dict_avg.pkl", "rb") as f:
    data_rate_dict_avg = pickle.load(f)
# 用來記錄T時間user一共會看到哪些衛星(順便排序)
def load_satellite_names_from_csv(csv_path):
    df = pd.read_csv(csv_path)
    df["visible_sats"] = df["visible_sats"].apply(ast.literal_eval) # 將字串轉換為 list
    all_sats = set()
    for sats in df["visible_sats"]:
        all_sats.update(sats)
    return sorted(list(all_sats))
# === 建立所有衛星初始負載，隨機介於[0~0.5]
def init_sat_load_all(satellites, max_channels_per_sat=200, randomize=True, max_background_users=0):
    sat_load_dict = defaultdict(int)
    for sat in satellites:
        usage = 0
        if randomize:
            usage = random.randint(0, max_background_users)
        sat_load_dict[(sat, 0)] = usage  # 只在 t=0 給初始 load

    return sat_load_dict
#######################################################
# === 擷取所有衛星名稱（不論可見與否）===
satellites = load_satellite_names_from_csv("data/access_matrix.csv") #satellites是指0~T-1總共會看到的那些衛星(1000多個)
sat_load_dict_backup  = init_sat_load_all(
    all_satellite_names,
    max_channels_per_sat=200,
    randomize=True,
    max_background_users=150
    )
K_columns = [f"K{i}" for i in range(6)]
def count_handovers(path):
    return sum(1 for (s1, _), (s2, _) in zip(path[:-1], path[1:]) if s1 != s2)
#######################################################################
def run_baseline_yens_per_K(k_col):
    print(f"\n--- Running for {k_col} ---")

    sat_load_dict = deepcopy(sat_load_dict_backup)
    all_user_paths = []
    results = []

    for _, user in user_df.iterrows():
        user_id = int(user["user_id"])
        t_start = int(user["t_start"])
        t_end = int(user["t_end"])

        if t_start >= len(access_matrix) or t_end >= len(access_matrix):
            continue

        # 清除舊任務使用的資源
        to_delete = []
        for old_user in all_user_paths:
            if old_user["t_end"] < t_start:
                for (sat, ts) in old_user["path"]:
                    sat_load_dict[(sat, ts)] -= 1
                to_delete.append(old_user)
        for u in to_delete:
            all_user_paths.remove(u)

        # 建圖與 source/target nodes
        G = yens_algo.build_graph_for_yens(
            access_matrix, user_id, data_rate_dict_user, sat_load_dict,
            t_start, t_end, LAMBDA=10000
        )
        G_nodes = set(G.nodes)
        source_nodes = [(sat, t_start) for sat in access_matrix[t_start]["visible_sats"] if (sat, t_start) in G_nodes]
        target_nodes = [(sat, t_end) for sat in access_matrix[t_end]["visible_sats"] if (sat, t_end) in G_nodes]

        max_handover = int(user[k_col])
        path, reward = yens_algo.run_yens_algorithm(G, source_nodes, target_nodes, k=2)
        handover_count =count_handovers(path) if path else float("inf")
        success = handover_count <= max_handover

        if success and path:
            for (sat, ts) in path:
                if t_start <= ts <= t_end:
                    sat_load_dict[(sat, ts)] += 1

            all_user_paths.append({
                "user_id": user_id,
                "path": path,
                "t_begin": t_start,
                "t_end": t_end
            })

        results.append({
            "user_id": user_id,
            "K": k_col,
            "K_limit": max_handover,
            "reward": reward if success else None,
            "handover_count": handover_count,
            "success": success
        })

    return pd.DataFrame(results)
###############################################
###############################################
def is_valid_path(path, t_start, t_end):
    times = [ts for _, ts in path]
    return sorted(times) == list(range(t_start, t_end + 1))

def run_dp_per_K(k_col):
    print(f"\n--- Running DP for {k_col} ---")

    sat_load_dict = deepcopy(sat_load_dict_backup)
    active_user_paths = []  # 專門用來釋放資源
    all_user_paths = []     # 所有有參與的 user（寫進 JSON）
    results = []
    for _, user in user_df.iterrows():
        user_id = int(user["user_id"])
        t_start = int(user["t_start"])
        t_end = int(user["t_end"])
        K_val = int(user[k_col])

        if t_start >= len(access_matrix) or t_end >= len(access_matrix):
            print(f"⚠️ user {user_id} skipped due to invalid t_start/t_end")
            all_user_paths.append({
                "user_id": user_id,
                "K": k_col,  # ✅ 加這行
                "path": [],
                "t_begin": t_start,
                "t_end": t_end,
                "success": False,
                "reward": 0.0,
                "handover_count": float("inf")
            })
            results.append({
                "user_id": user_id,
                "K": k_col,
                "K_limit": int(user[k_col]),
                "reward": None,
                "handover_count": float("inf"),
                "success": False
            })
            continue


        to_delete = []
        for old_user in active_user_paths:
            if old_user["t_end"] < t_start:
                for (sat, ts) in old_user["path"]:
                    sat_load_dict[(sat, ts)] -= 1
                to_delete.append(old_user)
        for u in to_delete:
            active_user_paths.remove(u)

        # user 在 [t_start, t_end] 中的可見衛星
        user_visible_sats = {
            row["time_slot"]: row["visible_sats"]
            for row in access_matrix[t_start:t_end + 1]
        }

        graph = build_full_time_expanded_graph(
            user_visible_sats=user_visible_sats,
            data_rate_dict=data_rate_dict_avg,
            load_dict=sat_load_dict,
            t_start=t_start,
            t_end=t_end
        )

        max_handover = int(user[k_col])
        path = []
        reward = 0.0
        try:
            path, reward = dp_k_handover_path_dp_style(
                graph=graph,
                t_start=t_start,
                t_end=t_end,
                K=max_handover
            )
        except Exception as e:
            print(f"❌ user {user_id} DP failed: {e}")
            path = []
            reward = 0.0

        handover_count = count_handovers(path) if path else float("inf")
        success = (path != []) and is_valid_path(path, t_start, t_end) and (handover_count <= max_handover)

        if success:
            for (sat, ts) in path:
                if t_start <= ts <= t_end:
                    sat_load_dict[(sat, ts)] += 1
            active_user_paths.append({
                "user_id": user_id,
                "K": k_col,  # ✅ 加這行
                "path": path,
                "t_begin": t_start,
                "t_end": t_end,
                "success": success,
                "reward": reward,
                "handover_count": handover_count
            })
        all_user_paths.append({
            "user_id": user_id,
            "K": k_col,
            "path": path,
            "t_begin": t_start,
            "t_end": t_end,
            "success": success,
            "reward": reward,
            "handover_count": handover_count
        })
        results.append({
            "user_id": user_id,
            "K": k_col,
            "K_limit": max_handover,
            "reward": reward if success else None,
            "handover_count": handover_count,
            "success": success
        })

    return pd.DataFrame(results), all_user_paths

# === Yens 跑每個 K ===
dfs = []
for k_col in K_columns:
    df_k = run_baseline_yens_per_K(k_col)
    dfs.append(df_k)

df_result = pd.concat(dfs, ignore_index=True)
df_result.to_csv("results/baseline_yens_results.csv", index=False)
#=====DP跑每個K================
dfs_dp = []
all_user_paths_dp = []
for k_col in K_columns:
    df_k, paths_k = run_dp_per_K(k_col) 
    dfs_dp.append(df_k)
    all_user_paths_dp.extend(paths_k)

df_result_dp = pd.concat(dfs_dp, ignore_index=True)
assert len(df_result_dp) == 500 * len(K_columns), f"⚠️ DP 結果筆數應為 {500 * len(K_columns)}，但實際為 {len(df_result_dp)}"
df_result_dp.to_csv("results/baseline_dp_results.csv", index=False)
with open("results/dp_user_paths.json", "w") as f:
    json.dump(all_user_paths_dp, f, indent=2)
# === 成功率統計(Yens) ===
success_rate = (
    df_result.groupby("K")["success"]
    .mean()
    .reset_index()
    .rename(columns={"success": "success_rate"})
)
success_rate["success_rate"] = (success_rate["success_rate"] * 100).round(2)
print("\n✅ Yens 每個 K 值的成功率 (%):")
print(success_rate)
success_rate.to_csv("results/yens_success_rate.csv", index=False)
#===========成功率統計(DP)========
success_rate_dp = (
    df_result_dp.groupby("K")["success"]
    .mean()
    .reset_index()
    .rename(columns={"success": "success_rate"})
)
success_rate_dp["success_rate"] = (success_rate_dp["success_rate"] * 100).round(2)
print("\n✅ DP 每個 K 值的成功率 (%):")
print(success_rate_dp)
success_rate_dp.to_csv("results/dp_success_rate.csv", index=False)