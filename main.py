import sys
sys.path.append("src")
import pandas as pd
import ast  # 為了安全轉換字串成 list,Abstract Syntax Trees
import random
import pickle
from collections import defaultdict,Counter
import performance_calculate
from copy import deepcopy
import time
import os
import json
import networkx as nx
import yens_algo
import dp_algo
import performance_calculate
from proposed_method import run_proposed_method_for_user
import tpb_method
import lbb_method
import sdb_method
import mslb
#================================================
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
# === user info 載入 ===
user_df = pd.read_csv("data/user_info_with_Ks.csv")
#================================================
with open("data/data_rate_dict_user.pkl", "rb") as f:
    data_rate_dict_user = pickle.load(f)

# 用來記錄T時間user一共會看到哪些衛星(順便排序)
def load_satellite_names_from_csv(csv_path):
    df = pd.read_csv(csv_path)
    df["visible_sats"] = df["visible_sats"].apply(ast.literal_eval) # 將字串轉換為 list
    all_sats = set()
    for sats in df["visible_sats"]:
        all_sats.update(sats)
    return sorted(list(all_sats))
# === 建立所有衛星初始使用的channel數量 ===
def init_sat_load_all(satellites, max_channels_per_sat=200, randomize=True, max_background_users=0):
    sat_load_dict = defaultdict(int)
    for sat in satellites:
        usage = 0
        if randomize:
            usage = random.randint(0, max_background_users)
        sat_load_dict[sat] = usage  

    return sat_load_dict
#######################################################
satellites = load_satellite_names_from_csv("data/access_matrix.csv") #satellites是指0~T-1總共會看到的那些衛星(1000多個)
sat_load_dict_backup  = init_sat_load_all(
    all_satellite_names,
    max_channels_per_sat=200,
    randomize=True,
    max_background_users=100
    )

K_columns = [f"K{i}" for i in range(5)]
def count_handovers(path):
    return sum(1 for (s1, _), (s2, _) in zip(path[:-1], path[1:]) if s1 != s2)
#######################################################################
def run_baseline_yens_per_K(k_col):
    MAX_CHANNELS_PER_SAT = 200 
    print(f"\n--- Running for {k_col} ---")
    sat_load_dict = defaultdict(int, deepcopy(sat_load_dict_backup))
    active_user_paths = []  # 專門用來釋放資源
    all_user_paths = []     # 全部記錄到 JSON 分析用
    results = []
    #load_by_time是Dict[str, Dict[str, int]]:time_slot -> {sat_name -> load}
    load_by_time = defaultdict(lambda: defaultdict(int))

    for _, user in user_df.iterrows():
        user_id = int(user["user_id"])
        t_start = int(user["t_start"])
        t_end = int(user["t_end"])
        max_handover = int(user[k_col])
        # === ✅ Step 1: 清除已過期使用者的佔用 load
        to_delete = []
        for old_user in active_user_paths:
            if old_user["t_end"] < t_start:
                path = old_user.get("path")
                if path is None:
                    continue
                from collections import Counter
                #Counter會return一個字典，key是path裡面出現的衛星名稱，value是出現的次數
                usage_count = Counter(s for s, _ in path)
                for sat, count in usage_count.items():
                    sat_load_dict[sat] = max(0, sat_load_dict.get(sat, 0) - count)
                to_delete.append(old_user)
        for u in to_delete:
            active_user_paths.remove(u)

        # === 建圖與路徑選擇
        G = yens_algo.build_graph_for_yens(
            access_matrix, user_id, data_rate_dict_user, sat_load_dict,
            t_start, t_end, 10000, MAX_CHANNELS_PER_SAT
        )
        G_nodes = set(G.nodes)
        source_nodes = [(sat, t_start) for sat in access_matrix[t_start]["visible_sats"] if (sat, t_start) in G_nodes]
        target_nodes = [(sat, t_end) for sat in access_matrix[t_end]["visible_sats"] if (sat, t_end) in G_nodes]

        path, reward = yens_algo.run_yens_algorithm(G, source_nodes, target_nodes, k=2, max_handover=max_handover)
        handover_count = count_handovers(path) if path else float("inf")
        success = handover_count <= max_handover

#       # 🌟不管 success 與否，只要有 path 就要加 load
        if path:
            from collections import Counter
            usage_count = Counter(s for s, _ in path)
            for sat, count in usage_count.items():
                sat_load_dict[sat] += count
            for sat, ts in path:
                load_by_time[ts][sat] += 1
            active_user_paths.append({
                "user_id": user_id,
                "path": path,
                "t_start": t_start,
                "t_end": t_end
            })

        # === 結果紀錄
        all_user_paths.append({
            "user_id": user_id,
            "path": path,
            "t_start": t_start,
            "t_end": t_end,
            "success": success,
            "K": max_handover,
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

    return pd.DataFrame(results), all_user_paths,load_by_time 
###############################################
###############################################
def is_valid_path(path, t_start, t_end):
    times = [ts for _, ts in path]
    return sorted(times) == list(range(t_start, t_end + 1))

def run_dp_per_K(k_col):
    print(f"\n--- Running DP for {k_col} ---")
    MAX_CHANNELS_PER_SAT = 200 
    sat_load_dict = defaultdict(int, deepcopy(sat_load_dict_backup))
    active_user_paths = []  # 專門用來釋放資源
    all_user_paths = []     # 所有有參與的 user（寫進 JSON）
    results = []
    load_by_time = {} 
    for _, user in user_df.iterrows():
        user_id = int(user["user_id"])
        t_start = int(user["t_start"])
        t_end = int(user["t_end"])
        max_handover = int(user[k_col])
        # === ✅ Step 1: 清除已過期使用者的佔用 load
        to_delete = []
        for old_user in active_user_paths:
            if old_user["t_end"] < t_start:
                path = old_user.get("path")
                if path is None:
                    continue
                from collections import Counter
                usage_count = Counter(s for s, _ in path)
                for sat, count in usage_count.items():
                    sat_load_dict[sat] = max(0, sat_load_dict.get(sat, 0) - count)
                to_delete.append(old_user)
        for u in to_delete:
            active_user_paths.remove(u)


        # user 在 [t_start, t_end] 中的可見衛星
        user_visible_sats = {
            row["time_slot"]: row["visible_sats"]
            for row in access_matrix[t_start:t_end + 1]
        }

        graph = dp_algo.build_full_time_expanded_graph(
            user_id=user_id,
            user_visible_sats=user_visible_sats,
            data_rate_dict_user=data_rate_dict_user,
            load_dict=sat_load_dict,
            t_start=t_start,
            t_end=t_end,
            max_channels=MAX_CHANNELS_PER_SAT
        )

        path = []
        reward = 0.0
        try:
            path, reward = dp_algo.dp_k_handover_path_dp_style(
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

        if path:
            from collections import Counter
            usage_count = Counter(s for s, t in path)
            for sat, count in usage_count.items():
                sat_load_dict[sat] += count
            for s, t in path:
                if t not in load_by_time:
                    load_by_time[t] = {}
                load_by_time[t][s] = load_by_time[t].get(s, 0) + 1
        if success:
            active_user_paths.append({
                "user_id": user_id,
                "K": k_col,
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

    return pd.DataFrame(results), all_user_paths,load_by_time
##################run proposed method#####################
def run_proposed_per_K(k_col: str):
    from collections import Counter
    print(f"\n--- Running Proposed for {k_col} ---")
    MAX_CHANNELS_PER_SAT = 200
    T = len(access_matrix)
    sat_load_dict = defaultdict(int, deepcopy(sat_load_dict_backup))
    active_user_paths = []
    all_user_paths = []
    results = []
    #load_by_time是Dict[str, Dict[str, int]]:time_slot -> {sat_name -> load}
    load_by_time = defaultdict(lambda: defaultdict(int))


    for _, user in user_df.iterrows():
        user_id = int(user["user_id"])
        t_start = int(user["t_start"])
        t_end = int(user["t_end"])
        K_limit = int(user[k_col])
        # === ✅ Step 1: 清除已過期使用者的佔用 load
        to_delete = []
        for old_user in active_user_paths:
            if old_user["t_end"] < t_start:
                path = old_user.get("path")
                if path is None:
                    continue
                usage_count = Counter(s for s, _ in path)
                for sat, count in usage_count.items():
                    sat_load_dict[sat] = max(0, sat_load_dict.get(sat, 0) - count)
                to_delete.append(old_user)
        for u in to_delete:
            active_user_paths.remove(u)
        # user 在 [t_start, t_end] 中的可見衛星
        user_visible_sats = {
            row["time_slot"]: row["visible_sats"]
            for row in access_matrix[t_start:t_end + 1]
        }

        path, reward, handover_count, success = run_proposed_method_for_user(
            user_id=user_id,
            t_start=t_start,
            t_end=t_end,
            K_limit=K_limit,
            access_matrix=access_matrix,
            data_rate_dict_user=data_rate_dict_user,
            sat_load_dict=sat_load_dict,
            user_visible_sats=user_visible_sats,
            max_channels_per_sat=MAX_CHANNELS_PER_SAT,
            LAMBDA=1000000
        )

#       # 🌟不管 success 與否，只要有 path 就要加 load
        if path:
            from collections import Counter
            usage_count = Counter(s for s, t in path)
            for sat, count in usage_count.items():
                sat_load_dict[sat] += count
            for s, t in path:
                if t not in load_by_time:
                    load_by_time[t] = {}
                load_by_time[t][s] = load_by_time[t].get(s, 0) + 1

                

        if success:
            active_user_paths.append({
                "user_id": user_id,
                "K": k_col,
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
            "K_limit": K_limit,
            "reward": reward if success else None,
            "handover_count": handover_count,
            "success": success
        })

    return pd.DataFrame(results), all_user_paths, load_by_time

########################比較用#######################
timing_records = []
variance_records = []
#=====Yens 跑每個 K================
dfs_yens = []
all_user_paths_yens = []
all_user_throughput_Yens = {}   # key: (user_id, K)
avg_throughput_per_user_yens = []
for k_col in K_columns:
    start_yen = time.time()
    df_k, paths_k,load_by_time_k = run_baseline_yens_per_K(k_col)
    end_yen = time.time()
    yen_time = end_yen - start_yen
    timing_records.append({
        "K": k_col,
        "method": "Yens",
        "time_sec": yen_time
    })
    dfs_yens.append(df_k)
    all_user_paths_yens.extend(paths_k)
    avg_var = performance_calculate.compute_variance_total_usage(load_by_time_k, access_matrix, T=len(access_matrix))
    print(f"✅ Yens平均負載變異數 for {k_col}: {avg_var:.2f}")
    variance_records.append({
        "K": k_col,
        "method": "Yens",
        "avg_variance": avg_var
    })
    # ========= 紀錄 user throughput =========
    for path_entry in paths_k:
        user_id = path_entry["user_id"]
        path = path_entry["path"]  # list of [sat_id, t]
        handover_count = path_entry.get("handover_count", 0)
        handover_limit = user_df.loc[user_id, k_col]
        user_throughput = {}
        handover_seen = 0
        for i in range(len(path)):
            if i > 0 and path[i][0] != path[i - 1][0]:
                handover_seen += 1
            sat_id, t = path[i]
            if handover_seen > handover_limit:
                user_throughput[t] = 0.0  # 超過 handover 限制的 slot 設為 0
            else:
                throughput = data_rate_dict_user.get(user_id, {}).get((sat_id, t), 0.0)
                user_throughput[t] = throughput

        all_user_throughput_Yens[(user_id, k_col)] = user_throughput
# ========= 統一計算平均 throughput =========
for (user_id, k_col), throughput_dict in all_user_throughput_Yens.items():
    total = sum(throughput_dict.values())
    count = len(throughput_dict)
    avg = total / count if count > 0 else 0.0
    avg_throughput_per_user_yens.append({
        "user_id": user_id,
        "K": k_col,
        "avg_throughput": avg
    })
# 將tuple轉成dict，方便json化
all_user_throughput_Yens_list = [
    {
        "user_id": user_id,
        "K": k,
        "throughput_per_slot": throughput_dict
    }
    for (user_id, k), throughput_dict in all_user_throughput_Yens.items()
]
with open("results/yens_user_throughput.json", "w") as f:
    json.dump(all_user_throughput_Yens_list, f, indent=2)

df_result_yens = pd.concat(dfs_yens, ignore_index=True)
df_result_yens.to_csv("results/baseline_yens_results.csv", index=False)

with open("results/yens_user_paths.json", "w") as f:
    json.dump(all_user_paths_yens, f, indent=2)

df_avg_throughput_yens = pd.DataFrame(avg_throughput_per_user_yens)
df_avg_throughput_yens.to_csv("results/yens_avg_throughput_per_user.csv", index=False)
#=====DP跑每個K================
dfs_dp = []
all_user_paths_dp = []
all_user_throughput_DP = {}   # key: (user_id, K)
avg_throughput_per_user_DP = []
for k_col in K_columns:
    start_dp = time.time()
    df_k, paths_k,load_by_time_k  = run_dp_per_K(k_col)
    end_dp = time.time()
    dp_time = end_dp - start_dp
    timing_records.append({
        "K": k_col,
        "method": "DP",
        "time_sec": dp_time
    })
    dfs_dp.append(df_k)
    all_user_paths_dp.extend(paths_k)
    avg_var = performance_calculate.compute_variance_total_usage(load_by_time_k, access_matrix, T=len(access_matrix))
    print(f"✅ 平均負載變異數 for {k_col}: {avg_var:.2f}")
    variance_records.append({
        "K": k_col,
        "method": "DP",
        "avg_variance": avg_var
    })
    # ========= 在每個 K 下記錄 user 的 throughput =========
    for path_entry in paths_k: #paths_k=user_id, path, t_start, t_end, success, reward, handover_count
        user_id = path_entry["user_id"]
        path = path_entry["path"]  # list of [sat_id, t]
        user_throughput = {}
        for sat_id, t in path:
            key = (user_id, sat_id, t)
            throughput = data_rate_dict_user.get(user_id, {}).get((sat_id, t), 0.0)
            user_throughput[t] = throughput
        all_user_throughput_DP[(user_id, k_col)] = user_throughput #user_throughput={t: throughput},tuple->dict
# ========= K 迴圈結束後再統一計算 avg throughput =========
for (user_id, k_col), throughput_dict in all_user_throughput_DP.items():
    total = sum(throughput_dict.values())
    count = len(throughput_dict)
    avg = total / count if count > 0 else 0.0
    avg_throughput_per_user_DP.append({
        "user_id": user_id,
        "K": k_col,
        "avg_throughput": avg
    })
# 將 tuple 轉成 dict，方便 json 化
all_user_throughput_DP_list = [
    {
        "user_id": user_id,
        "K": k,
        "throughput_per_slot": throughput_dict
    }
    for (user_id, k), throughput_dict in all_user_throughput_DP.items()
]

with open("results/dp_user_throughput.json", "w") as f:
    json.dump(all_user_throughput_DP_list, f, indent=2)
df_avg_throughput = pd.DataFrame(avg_throughput_per_user_DP)
df_avg_throughput.to_csv("results/dp_avg_throughput_per_user.csv", index=False)
df_result_dp = pd.concat(dfs_dp, ignore_index=True)
df_result_dp.to_csv("results/baseline_dp_results.csv", index=False)
with open("results/dp_user_paths.json", "w") as f:
    json.dump(all_user_paths_dp, f, indent=2)
########################  proposed method   ######################################################################
dfs_proposed = []
all_user_paths_proposed = []
all_user_throughput_proposed = {}  # (user_id, K) -> {t: throughput}
avg_throughput_per_user_proposed = []

for k_col in K_columns:
    start = time.time()
    df_k, paths_k,load_by_time_k = run_proposed_per_K(k_col)
    end = time.time()
    proposed_time = end - start

    timing_records.append({
        "K": k_col,
        "method": "PROPOSED",
        "time_sec": proposed_time
    })

    dfs_proposed.append(df_k)
    all_user_paths_proposed.extend(paths_k)

    avg_var = performance_calculate.compute_variance_total_usage(load_by_time_k, access_matrix, T=len(access_matrix))
    print(f"✅ Proposed 平均負載變異數 for {k_col}: {avg_var:.2f}")
    variance_records.append({
        "K": k_col,
        "method": "PROPOSED",
        "avg_variance": avg_var
    })
    for path_entry in paths_k:
        user_id = path_entry["user_id"]
        path = path_entry["path"]
        user_throughput = {}
        for sat_id, t in path:
            rate = data_rate_dict_user.get(user_id, {}).get((sat_id, t), 0.0)
            user_throughput[t] = rate
        all_user_throughput_proposed[(user_id, k_col)] = user_throughput

for (user_id, k_col), throughput_dict in all_user_throughput_proposed.items():
    total = sum(throughput_dict.values())
    count = len(throughput_dict)
    avg = total / count if count > 0 else 0.0
    avg_throughput_per_user_proposed.append({
        "user_id": user_id,
        "K": k_col,
        "avg_throughput": avg
    })

df_avg_throughput = pd.DataFrame(avg_throughput_per_user_proposed)
df_result_proposed = pd.concat(dfs_proposed, ignore_index=True)

# 儲存
with open("results/proposed_user_throughput.json", "w") as f:
    json.dump([
        {"user_id": user_id, "K": k, "throughput_per_slot": d}
        for (user_id, k), d in all_user_throughput_proposed.items()
    ], f, indent=2)

df_avg_throughput.to_csv("results/proposed_avg_throughput_per_user.csv", index=False)
df_result_proposed.to_csv("results/baseline_proposed_results.csv", index=False)

with open("results/proposed_user_paths.json", "w") as f:
    json.dump(all_user_paths_proposed, f, indent=2)
############################################################################################################
########################  TPB method   ######################################################################
dfs_tpb = []
all_user_paths_tpb = []
all_user_throughput_tpb = {}  # (user_id, K) -> {t: throughput}
avg_throughput_per_user_tpb = []

for k_col in K_columns:
    start = time.time()
    df_k, paths_k,load_by_time_k  = tpb_method.run_throughput_based_single_slot(
        k_col=k_col,
        user_df=user_df,
        access_matrix=access_matrix,
        data_rate_dict_user=data_rate_dict_user,
        sat_load_dict_backup=defaultdict(int, deepcopy(sat_load_dict_backup))
    )
    end = time.time()
    tpb_time = end - start

    timing_records.append({
        "K": k_col,
        "method": "TPB",
        "time_sec": tpb_time
    })

    dfs_tpb.append(df_k)
    all_user_paths_tpb.extend(paths_k)

    avg_var = performance_calculate.compute_variance_total_usage(load_by_time_k, access_matrix, T=len(access_matrix))

    print(f"✅ TPB 平均負載變異數 for {k_col}: {avg_var:.2f}")
    variance_records.append({
        "K": k_col,
        "method": "TPB",
        "avg_variance": avg_var
    })
    for path_entry in paths_k:
        user_id = path_entry["user_id"]
        k = k_col
        user_throughput = {}
        path = path_entry["path"] 
        t_start = path_entry["t_begin"]
        t_end = path_entry["t_end"]
        handover_count = path_entry["handover_count"]
        handover_limit = user_df.loc[user_id, k_col]

        if not path:
            for t in range(t_start, t_end + 1):
                user_throughput[t] = 0.0
        else:
            handover_seen = 0
            for i in range(len(path)):
                if i > 0 and path[i][0] != path[i - 1][0]:
                    handover_seen += 1
                sat_id, t = path[i]
                if handover_seen > handover_limit:
                    user_throughput[t] = 0.0
                else:
                    rate = data_rate_dict_user.get(user_id, {}).get((sat_id, t), 0.0)
                    user_throughput[t] = rate

        all_user_throughput_tpb[(user_id, k)] = user_throughput

# 計算每位 user 的平均 throughput
for (user_id, k_col), throughput_dict in all_user_throughput_tpb.items():
    total = sum(throughput_dict.values())
    count = len(throughput_dict)
    avg = total / count if count > 0 else 0.0
    avg_throughput_per_user_tpb.append({
        "user_id": user_id,
        "K": k_col,
        "avg_throughput": avg
    })

df_avg_throughput = pd.DataFrame(avg_throughput_per_user_tpb)
df_result_tpb = pd.concat(dfs_tpb, ignore_index=True)

# 儲存
with open("results/tpb_user_throughput.json", "w") as f:
    json.dump([
        {"user_id": user_id, "K": k, "throughput_per_slot": d}
        for (user_id, k), d in all_user_throughput_tpb.items()
    ], f, indent=2)

df_avg_throughput.to_csv("results/tpb_avg_throughput_per_user.csv", index=False)
df_result_tpb.to_csv("results/baseline_tpb_results.csv", index=False)

with open("results/tpb_user_paths.json", "w") as f:
    json.dump(all_user_paths_tpb, f, indent=2)
###########################
########################  LBB method   ######################################################################
dfs_lbb = []
all_user_paths_lbb = []
all_user_throughput_lbb = {}  # (user_id, K) -> {t: throughput}
avg_throughput_per_user_lbb = []

for k_col in K_columns:
    start = time.time()
    df_k, paths_k, load_by_time_k = lbb_method.run_load_balance_based_single_slot(
        k_col=k_col,
        user_df=user_df,
        access_matrix=access_matrix,
        data_rate_dict_user=data_rate_dict_user,
        sat_load_dict_backup=defaultdict(int, deepcopy(sat_load_dict_backup))
    )
    end = time.time()
    lbb_time = end - start

    timing_records.append({
        "K": k_col,
        "method": "LBB",
        "time_sec": lbb_time
    })

    dfs_lbb.append(df_k)
    all_user_paths_lbb.extend(paths_k)

    avg_var = performance_calculate.compute_variance_total_usage(
        load_by_time_k, access_matrix, T=len(access_matrix)
    )
    print(f"✅ LBB 平均負載變異數 for {k_col}: {avg_var:.2f}")
    variance_records.append({
        "K": k_col,
        "method": "LBB",
        "avg_variance": avg_var
    })
    # 每個 user 的 throughput 記錄
    for path_entry in paths_k:
        user_id = path_entry["user_id"]
        k = k_col
        user_throughput = {}
        path = path_entry["path"] 
        t_start = path_entry["t_begin"]
        t_end = path_entry["t_end"]
        handover_count = path_entry["handover_count"]
        handover_limit = user_df.loc[user_id, k_col]

        if not path:
            for t in range(t_start, t_end + 1):
                user_throughput[t] = 0.0
        else:
            handover_seen = 0
            for i in range(len(path)):
                if i > 0 and path[i][0] != path[i - 1][0]:
                    handover_seen += 1
                sat_id, t = path[i]
                if handover_seen > handover_limit:
                    user_throughput[t] = 0.0
                else:
                    rate = data_rate_dict_user.get(user_id, {}).get((sat_id, t), 0.0)
                    user_throughput[t] = rate

        all_user_throughput_lbb[(user_id, k)] = user_throughput
# 計算每位 user 的平均 throughput
for (user_id, k_col), throughput_dict in all_user_throughput_lbb.items():
    total = sum(throughput_dict.values())
    count = len(throughput_dict)
    avg = total / count if count > 0 else 0.0
    avg_throughput_per_user_lbb.append({
        "user_id": user_id,
        "K": k_col,
        "avg_throughput": avg
    })

df_avg_throughput_lbb = pd.DataFrame(avg_throughput_per_user_lbb)
df_result_lbb = pd.concat(dfs_lbb, ignore_index=True)

# 儲存結果
with open("results/lbb_user_throughput.json", "w") as f:
    json.dump([
        {"user_id": user_id, "K": k, "throughput_per_slot": d}
        for (user_id, k), d in all_user_throughput_lbb.items()
    ], f, indent=2)

df_avg_throughput_lbb.to_csv("results/lbb_avg_throughput_per_user.csv", index=False)
df_result_lbb.to_csv("results/baseline_lbb_results.csv", index=False)

with open("results/lbb_user_paths.json", "w") as f:
    json.dump(all_user_paths_lbb, f, indent=2)
########################  Service Duration-Based  ###########################################
dfs_dura = []
all_user_paths_dura = []
all_user_throughput_dura = {}  # (user_id, K) -> {t: throughput}
avg_throughput_per_user_dura = []

for k_col in K_columns:
    start = time.time()
    df_k, paths_k, load_by_time_k = sdb_method.run_duration_based_single_slot(
        k_col=k_col,
        user_df=user_df,
        access_matrix=access_matrix,
        data_rate_dict_user=data_rate_dict_user,
        sat_load_dict_backup=defaultdict(int, deepcopy(sat_load_dict_backup))
    )
    end = time.time()
    dura_time = end - start

    timing_records.append({
        "K": k_col,
        "method": "SDB",
        "time_sec": dura_time
    })

    dfs_dura.append(df_k)
    all_user_paths_dura.extend(paths_k)

    # 計算 Load Variance
    avg_var = performance_calculate.compute_variance_total_usage(
        load_by_time_k, access_matrix, T=len(access_matrix)
    )
    print(f"✅ SDB 平均負載變異數 for {k_col}: {avg_var:.2f}")
    variance_records.append({
        "K": k_col,
        "method": "SDB",
        "avg_variance": avg_var
    })
    # 每 user 的 throughput 記錄
    for path_entry in paths_k:
        user_id = path_entry["user_id"]
        k = k_col
        user_throughput = {}
        path = path_entry["path"] 
        t_start = path_entry["t_begin"]
        t_end = path_entry["t_end"]
        handover_count = path_entry["handover_count"]
        handover_limit = user_df.loc[user_id, k_col]

        if not path:
            for t in range(t_start, t_end + 1):
                user_throughput[t] = 0.0
        else:
            handover_seen = 0
            for i in range(len(path)):
                if i > 0 and path[i][0] != path[i - 1][0]:
                    handover_seen += 1
                sat_id, t = path[i]
                if handover_seen > handover_limit:
                    user_throughput[t] = 0.0
                else:
                    rate = data_rate_dict_user.get(user_id, {}).get((sat_id, t), 0.0)
                    user_throughput[t] = rate

        all_user_throughput_dura[(user_id, k)] = user_throughput

# 計算平均 throughput per user
for (user_id, k_col), throughput_dict in all_user_throughput_dura.items():
    total = sum(throughput_dict.values())
    count = len(throughput_dict)
    avg = total / count if count > 0 else 0.0
    avg_throughput_per_user_dura.append({
        "user_id": user_id,
        "K": k_col,
        "avg_throughput": avg
    })

df_avg_throughput_dura = pd.DataFrame(avg_throughput_per_user_dura)
df_result_dura = pd.concat(dfs_dura, ignore_index=True)

# === 儲存 ===
with open("results/dura_user_throughput.json", "w") as f:
    json.dump([
        {"user_id": user_id, "K": k, "throughput_per_slot": d}
        for (user_id, k), d in all_user_throughput_dura.items()
    ], f, indent=2)

df_avg_throughput_dura.to_csv("results/dura_avg_throughput_per_user.csv", index=False)
df_result_dura.to_csv("results/baseline_dura_results.csv", index=False)

with open("results/dura_user_paths.json", "w") as f:
    json.dump(all_user_paths_dura, f, indent=2)
############################# mslb method #####################################
dfs_mslb = []
all_user_paths_mslb = []
all_user_throughput_mslb = {}  # (user_id, K) -> {t: throughput}
avg_throughput_per_user_mslb = []

for k_col in K_columns:
    print(f"\n--- Running MSLB for {k_col} ---")
    start = time.time()
    sat_load_dict = defaultdict(int, deepcopy(sat_load_dict_backup))
    active_user_paths = []  # ✅ 為了釋放 load
    user_results = []
    load_by_time_k = defaultdict(lambda: defaultdict(int))  # ✅ 記錄時間點負載

    for _, user in user_df.iterrows():
        user_id = int(user["user_id"])
        t_start = int(user["t_start"])
        t_end = int(user["t_end"])
        handover_limit = int(user[k_col])

        # === ✅ Step 1: 清除過期使用者的佔用 load
        to_delete = []
        for old_user in active_user_paths:
            if old_user["t_end"] < t_start:
                path = old_user.get("path", [])
                usage_count = Counter(s for s, _ in path)
                for sat, count in usage_count.items():
                    sat_load_dict[sat] = max(0, sat_load_dict[sat] - count)
                for sat, t in path:
                    load_by_time_k[t][sat] = max(0, load_by_time_k[t].get(sat, 0) - 1)
                to_delete.append(old_user)
        for u in to_delete:
            active_user_paths.remove(u)

        # === ✅ Step 2: 建圖並找最短路徑
        G = mslb.build_graph_for_user(
            user_id=user_id,
            t_start=t_start,
            t_end=t_end,
            access_matrix=access_matrix,
            data_rate_dict_user=data_rate_dict_user,
            sat_load_dict=sat_load_dict,
            tau=1.0
        )

        path_nodes = nx.shortest_path(G, source="START", target="END", weight="weight")[1:-1]
        path = [(node.split("@")[0], int(node.split("@")[1])) for node in path_nodes]
        handover_count = sum(1 for i in range(1, len(path)) if path[i][0] != path[i - 1][0])
        success = handover_count <= handover_limit
        # === ✅ Step 3: 計算 throughput
        user_throughput = {}
        handover_seen = 0
        reward = 0.0

        if not path:
            for t in range(t_start, t_end + 1):
                user_throughput[t] = 0.0
            reward = None
        else:
            for i in range(len(path)):
                if i > 0 and path[i][0] != path[i - 1][0]:
                    handover_seen += 1
                sat_id, t = path[i]
                if handover_seen > handover_limit:
                    user_throughput[t] = 0.0
                else:
                    rate = data_rate_dict_user.get(user_id, {}).get((sat_id, t), 0.0)
                    user_throughput[t] = rate
                    reward += rate
        # === ✅ Step 4: 紀錄結果
        all_user_paths_mslb.append({
            "user_id": user_id,
            "path": path,
            "t_start": t_start,
            "t_end": t_end,
            "success": success,
            "K": handover_limit,
            "reward": reward,
            "handover_count": handover_count
        })

        user_results.append({
            "user_id": user_id,
            "K": k_col,
            "K_limit": handover_limit,
            "reward": reward,
            "handover_count": handover_count,
            "success": success
        })
        all_user_throughput_mslb[(user_id, k_col)] = user_throughput
        # === ✅ Step 4: 更新負載（sat_load_dict、load_by_time）
        if path:
            for sat, t in path:
                sat_load_dict[sat] += 1
                load_by_time_k[t][sat] += 1
            active_user_paths.append({
                "user_id": user_id,
                "t_end": t_end,
                "path": path
            })
    end = time.time()
    mslb_time = end - start
    timing_records.append({
        "K": k_col,
        "method": "MSLB",
        "time_sec": mslb_time
    })

    avg_var = performance_calculate.compute_variance_total_usage(load_by_time_k, access_matrix, T=len(access_matrix))
    variance_records.append({
        "K": k_col,
        "method": "MSLB",
        "avg_variance": avg_var
    })
    print(f"✅ MSLB 平均負載變異數 for {k_col}: {avg_var:.2f}")

    df_k = pd.DataFrame(user_results)
    dfs_mslb.append(df_k)
    all_user_paths_mslb.extend(user_results)

# === 平均 throughput 統計
for (user_id, k_col), throughput_dict in all_user_throughput_mslb.items():
    total = sum(throughput_dict.values())
    count = len(throughput_dict)
    avg = total / count if count > 0 else 0.0
    avg_throughput_per_user_mslb.append({
        "user_id": user_id,
        "K": k_col,
        "avg_throughput": avg
    })

df_avg_throughput = pd.DataFrame(avg_throughput_per_user_mslb)
df_result_mslb = pd.concat(dfs_mslb, ignore_index=True)

# === 儲存
with open("results/mslb_user_throughput.json", "w") as f:
    json.dump([
        {"user_id": user_id, "K": k, "throughput_per_slot": d}
        for (user_id, k), d in all_user_throughput_mslb.items()
    ], f, indent=2)

df_avg_throughput.to_csv("results/mslb_avg_throughput_per_user.csv", index=False)
df_result_mslb.to_csv("results/mslb_results.csv", index=False)

with open("results/mslb_user_paths.json", "w") as f:
    json.dump(all_user_paths_mslb, f, indent=2)
#############################
# === 成功率統計(Yens) ===
success_rate = (
    df_result_yens.groupby("K")["success"]
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
#===========成功率統計 (Proposed)========
success_rate_proposed = (
    df_result_proposed.groupby("K")["success"]
    .mean()
    .reset_index()
    .rename(columns={"success": "success_rate"})
)
success_rate_proposed["success_rate"] = (success_rate_proposed["success_rate"] * 100).round(2)

print("\n✅ Proposed 每個 K 值的成功率 (%):")
print(success_rate_proposed)
success_rate_proposed.to_csv("results/proposed_success_rate.csv", index=False)
# === 成功率統計(TPB) ===
success_rate_tpb = (
    df_result_tpb.groupby("K")["success"]
    .mean()
    .reset_index()
    .rename(columns={"success": "success_rate"})
)
success_rate_tpb["success_rate"] = (success_rate_tpb["success_rate"] * 100).round(2)

print("\n✅ TPB 每個 K 值的成功率 (%):")
print(success_rate_tpb)

success_rate_tpb.to_csv("results/tpb_success_rate.csv", index=False)
# === 成功率統計 (LBB) ===
success_rate_lbb = (
    df_result_lbb.groupby("K")["success"]
    .mean()
    .reset_index()
    .rename(columns={"success": "success_rate"})
)
success_rate_lbb["success_rate"] = (success_rate_lbb["success_rate"] * 100).round(2)

print("\n✅ LBB 每個 K 值的成功率 (%):")
print(success_rate_lbb)
# === 成功率統計 (SDB) ===
success_rate_dura = (
    df_result_dura.groupby("K")["success"]
    .mean()
    .reset_index()
    .rename(columns={"success": "success_rate"})
)
success_rate_dura["success_rate"] = (success_rate_dura["success_rate"] * 100).round(2)

print("\n✅SDB 每個 K 值的成功率 (%):")
print(success_rate_dura)

success_rate_dura.to_csv("results/dura_success_rate.csv", index=False)
success_rate_lbb.to_csv("results/lbb_success_rate.csv", index=False)
# === 成功率統計 (MSLB) ===
success_rate_mslb = (
    df_result_mslb.groupby("K")["success"]
    .mean()
    .reset_index()
    .rename(columns={"success": "success_rate"})
)
success_rate_mslb["success_rate"] = (success_rate_mslb["success_rate"] * 100).round(2)

print("\n✅ MSLB 每個 K 值的成功率 (%):")
print(success_rate_mslb)

success_rate_mslb.to_csv("results/mslb_success_rate.csv", index=False)
# === 儲存所有的時間紀錄 ===
df_timing = pd.DataFrame(timing_records)
df_timing.to_csv("results/running_time_per_k.csv", index=False)
# === variance 統計 ===
df_var = pd.DataFrame(variance_records)
df_var.to_csv("results/variance_by_method.csv", index=False)
print("✅ 變異數結果已寫入 results/variance_by_method.csv")
##############  draw ###################