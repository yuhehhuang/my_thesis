import sys
sys.path.append("src")
import pandas as pd
import ast  # 為了安全轉換字串成 list,Abstract Syntax Trees
import random
import pickle
from collections import defaultdict,Counter
from copy import deepcopy
import time
import os
import json
import yens_algo
import dp_algo
import performance_calculate
from proposed_method import run_proposed_method_for_user
import tpb_method

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
    max_background_users=150
    )

K_columns = [f"K{i}" for i in range(3)]
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
    latest_user_for_t = {} 
    # 記錄每個 time slot 衛星的 snapshot 負載（包含其他仍 active 的 user）
    load_by_user_time = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))  # user_id → t → sat → load

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
        #=== ✅ Step 2: 記錄 snapshot → 當前 user 的視角
        for t in range(t_start, t_end + 1):
            latest_user_for_t[t] = user_id
            for old_user in active_user_paths:
                if old_user["t_start"] <= t <= old_user["t_end"]:
                    for s, t_ in old_user.get("path", []):
                        if t_ == t:
                            load_by_user_time[user_id][t][s] += 1 

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

        # === 📌 不管成功與否，只要有 path 就要更新負載
        if path:
            from collections import Counter
            usage_count = Counter(s for s, _ in path)
            for sat, count in usage_count.items():
                sat_load_dict[sat] += count
            for sat, t in path:
                load_by_user_time[user_id][t][sat] += 1 

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

    return pd.DataFrame(results), all_user_paths, load_by_user_time, latest_user_for_t
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
    latest_user_for_t = {} 
    load_by_user_time = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))  # user_id → t → sat → load
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

        # === ✅ Step 2: 記錄 snapshot → 當前 user 的視角
        for t in range(t_start, t_end + 1):
            latest_user_for_t[t] = user_id
            for old_user in active_user_paths:
                if old_user["t_begin"] <= t <= old_user["t_end"]:
                    for s, t_ in old_user.get("path", []):
                        if t_ == t:
                            load_by_user_time[user_id][t][s] += 1  # ✅ 用 current user 為 key

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

        # === ✅ 若有 path，更新 sat_load_dict 與 snapshot load
        if path:
            from collections import Counter
            usage_count = Counter(s for s, t in path)
            for sat, count in usage_count.items():
                sat_load_dict[sat] += count
            for s, t in path:
                 load_by_user_time[user_id][t][s] += 1   # 疊加 snapshot，不覆蓋
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

    return pd.DataFrame(results), all_user_paths,load_by_user_time,latest_user_for_t
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
    latest_user_for_t = {}
    load_by_user_time = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))  # user_id → t → sat → load


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
        # === ✅ Step 2: 記錄 snapshot → 當前 user 的視角
        for t in range(t_start, t_end + 1):
            latest_user_for_t[t] = user_id
            for old_user in active_user_paths:
                if old_user["t_begin"] <= t <= old_user["t_end"]:
                    for s, t_ in old_user.get("path", []):
                        if t_ == t:
                            load_by_user_time[user_id][t][s] += 1

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

        if path:
            usage_count = Counter(s for s, _ in path)
            for sat, count in usage_count.items():
                sat_load_dict[sat] += count
            for s, t in path:
                load_by_user_time[user_id][t][s] += 1  # ✅ 將自己的使用也記錄進來
                

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

    return pd.DataFrame(results), all_user_paths, load_by_user_time,latest_user_for_t

########################比較用#######################

timing_records = []
#=====Yens 跑每個 K================
dfs_yens = []
all_user_paths_yens = []
all_user_throughput_Yens = {}   # key: (user_id, K)
avg_throughput_per_user_yens = []
for k_col in K_columns:
    start_yen = time.time()
    df_k, paths_k,load_by_time_yen_k,latest_user_for_t_k = run_baseline_yens_per_K(k_col)
    end_yen = time.time()
    yen_time = end_yen - start_yen
    timing_records.append({
        "K": k_col,
        "method": "Yens",
        "time_sec": yen_time
    })
    dfs_yens.append(df_k)
    all_user_paths_yens.extend(paths_k)
    avg_var = performance_calculate.compute_variance_latest_snapshot(load_by_time_yen_k, access_matrix,len(access_matrix) , latest_user_for_t_k)
    print(f"✅ Yens平均負載變異數 for {k_col}: {avg_var:.2f}")
    # ========= 紀錄 user throughput =========
    for path_entry in paths_k:
        user_id = path_entry["user_id"]
        path = path_entry["path"]  # list of [sat_id, t]
        handover_count = path_entry.get("handover_count", 0)
        handover_limit = user_df.loc[user_id, k_col]
        user_throughput = {}

        if handover_count > handover_limit:
            # 超出 handover 限制，全部 throughput 當作 0
            for _, t in path:
                user_throughput[t] = 0.0
        else:
            # 正常計算 throughput
            for sat_id, t in path:
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
    df_k, paths_k,load_by_time_dp_k,latest_user_for_t_k  = run_dp_per_K(k_col)
    end_dp = time.time()
    dp_time = end_dp - start_dp
    timing_records.append({
        "K": k_col,
        "method": "DP",
        "time_sec": dp_time
    })
    dfs_dp.append(df_k)
    all_user_paths_dp.extend(paths_k)
    avg_var = performance_calculate.compute_variance_latest_snapshot(
    load_by_time_dp_k, access_matrix, len(access_matrix), latest_user_for_t_k)
    print(f"✅ 平均負載變異數 for {k_col}: {avg_var:.2f}")
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
    df_k, paths_k, load_by_time_k,latest_user_for_t_k = run_proposed_per_K(k_col)
    end = time.time()
    proposed_time = end - start

    timing_records.append({
        "K": k_col,
        "method": "PROPOSED",
        "time_sec": proposed_time
    })

    dfs_proposed.append(df_k)
    all_user_paths_proposed.extend(paths_k)

    avg_var = performance_calculate.compute_variance_latest_snapshot(
        load_by_time_k, access_matrix,len(access_matrix), latest_user_for_t_k
    )
    print(f"✅ Proposed 平均負載變異數 for {k_col}: {avg_var:.2f}")

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
    df_k, paths_k, load_by_time_k,latest_user_for_t_k = tpb_method.run_throughput_based_single_slot(
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

    avg_var = performance_calculate.compute_variance_latest_snapshot(
        load_by_time_k, access_matrix, len(access_matrix), latest_user_for_t_k
    )
    print(f"✅ TPB 平均負載變異數 for {k_col}: {avg_var:.2f}")

    for path_entry in paths_k:
        user_id = path_entry["user_id"]
        path = path_entry["path"]
        user_throughput = {}
        for sat_id, t in path:
            rate = data_rate_dict_user.get(user_id, {}).get((sat_id, t), 0.0)
            user_throughput[t] = rate
        all_user_throughput_tpb[(user_id, k_col)] = user_throughput

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
# === 儲存所有的時間紀錄 ===
df_timing = pd.DataFrame(timing_records)
df_timing.to_csv("results/running_time_per_k.csv", index=False)