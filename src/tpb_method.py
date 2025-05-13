# tpb_method.py
from typing import Dict, List, Tuple
from collections import defaultdict
import pandas as pd
import copy
def run_throughput_based_single_slot(
    k_col: str,
    user_df,
    access_matrix,
    data_rate_dict_user,
    sat_load_dict_backup,
    max_channels_per_sat: int = 200
) -> Tuple[pd.DataFrame, List[Dict], Dict[int, Dict[int, Dict[str, int]]], Dict[int, int]]:
    print(f"\n--- Running TPB (Single-slot Throughput Based) for {k_col} ---")
    T = len(access_matrix)
    sat_load_dict = defaultdict(int, copy.deepcopy(sat_load_dict_backup))
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
        # === Step 1️⃣: 清除過期 user 的 sat_load
        to_delete = []
        for old_user in active_user_paths:
            if old_user["t_end"] < t_start:
                for s, t in old_user["path"]:
                    sat_load_dict[s] = max(0, sat_load_dict[s] - 1)
                to_delete.append(old_user)
        for u in to_delete:
            active_user_paths.remove(u)
        # === Step 2️⃣: snapshot → user_id 視角
        for t in range(t_start, t_end + 1):
            latest_user_for_t[t] = user_id
            for old_user in active_user_paths:
                if old_user["t_begin"] <= t <= old_user["t_end"]:
                    for s, t_ in old_user.get("path", []):
                        if t_ == t:
                            load_by_user_time[user_id][t][s] += 1

        path = []
        reward = 0.0
        prev_sat = None
        handover_count = 0
        success = True

        for t in range(t_start, t_end + 1):
            visible_sats = access_matrix[t]["visible_sats"]
            best_sat = None
            best_rate = -1
            for sat in visible_sats:
                rate = data_rate_dict_user.get(user_id, {}).get((sat, t), 0.0)
                if rate > best_rate:
                    best_rate = rate
                    best_sat = sat

            if best_sat is None:
                success = False
                break

            if prev_sat is not None and best_sat != prev_sat:
                handover_count += 1
            prev_sat = best_sat
            reward += best_rate
            path.append((best_sat, t))
            sat_load_dict[best_sat] += 1
            load_by_user_time[user_id][t][best_sat] += 1  # ✅ 加上自己的佔用

        # 若換手次數超過限制，則視為失敗
        if handover_count > K_limit:
            success = False

        user_path_entry = {
            "user_id": user_id,
            "K": k_col,
            "path": path,
            "t_begin": t_start,
            "t_end": t_end,
            "success": success,
            "reward": reward,
            "handover_count": handover_count
        }
        all_user_paths.append(user_path_entry)

        results.append({
            "user_id": user_id,
            "K": k_col,
            "K_limit": K_limit,
            "reward": reward if success else None,
            "handover_count": handover_count if success else float("inf"),
            "success": success
        })

        if success:
            active_user_paths.append(user_path_entry)

    return pd.DataFrame(results), all_user_paths, load_by_user_time,latest_user_for_t
