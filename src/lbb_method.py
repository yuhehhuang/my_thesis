from typing import Dict, List, Tuple
from collections import defaultdict
import pandas as pd
import copy
def run_load_balance_based_single_slot(
    k_col: str,
    user_df,
    access_matrix,
    data_rate_dict_user,
    sat_load_dict_backup,
    max_channels_per_sat: int = 200
) -> Tuple[pd.DataFrame, List[Dict], Dict[int, Dict[str, int]]]:

    print(f"\n--- Running Load-Balance-Based (Single-slot) for {k_col} ---")
    T = len(access_matrix)
    sat_load_dict = defaultdict(int, copy.deepcopy(sat_load_dict_backup))
    active_user_paths = []
    all_user_paths = []
    results = []
    load_by_time = {}

    for _, user in user_df.iterrows():
        user_id = int(user["user_id"])
        t_start = int(user["t_start"])
        t_end = int(user["t_end"])
        K_limit = int(user[k_col])

        # 1️⃣ 釋放已完成 user 的資源
        to_delete = []
        for old_user in active_user_paths:
            if old_user["t_end"] < t_start:
                for s, t in old_user["path"]:
                    sat_load_dict[s] = max(0, sat_load_dict[s] - 1)
                to_delete.append(old_user)
        for u in to_delete:
            active_user_paths.remove(u)

        path = []
        reward = 0.0
        prev_sat = None
        handover_count = 0
        success = True

        # 2️⃣ 為每個時槽選擇負載最低的可見衛星
        for t in range(t_start, t_end + 1):
            visible_sats = access_matrix[t]["visible_sats"]
            best_sat = None
            min_load = float("inf")

            for sat in visible_sats:
                load = sat_load_dict.get(sat, 0)
                if load < min_load:
                    min_load = load
                    best_sat = sat

            if best_sat is None:
                success = False
                break

            # 更新 handover 次數
            if prev_sat is not None and best_sat != prev_sat:
                handover_count += 1
            prev_sat = best_sat

            rate = data_rate_dict_user.get(user_id, {}).get((best_sat, t), 0.0)
            curr_load = sat_load_dict[best_sat] / max_channels_per_sat
            slot_reward = (1 - curr_load) * rate
            reward += slot_reward
            path.append((best_sat, t))

            # 更新衛星負載
            sat_load_dict[best_sat] += 1
            if t not in load_by_time:
                load_by_time[t] = {}
            load_by_time[t][best_sat] = load_by_time[t].get(best_sat, 0) + 1

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

    return pd.DataFrame(results), all_user_paths, load_by_time