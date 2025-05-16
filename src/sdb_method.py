from typing import Dict, List, Tuple
from collections import defaultdict
import pandas as pd
import copy
import random
def run_duration_based_single_slot(
    k_col: str,
    user_df,
    access_matrix,
    data_rate_dict_user,
    sat_load_dict_backup,
    max_channels_per_sat: int = 200
) -> Tuple[pd.DataFrame, List[Dict], Dict[int, Dict[str, int]]]:

    print(f"\n--- Running Duration-Based (Single-slot) for {k_col} ---")
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

        # 清除過期使用者佔用
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

        t = t_start
        while t <= t_end:
            visible_sats = access_matrix[t]["visible_sats"]
            best_sat = None
            best_sats = []
            best_duration = 0

            # 找出從 t 起連續可見時間最久的衛星
            for sat in visible_sats:
                dur = 0
                t_tmp = t
                while t_tmp <= t_end and sat in access_matrix[t_tmp]["visible_sats"]:
                    dur += 1
                    t_tmp += 1
                if dur > best_duration:
                    best_duration = dur
                    best_sats = [sat]  
                elif dur == best_duration:
                    best_sats.append(sat)

            if not best_sats:
                best_sat = None
                success = False
                break
            else:
                best_sat = random.choice(best_sats)

            # 如果換了衛星就算 handover
            if prev_sat is not None and best_sat != prev_sat:
                handover_count += 1
            prev_sat = best_sat

            # 使用這顆衛星直到不可見
            for i in range(best_duration):
                curr_t = t + i
                if curr_t > t_end:
                    break
                rate = data_rate_dict_user.get(user_id, {}).get((best_sat, curr_t), 0.0)
                curr_load = sat_load_dict[best_sat] / max_channels_per_sat
                slot_reward = (1 - curr_load) * rate
                reward += slot_reward
                path.append((best_sat, curr_t))

                sat_load_dict[best_sat] += 1
                if curr_t not in load_by_time:
                    load_by_time[curr_t] = {}
                load_by_time[curr_t][best_sat] = load_by_time[curr_t].get(best_sat, 0) + 1

            t += best_duration

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