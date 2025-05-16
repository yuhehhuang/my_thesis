from collections import defaultdict
from typing import Dict, List, Tuple
import pandas as pd
import copy

def run_throughput_based_single_slot(
    k_col: str,
    user_df: pd.DataFrame,
    access_matrix: List[Dict],
    data_rate_dict_user: Dict[int, Dict[Tuple[str, int], float]],
    sat_load_dict_backup: Dict[str, int],
    max_channels_per_sat: int = 200
) -> Tuple[pd.DataFrame, List[Dict], Dict[int, Dict[str, int]]]:
    print(f"\n--- Running Throughput-Based (Single-slot) for {k_col} ---")

    T = len(access_matrix)
    sat_load_dict = copy.deepcopy(sat_load_dict_backup)
    load_by_time = defaultdict(dict)

    results = []
    all_user_paths = []

    for _, user in user_df.iterrows():
        user_id = int(user["user_id"])
        t_start = int(user["t_start"])
        t_end = int(user["t_end"])
        max_handover = int(user[k_col])

        path = []
        reward = 0
        success = True
        handover_count = 0
        prev_sat = None

        for t in range(t_start, t_end + 1):
            visible_sats = access_matrix[t]["visible_sats"]
            best_sat = None
            best_rate = -1

            for sat in visible_sats:
                usage = sat_load_dict.get(sat, 0)
                if usage >= max_channels_per_sat:
                    continue

                rate = data_rate_dict_user.get(user_id, {}).get((sat, t), 0)
                if rate > best_rate:
                    best_rate = rate
                    best_sat = sat

            if best_sat is None:
                success = False
                break

            # handover count
            if prev_sat is not None and best_sat != prev_sat:
                handover_count += 1
            prev_sat = best_sat

            # 更新負載
            sat_load_dict[best_sat] += 1
            load_by_time[t][best_sat] = load_by_time[t].get(best_sat, 0) + 1

            reward += best_rate
            path.append([best_sat, t])

        result = {
            "user_id": user_id,
            "K": k_col,
            "K_limit": max_handover,
            "reward": reward if success else None,
            "handover_count": handover_count,
            "success": success
        }
        results.append(result)

        all_user_paths.append({
            "user_id": user_id,
            "path": path,
            "t_start": t_start,
            "t_end": t_end,
            "success": success,
            "reward": reward,
            "handover_count": handover_count
        })

    df_result = pd.DataFrame(results)
    return df_result, all_user_paths, load_by_time