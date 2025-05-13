######load_by_user_time:Dict[user_id][t][sat]
from collections import defaultdict
def compute_variance_latest_snapshot(
    load_by_user_time,     # Dict[user_id][t][sat]
    access_matrix,         # time slot → visible_sats
    T,                     # 總時間長度
    latest_user_for_t      # Dict[t] → 最後一個處理該 t 的 user_id
):
    total_var = 0
    valid_t = 0

    for t in range(T):
        visible_sats = access_matrix[t]["visible_sats"]
        if not visible_sats or t not in latest_user_for_t:
            continue

        user_id = latest_user_for_t[t]
        snapshot = load_by_user_time.get(user_id, {}).get(t, {})
        if not snapshot:
            continue

        loads = [snapshot.get(sat, 0) for sat in visible_sats]
        avg = sum(loads) / len(loads)
        var = sum((x - avg) ** 2 for x in loads) / len(loads)
        total_var += var
        valid_t += 1

    return total_var / valid_t if valid_t else 0
def save_success_rate(df_result, method_name):
    success_rate = (
        df_result.groupby("K")["success"]
        .mean()
        .reset_index()
        .rename(columns={"success": "success_rate"})
    )
    success_rate["success_rate"] = (success_rate["success_rate"] * 100).round(2)
    print(f"\n✅ 每個 K 值的成功率 (%): [{method_name}]")
    print(success_rate)
    success_rate.to_csv(f"results/{method_name}_success_rate.csv", index=False)