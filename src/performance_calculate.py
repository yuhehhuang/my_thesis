def compute_variance_total_usage(sat_load_dict, satellites, T):
    from collections import defaultdict

    total_usage = defaultdict(int)
    for (sat, t), usage in sat_load_dict.items():
        if sat in satellites and 0 <= t < T:
            total_usage[sat] += usage

    ms_list = [total_usage[sat] for sat in satellites]
    mean_m = sum(ms_list) / len(satellites)
    variance = sum((m - mean_m) ** 2 for m in ms_list) / len(satellites)
    return variance
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