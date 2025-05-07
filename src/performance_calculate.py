######load_by_time[t][sat]=所有usery在time t 總共使用的此sat的channel數量
def compute_variance_total_usage(load_by_time, access_matrix, T):
    
    total_var = 0
    for t in range(T):
        visible_sats = access_matrix[t]["visible_sats"]
        if not visible_sats:
            continue
        loads = [load_by_time[t].get(sat, 0) for sat in visible_sats] #抓取time t可見衛星的使用channel 數
        avg = sum(loads) / len(loads) #time t所有可見衛星的平均使用channel 
        var = sum((x - avg) ** 2 for x in loads) / len(loads)
        total_var += var
    return total_var / T
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