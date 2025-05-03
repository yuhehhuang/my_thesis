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