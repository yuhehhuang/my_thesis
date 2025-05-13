from typing import Dict, List, Tuple
from dp_algo import build_full_time_expanded_graph, dp_k_handover_path_dp_style
from yens_algo import build_graph_for_yens, run_yens_algorithm
def run_proposed_method_for_user(
    user_id: int,
    t_start: int,
    t_end: int,
    K_limit: int,
    access_matrix,
    data_rate_dict_user,
    sat_load_dict,
    user_visible_sats,
    max_channels_per_sat=200,
    LAMBDA=1000000,
) -> Tuple[List[Tuple[str, int]], float, int, bool]:
    """
    回傳: path, reward, handover_count, success
    """

    # === 試著用 Yen（多路徑、限制換手數）先解看看 ===
    G_yen = build_graph_for_yens(
        access_matrix, user_id, data_rate_dict_user,
        sat_load_dict, t_start, t_end,
        LAMBDA=LAMBDA, MAX_CHANNELS_PER_SAT=max_channels_per_sat
    )

    source_nodes = [(s, t_start) for s in access_matrix[t_start]["visible_sats"] if (s, t_start) in G_yen]
    target_nodes = [(s, t_end) for s in access_matrix[t_end]["visible_sats"] if (s, t_end) in G_yen]

    path_yen, reward_yen = run_yens_algorithm(G_yen, source_nodes, target_nodes, k=2, max_handover=K_limit)

    if path_yen:
        handover_count = sum(1 for i in range(1, len(path_yen)) if path_yen[i][0] != path_yen[i - 1][0])
        return path_yen, reward_yen, handover_count, True

    # === Yen 失敗，再用 DP 強制找恰好 K 換手的最佳路徑 ===
    graph_dp = build_full_time_expanded_graph(
        user_id, user_visible_sats, data_rate_dict_user,
        sat_load_dict, t_start, t_end, max_channels_per_sat
    )

    path_dp, reward_dp = dp_k_handover_path_dp_style(graph_dp, t_start, t_end, K_limit)
    if not path_dp:
        return [], 0.0, 0, False

    handover_count = sum(1 for i in range(1, len(path_dp)) if path_dp[i][0] != path_dp[i - 1][0])
    return path_dp, reward_dp, handover_count, True