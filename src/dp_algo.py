
from typing import Dict, List, Tuple
#user_visible_sats: time slot -> List[sat]
#data_rate_dict_user u_id-> {(sat, t) -> rate}
# #load_dict: (time slot,satellite) -> load
#回傳:time->reward(此time的), prev_candidates
def build_full_time_expanded_graph(
    user_id: int,
    user_visible_sats: Dict[int, List[str]],
    data_rate_dict_user: Dict[int, Dict[Tuple[str, int], float]],
    load_dict: Dict[Tuple[int, str], float],
    t_start: int,
    t_end: int,
) -> Dict[int, Dict[str, Dict]]:
    """
    建立完整 DP 用的 time-expanded graph。
    graph[t][sat] = {
        'reward': reward_value,
        'prev_candidates': List[str]  # 從 t-1 可以轉移過來的 sat
    }
    """
    graph = {}
    for t in range(t_start, t_end + 1):
        graph[t] = {}
        current_sats = user_visible_sats.get(t, [])
        for s in current_sats:
            rate = data_rate_dict_user[user_id].get((s, t), 0.0)
            load = load_dict.get((t, s), 0.0)
            reward = (1 - load) * rate
            graph[t][s] = {
                "reward": reward,
                "prev_candidates": []
            }

    # 建立 prev_candidates：從 t-1 到 t 能轉移的衛星
    for t in range(t_start + 1, t_end + 1):
        for s_now in graph[t]:
            graph[t][s_now]["prev_candidates"] = [
                s_prev for s_prev in graph[t - 1]  # 所有 t-1 的衛星
                if True  # 若有需要可加 transition 限制
            ]

    return graph


def dp_k_handover_path_dp_style(
    graph: Dict[int, Dict[str, Dict]],
    t_start: int,
    t_end: int,
    K: int
) -> Tuple[List[Tuple[str, int]], float]:
    """
path=[("SAT-001", 5), ("SAT-001", 6), ("SAT-003", 7)]是一個 list，裡面每一個元素是 (satellite_name, time_slot)=>list of tuples
culmulated reward => float
    """
    dp = {}  # dp[t][s][k] = reward
    path = {}  # path[t][s][k] = prev_s

    for t in range(t_start, t_end + 1):
        dp[t] = {}
        path[t] = {}
        for s in graph[t]:
            dp[t][s] = {}
            path[t][s] = {}
            for k in range(K + 1): 
                dp[t][s][k] = float("-inf")
                path[t][s][k] = None
    # 初始化 t_start 時刻所有衛星
    for s in graph[t_start]:
        for k in range(K + 1): 
            dp[t_start][s][k] = graph[t_start][s]["reward"]
            path[t_start][s][k] = None

    for t in range(t_start + 1, t_end + 1):
        for s_now in graph[t]:  # 當前時間 t 可用的衛星
            for s_prev in graph[t][s_now]["prev_candidates"]:  # 從 t-1 轉移過來的衛星
                for k in range(K + 1):  # 嘗試所有 k（目前手上這格的 k）

# Case I: No handover
                    if s_now == s_prev:
                        reward = dp[t - 1][s_prev][k] + graph[t][s_now]["reward"]
                        if reward > dp[t][s_now][k]:
                            dp[t][s_now][k] = reward
                            path[t][s_now][k] = s_prev

                    # Case II: Handover
                    elif k > 0:
                        reward = dp[t - 1][s_prev][k - 1] + graph[t][s_now]["reward"]
                        if reward > dp[t][s_now][k]:
                            dp[t][s_now][k] = reward
                            path[t][s_now][k] = s_prev

    # 從 t_end 找出最大 reward 與終點
    best_reward = -1
    best_end = None
    best_k = -1
    for s in dp[t_end]:
        for k in dp[t_end][s]:
            if k <= K and dp[t_end][s][k] > best_reward:
                best_reward = dp[t_end][s][k]
                best_end = s
                best_k = k
        # 回推路徑
    best_path = []
    if best_end is None:
        return [], 0.0
    t = t_end
    s = best_end
    k = best_k
    while t >= t_start:
        best_path.append((s, t))
        s_prev = path[t][s][k]
        if s_prev is None:
            break
        if s_prev != s:
            k -= 1
        s = s_prev
        t -= 1
    best_path.reverse()
    # 允許 t_start 到 t_end 每個時間點都要剛好在 path 中
    if len(best_path) != (t_end - t_start + 1):
        return [], 0.0
    return best_path, best_reward