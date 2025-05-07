import networkx as nx
from networkx.algorithms.simple_paths import shortest_simple_paths

#需要user_id,date_rate(constant),sat_load,begin,end 
##每個time slot最多兩顆candiate衛星
#data_rate_dict_user u_id-> {(sat, t) -> rate}
# === Yens Graph 建構（改為 sat 為 key） ===
def build_graph_for_yens(access_matrix, user_id, data_rate_dict_user, sat_load_dict,
                         t_start, t_end, LAMBDA=1000000,MAX_CHANNELS_PER_SAT = 200):
    G = nx.DiGraph()
    for t in range(t_start, min(t_end, len(access_matrix) - 1)):
        curr_sats = access_matrix[t]["visible_sats"]
        next_sats = access_matrix[t + 1]["visible_sats"]

        curr_candidates = sorted(curr_sats, key=lambda s: sat_load_dict.get(s, 0))[:2]
        next_candidates = sorted(next_sats, key=lambda s: sat_load_dict.get(s, 0))[:2]

        for s1 in curr_candidates:
            for s2 in next_candidates:
                user_data = data_rate_dict_user.get(user_id, {})
                c_t1 = user_data.get((s2, t+1), 0)
                L = sat_load_dict.get(s2, 0)
                reward = (1 - L / MAX_CHANNELS_PER_SAT) * c_t1
                if reward > 0:
                    weight = max(LAMBDA - reward, 0)
                    G.add_edge((s1, t), (s2, t+1), weight=weight, raw_reward=reward)

    return G
# === Yens Path 選擇 ===
def run_yens_algorithm(G, source_nodes, target_nodes, k=2, max_handover=float('inf')):
    candidate_paths = []

    for src in source_nodes:
        for tgt in target_nodes:
            try:
                paths = shortest_simple_paths(G, src, tgt, weight='weight')
                for i, path in enumerate(paths):
                    if i >= k:
                        break
                    reward = sum(G[u][v]["raw_reward"] for u, v in zip(path[:-1], path[1:]))
                    handovers = sum(1 for (s1, _), (s2, _) in zip(path[:-1], path[1:]) if s1 != s2)
                    candidate_paths.append((path, reward, handovers))
            except nx.NetworkXNoPath:
                continue

    # 優先回傳第一個 handover 合法的路徑
    for path, reward, handovers in sorted(candidate_paths, key=lambda x: -x[1]):
        if handovers <= max_handover:
            return path, reward

    # 若沒有一條合法，則回傳分數最高的路徑（即使違反換手限制）
    if candidate_paths:
        path, reward, _ = max(candidate_paths, key=lambda x: x[1])
        return path, reward

    return None, 0.0