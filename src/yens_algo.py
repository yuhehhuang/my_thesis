import networkx as nx
from networkx.algorithms.simple_paths import shortest_simple_paths

#需要user_id,date_rate(constant),sat_load,begin,end 
##每個time slot最多兩顆candiate衛星
#data_rate_dict_user u_id-> {(sat, t) -> rate}
def build_graph_for_yens(access_matrix, user_id, data_rate_dict_user, sat_load_dict,
                         t_start, t_end, LAMBDA=1000000):
    G = nx.DiGraph()

    for t in range(t_start, min(t_end, len(access_matrix) - 1)):
        curr_sats = access_matrix[t]["visible_sats"]
        next_sats = access_matrix[t + 1]["visible_sats"]
        for s in curr_sats:
            val = sat_load_dict.get((s, t))
            if val is None:
                raise ValueError(f"❌ sat_load_dict[({s}, {t})] = None before sorting curr_sats")       
        # 選出當下與下一個時刻，負載最小的兩顆衛星（可見且一定可用）
        curr_candidates = sorted(
            curr_sats,
            key=lambda s: sat_load_dict.get((s, t))
        )[:2]

        next_candidates = sorted(
            next_sats,
            key=lambda s: sat_load_dict.get((s, t+1))
        )[:2]

        # 建立 edge
        for s1 in curr_candidates:
            for s2 in next_candidates:
                # ✅ 巢狀格式取值：先確認 user_id 存在

                user_data = data_rate_dict_user.get(user_id, {})
                c_t1 = user_data.get((s2, t+1), 0)

                L = sat_load_dict.get(s2, t+1)
                reward = (1 - L) * c_t1
                if reward > 0:
                    weight = max(LAMBDA - reward, 0)
                    G.add_edge((s1, t), (s2, t+1),
                               weight=weight,
                               raw_reward=reward)

    return G

def run_yens_algorithm(G, source_nodes, target_nodes, k=2):
    best_path = None
    best_reward = -float('inf')
    for src in source_nodes:
        for tgt in target_nodes:
            try:
                paths = shortest_simple_paths(G, src, tgt, weight='weight')
                for i, path in enumerate(paths):
                    if i >= k:
                        break
                    reward = sum(G[u][v]["raw_reward"] for u, v in zip(path[:-1], path[1:]))
                    if reward > best_reward:
                        best_reward = reward
                        best_path = path
            except nx.NetworkXNoPath:
                continue
    return best_path, best_reward