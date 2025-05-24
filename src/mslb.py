import networkx as nx
import pandas as pd
import networkx as nx
import copy
from collections import defaultdict
def build_graph_for_user(
    user_id: int,
    t_start: int,
    t_end: int,
    access_matrix: list,
    data_rate_dict_user: dict,
    sat_load_dict: dict,
    tau: float = 1.0
):
    G = nx.DiGraph()
    start_node = "START"
    end_node = "END"
    G.add_node(start_node)
    G.add_node(end_node)

    for t in range(t_start, t_end):
        visible_sats = access_matrix[t]["visible_sats"]
        for sat in visible_sats:
            service_duration = 1
            for t_next in range(t + 1, t_end + 1):
                if sat in access_matrix[t_next]["visible_sats"]:
                    service_duration += 1
                else:
                    break

            total_rate = 0
            for i in range(t + 1, t + service_duration + 1):
                total_rate += data_rate_dict_user.get(user_id, {}).get((sat, i), 0.0)
            avg_rate = total_rate / service_duration if service_duration > 0 else 0

            reward = avg_rate * service_duration * tau
            sat_load = sat_load_dict.get(sat, 0.0)
            adjusted_reward = reward * (1 - sat_load)

            node = f"{sat}@{t}"
            G.add_node(node)

            if t == t_start:
                G.add_edge(start_node, node, weight=-adjusted_reward)

            end_t = t + service_duration
            if end_t >= t_end:
                G.add_edge(node, end_node, weight=0)
            else:
                next_visible_sats = access_matrix[end_t]["visible_sats"]
                for next_sat in next_visible_sats:
                    next_node = f"{next_sat}@{end_t}"
                    G.add_node(next_node)
                    G.add_edge(node, next_node, weight=-adjusted_reward)

    return G
