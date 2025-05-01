import networkx as nx
import matplotlib.pyplot as plt

G = nx.DiGraph()  # 若有方向建議用 DiGraph

# 加入幾條有 tuple 節點的邊
G.add_edge(("S1", 0), ("S2", 1), weight=5.0, raw_reward=10.0)
G.add_edge(("S2", 1), ("S3", 2), weight=3.0, raw_reward=8.0)

# 計算節點位置
pos = nx.spring_layout(G)  # 可換成 shell_layout 等

# 畫節點與邊
nx.draw(G, pos, with_labels=True, node_size=700)

# 畫出邊的 "weight" 屬性
edge_weights = nx.get_edge_attributes(G, 'weight')
nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_weights)

plt.title("Graph with (satellite, time) nodes and weights")
plt.show()