import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# 读取 CSV 数据
data = pd.read_csv('StandWalkJump_graph.csv', header=None, names=['X', 'Y'])

# 创建图
G = nx.Graph()

# 添加节点和边
for _, row in data.iterrows():
    G.add_edge(row['X'], row['Y'])

# 获取每条边的权重（边的出现次数作为权重）
edge_weights = nx.get_edge_attributes(G, 'weight')
if not edge_weights:
    edge_weights = {(u, v): 1 for u, v in G.edges()}

# 将边权重设置为边的粗细
nx.set_edge_attributes(G, edge_weights, 'weight')

# 获取系统字体
prop = fm.FontProperties(fname='/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf')

# 绘制网络图
plt.figure(figsize=(12, 12))

# 使用 spring 布局
pos = nx.spring_layout(G, seed=42)

# 绘制节点
nx.draw_networkx_nodes(G, pos, node_size=500, node_color='skyblue')

# 绘制边
nx.draw_networkx_edges(G, pos, width=[edge_weights.get((u, v), 1) for u, v in G.edges()], alpha=0.6)

# 绘制标签
nx.draw_networkx_labels(G, pos, font_size=10, font_family='sans-serif')

# 添加边标签
edge_labels = {(u, v): f'{edge_weights.get((u, v), 1)}' for u, v in G.edges()}
nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_family='sans-serif')

plt.title('Network Graph with Edge Weights')
plt.savefig("dtw.pdf", bbox_inches="tight")
plt.show()
