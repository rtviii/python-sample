import matplotlib as mpl
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

G = nx.DiGraph()

G.add_node('4af')
G.add_node('5af')
G.add_node('5bf')
G.add_node('5cf')
G.add_node('5rf')
G.add_node('5ff')
G.add_node('6bf')
G.add_node('6af')

G.add_edge('4af', '5af')
G.add_edge('4af', '5bf')
G.add_edge('4af', '5cf')
G.add_edge('4af', '5rf')
G.add_edge('4af', '5ff')
G.add_edge('5ff', '6bf')
G.add_edge('5rf', '6af')

pos = nx.layout.fruchterman_reingold_layout(G)
hashes = {}
stats = {}
for item in pos.items():
    hashes[item[0]] = np.array([ item[1][0], item[1][1]+0.08 ])
for item in pos.items():
    stats[item[0]] = np.array([ item[1][0], item[1][1]-0.08 ])

# node_sizes = [3 + 10 * i for i in range(len(G))]
M = G.number_of_edges()
# edge_colors = range(2, M + 2)
# edge_alphas = [(5 + i) / (M + 4) for i in range(M)]
nodes = nx.draw_networkx_nodes(G, pos, 
# node_size=node_sizes, 
node_color="blue", label=True)
edges = nx.draw_networkx_edges(
    G,
    pos,
    node_size=2,
    arrowstyle="->",
    arrowsize=5,
    # edge_color=edge_colors,
    edge_cmap=plt.cm.Blues,
    width=2,
)
nx.draw_networkx_labels(G,stats)
nx.draw_networkx_labels(G,hashes)

# set alpha value for each edge
# for i in range(M):
#     edges[i].set_alpha(edge_alphas[i])

# pc = mpl.collections.PatchCollection(edges, cmap=plt.cm.Blues)
# pc.set_array(edge_colors)
# plt.colorbar(pc)

ax = plt.gca()
ax.set_axis_off()
plt.show()