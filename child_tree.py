import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

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



# write dot file to use with graphviz
# run "dot -Tpng test.dot >test.png"
nx.nx_agraph.write_dot(G,'test.dot')
# Tune plot
nodeFontSize    = 5
nodeSize        = 20
pos   =  nx.layout.fruchterman_reingold_layout(G)
# nodeColorList   = list(nx.getnodegetNodeColor(nodesAttrDic,G.nodes()))
# edgeColorList   = getEdgeColor(G.edges())

hashes = {}
stats = {}
for item in pos.items():
    hashes[item[0]] = np.array([ item[1][0], item[1][1]+0.04 ])
for item in pos.items():
    stats[item[0]] = np.array([ item[1][0], item[1][1]-0.04 ])
nodes = nx.draw_networkx_nodes(
    G, pos, 
node_size=4, 
# node_size=nodeSize,
node_color="red",
label=True)

edges = nx.draw_networkx_edges(
    G,
    pos,
    arrowstyle="->",
    arrowsize=5,
    # edge_color=edge_colors,
    edge_cmap=plt.cm.Blues,
    width=1,
)
# Graphiz tunning

# set alpha value for each edge
for i in range(M):
    edges[i].set_alpha(edge_alphas[i])

for x in [stats,hashes]:
    nx.draw_networkx_labels(G,x,
    font_size=6
    
    )
plt.title('draw_networkx')
plt.show()
# pos=graphviz_layout(G, prog='dot')
# nx.draw(G, pos, with_labels=True, arrows=True)
# plt.savefig('nx_test.png')