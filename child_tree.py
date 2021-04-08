
import matplotlib.pyplot as plt
import networkx as nx


from networkx.drawing.nx_agraph import write_dot, graphviz_layout
G = nx.DiGraph()
G.add_node("ROOT")



G.add_node('4af')
G.add_node('5af')
G.add_node('5bf')
G.add_node('5cf')
G.add_node('5rf')
G.add_node('5ff')
G.add_node('6bf')
G.add_node('6af')

G.add_edge('ROOT', '4af')
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
nodeFontSize    = 10
nodeSize        = 20
# nodeColorList   = list(nx.getnodegetNodeColor(nodesAttrDic,G.nodes()))
# edgeColorList   = getEdgeColor(G.edges())

# Graphiz tunning
prog  =  'dot'
args  =  '-Gnodesep=1 -Granksep=2 -Gpad=0.5 -Grankdir=TD'
root  =  None
pos   =  graphviz_layout(G, prog = prog, root = root, args = args)

plt.title('draw_networkx')
pos=graphviz_layout(G, prog='dot')
nx.draw(G, pos, with_labels=True, arrows=True)
plt.show()
# plt.savefig('nx_test.png')