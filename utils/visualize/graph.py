import networkx as nx
import matplotlib.pyplot as plt
from pyvis.network import Network

from core.chip import ADJ_LIST
from temp.analyze_program import labels

G = nx.Graph()
G.add_nodes_from(range(0,66))

#G.add_edges_from([(2, 3), (3, 4), (4, 1)])
# 添加单条边

def setOption( nt: Network):
    nt.set_options("""
           var options = {
         "edges": {
           "smooth": false
         },
         "physics": {
           "enabled": false,
           "minVelocity": 0.75
         }
       }
           """)

for i,row in enumerate(ADJ_LIST):
    for r in row:
        G.add_edge(i, r)

# 为每个节点设置编号标签
labels = {i: f" {i}" for i in G.nodes()}
nx.set_node_attributes(G, labels, 'label')

nt = Network('1000px', '1000px')
setOption(nt)
nt.from_nx(G)
nt.show('topology.html', notebook=False)
# pos = nx.spring_layout(G)
# nx.draw(G, pos, with_labels=True)
# plt.show()