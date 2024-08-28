import networkx as nx
import matplotlib.pyplot as plt
g = nx.DiGraph()
g.add_nodes_from([1, 2, 3])
g.add_edge(1, 2)
g.add_edge(2, 3)
g = nx.Graph(g)
# nx.draw(g)
g = nx.DiGraph(g)
nx.draw(g)
plt.show()