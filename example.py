#%%
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import os

n = 10
m = 15
firms = np.arange(5)
A = np.arange(n) + 5
B = np.arange(n, m) + 5

firm_assignment = np.random.randint(0, 5, size = m)
employees = np.concatenate([A, B])
edge_list = np.stack((firm_assignment, employees), axis=1)

def get_color(x):
    if x in firms:
        return "black"
    elif x in A:
        return "tab:blue"
    elif x in B:
        return "tab:orange"

graph = nx.from_edgelist(edge_list)
colors = np.vectorize(get_color)(np.array(graph.nodes))
f, a = plt.subplots(figsize=(12, 4))
nx.draw(
    graph, 
    pos=nx.drawing.bipartite_layout(graph, employees, align="horizontal"),
    node_color=colors,
    ax=a
)

f.savefig(os.path.join("figs", "example.png"), facecolor="white", transparent=False)
# %%

# %%
