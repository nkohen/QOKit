import networkx as nx
import numpy as np

def erdos_reyni(num_verts: int, num_edges: int, seed: int | None = None) -> nx.Graph:
    G = nx.complete_graph(num_verts)
    if (seed is not None):
        np.random.seed(seed)
    edges = np.array(G.edges())
    indices = np.random.choice(a=len(edges), size = len(edges) - num_edges, replace = False)
    for index in indices:
        G.remove_edge(edges[index][0], edges[index][1])
    
    return G