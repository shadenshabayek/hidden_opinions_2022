import networkx as nx
import numpy as np
import random as rd
import matplotlib.pyplot as plt

from utils import (save_figure)

def generate_random_opinion_vector(n):

    initial_opinions = [rd.uniform(-1,1) for _ in range(n)]
    return initial_opinions

def generate_scale_free_network(n, m):

    G = nx.barabasi_albert_graph(n = n, m = m, seed = None)
    df = nx.to_pandas_edgelist(G, nodelist=G.nodes)
    df[["source", "target"]]
    u = df.groupby(["source"])["target"].apply(list).reset_index(name='list_neighbors')
    links = dict(zip(u.source, u.list_neighbors))

    return G, links, n, m

def plot_network (n, m):

    G, links, n, m = generate_scale_free_network(n, m)

    degree_sequence = sorted((d for n, d in G.degree()), reverse=True)
    fig = plt.figure("Degree of a random graph", figsize=(8, 8))
    # Create a gridspec for adding subplots of different sizes
    axgrid = fig.add_gridspec(5, 4)
    ax0 = fig.add_subplot(axgrid[0:3, :])
    Gcc = G.subgraph(sorted(nx.connected_components(G), key=len, reverse=True)[0])
    pos = nx.spring_layout(Gcc, seed=None)
    nx.draw_networkx_nodes(Gcc, pos, ax=ax0, node_size=20)
    nx.draw_networkx_edges(Gcc, pos, ax=ax0, alpha=0.4)

    ax0.set_title("Scale-Free Network G")
    ax0.set_axis_off()

    ax1 = fig.add_subplot(axgrid[3:, :2])
    ax1.plot(degree_sequence, "b-", marker="o")
    ax1.set_title("Degree Distribution Plot ({} nodes)".format(n))
    ax1.set_ylabel("Degree")
    ax1.set_xlabel("Node Count")

    ax2 = fig.add_subplot(axgrid[3:, 2:])
    ax2.bar(*np.unique(degree_sequence, return_counts=True))
    ax2.set_title("Degree histogram")
    ax2.set_xlabel("Degree")
    ax2.set_ylabel("Node Count")
    fig.tight_layout()
    #plt.show()

if __name__ == '__main__':
    
    plot_network (n = 20, m = 2)
