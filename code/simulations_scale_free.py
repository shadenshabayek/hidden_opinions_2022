import networkx as nx
import numpy as np
import random as rd
import matplotlib.pyplot as plt

from utils import (save_figure)

def get_local_centrality(links):
    #links = {0: [1,6], 1: [0,2], 2: [1,3], 3: [2,4], 4: [3,5], 5: [4,6], 6: [5,0]}

    centralities_dict = {}
    centralities_array = []

    for player in links.keys():
        d = len(links[player])
        sum_d = []

        for i in links[player]:
            d_neighbor = len(links[i])
            sum_d.append(d_neighbor)

        if sum(sum_d) > 0:
            centrality = d / sum(sum_d)
            centralities_array.append(centrality)
            centralities_dict[player] = centrality
        else:
            centrality = 0
            centralities_array.append(centrality)
            centralities_dict[player] = centrality

    print(centralities_array)
    return centralities_array, centralities_dict

def get_type(links):

    types_dict = {}
    types_array = []

    return types_array, types_dict

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

#def get_local_centrality():
#def get_type():
#def generate_graph_with_attribute():
#def level_polarization():

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

    #plot_network (n = 20, m = 2)
    get_local_centrality()
