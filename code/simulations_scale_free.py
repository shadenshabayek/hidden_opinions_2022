import networkx as nx
import numpy as np
import pandas as pd
import pickle
import random as rd
import matplotlib.pyplot as plt
import networkx.algorithms.community as nx_comm

from collections import Counter
from utils import (save_figure,
                   update_opinions_two_types,
                   save_list,
                   read_list)

def add_missing_links(df):

    list_source = df['source'].unique()
    list_target = df['target'].unique()

    list_missing_nodes = [x for x in list_target if x not in list_source]

    for i in range(0,len(list_missing_nodes)):
        idx = df.index[df['target'] == list_missing_nodes[i]]
        for j in range(0,len(idx)):
            source = df['target'].iloc[idx[j]]
            target = df['source'].iloc[idx[j]]
            df = df.append({'source': source ,
                            'target': target},
                             ignore_index=True)

    for i in list_source:
        idx = df.index[df['target'] == i]
        for j in range(0,len(idx)):
            if df['source'].iloc[idx[j]] < i:
                source = i
                target = df['source'].iloc[idx[j]]
                df = df.append({'source': source ,
                                'target': target},
                                 ignore_index=True)

    return df

def generate_scale_free_network(n, m):

    G = nx.barabasi_albert_graph(n = n, m = m, seed = None)
    df = nx.to_pandas_edgelist(G, nodelist=G.nodes)
    df[["source", "target"]]
    df = add_missing_links(df)
    u = df.groupby(["source"])["target"].apply(list).reset_index(name='list_neighbors')
    links = dict(zip(u.source, u.list_neighbors))

    return G, links, n, m

def get_local_centrality(links):

    centralities_dict = {}
    centralities_array = []

    for player in links.keys():
        d = len(links[player])
        sum_d = []

        for i in links[player]:
            if i in links.keys():
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

    return centralities_array, centralities_dict

def get_type(links, lim_centrality, centralities_dict):

    types_dict = {}
    types_array = []

    for player in centralities_dict.keys():
        if centralities_dict[player] >= lim_centrality:
            type = 'express'
        else:
            type = 'hide'
        types_dict[player] = type
        types_array.append(type)

    return types_array, types_dict

def generate_random_opinion_vector(n):

    initial_opinions = [rd.uniform(-1,1) for _ in range(n)]
    return initial_opinions

def plot_network (G, n, m):

    degree_sequence = sorted((d for n, d in G.degree()), reverse=True)
    fig = plt.figure("Degree of a random graph", figsize=(8, 8))

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
    save_figure('test_net.jpg')
    #plt.show()

def get_updated_opinions(n,m, N, tau, mu, opinions, quantile):

    G, links, n, m = generate_scale_free_network(n , m )
    centralities_array, centralities_dict = get_local_centrality(links)
    lim_centrality = np.quantile(centralities_array, quantile)
    print('mean centrality set as lim', lim_centrality)
    types_array, types_dict = get_type(links, lim_centrality, centralities_dict)

    matrix = np.zeros((n,N))
    matrix[:,0] = opinions

    for i in range(0,N-1):
        op_vector = update_opinions_two_types(n, links, opinions, mu, tau, types_dict)
        matrix[:,i+1] = op_vector

    return links, types_array, types_dict, matrix, N

def plot_hist_opinions(data):

    cm = plt.cm.get_cmap("bwr")
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    _, bins, patches = ax.hist(data,color="r",bins=20)
    bin_centers = 0.5*(bins[:-1]+bins[1:])
    col = bin_centers - min(bin_centers)
    col /= max(col)
    ax.tick_params(axis='x', rotation=45)

    for c, p in zip(col, patches):
        plt.setp(p, "facecolor", cm(c))


def add_attributes(n,m, N, tau, mu, name_plot, opinions, quantile):

    links, types_array, types_dict, matrix, N = get_updated_opinions(n,m, N, tau, mu, opinions, quantile)
    G = nx.from_dict_of_lists(links)
    #plot_network (G, n, m)
    op = matrix[:,N-1]

    plot_hist_opinions(op)
    save_figure('hist_opinions_{}'.format(name_plot))

    attributes_1 = {}

    for node in G.nodes:
        attributes_1[node] = op[node]

    attributes_2 = {}

    for node in G.nodes:
        attributes_2[node] = types_array[node]

    nx.set_node_attributes(G, attributes_1, name="opinion")
    nx.set_node_attributes(G, attributes_2, name="type")

    set_expressers = []
    for node in G.nodes:
        if attributes_2[node] == 'express':
            set_expressers.append(node)

    H = G.subgraph(set_expressers)
    n_H = nx.number_connected_components(H)
    print('number connected components of expressers', n_H)
    assortativity_G = nx.degree_assortativity_coefficient(G)
    print('degree assortativity of G', assortativity_G)
    #modularity = nx_comm.modularity(G, list(G.nodes))
    #print(modularity)

    print('average op', np.mean(op))
    print('max op', np.max(op))
    print('var op', np.var(op))
    print(Counter(types_dict.values()))
    nx.write_gexf(G, './data/BA_{}.gexf'.format(name_plot))
    nx.write_gexf(H, './data/BA_{}_subgraph_expressers.gexf'.format(name_plot))

    return G

def run_simulations(opinions, n, name_plot, quantile):

    m = 6
    N = 300
    tau = 0.45
    mu = 0.4
    add_attributes(n, m, N, tau, mu, name_plot, opinions, quantile)

def main(new_opinion_vector):

    n = 600
    if new_opinion_vector == 1:
        opinions = generate_random_opinion_vector(n)
        save_list(opinions, 'opinions_5.txt')
    else:
        opinions = read_list('opinions_5.txt')


    quantile = 0.8
    name_plot = 'run_8_nodes_08_quantile_{}'.format(n)
    run_simulations(opinions, n, name_plot, quantile)

if __name__ == '__main__':

    main(new_opinion_vector = 0)
