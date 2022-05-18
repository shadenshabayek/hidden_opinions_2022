import ast
from datetime import date
import lxml.etree as etree
import pandas as pd
import numpy as np
import os
import pickle
import time
import json_lines
import networkx as nx

from matplotlib import pyplot as plt
from utils import (save_figure)

def update_opinion_single_step(n, links, op, mu, tau):

    matrix_op = np.zeros((n,n))
    vector_op = []

    for player in links.keys():
        for i in links[player]:
            if abs(op[player] - op[i]) < tau:
                a = (op[i]-op[player])
                matrix_op[player, i ] = a

            elif abs(op[player] - op[i]) >= tau:
                a = (op[player]- op[i])
                matrix_op[player, i ] = a

        b = op[player] + mu*sum(matrix_op[player, : ])

        if b > 1 :
            b = 1
        elif b < -1:
            b=-1

        vector_op.append(b)

    return vector_op

def get_colors(val):

    if val <= -0.69 and val > -1.1:
        color = 'steelblue'
    elif val <= -0.4 and val > -0.69:
        color = 'skyblue'
    elif val <= -0.1 and val > -0.4:
        color = 'cyan'
    elif val <= 0.1 and val > -0.1:
        color = 'cornsilk'
    elif val <= 0.4 and val > 0.1:
        color = 'pink'
    elif val <= 0.69 and val > 0.4:
        color = 'lightcoral'
    elif val <= 1.1 and val > 0.69:
        color = 'brown'

    return color

def get_opinion_updated_expresser(n, op, links, N, mu, tau, name_plot):

    G = nx.from_dict_of_lists(links)

    attributes = {}

    for node in G.nodes:
        attributes[node] = op[node]

    nx.set_node_attributes(G, attributes, name="opinion")
    print(G.nodes[0]['opinion'])

    nx.write_gexf(G, './data/circle_{}.gexf'.format(name_plot))


    A = nx.to_numpy_array(G)

    matrix = np.zeros((n,N))
    matrix[:,0] = op

    plt.figure(figsize=(4, 2))
    ax = plt.subplot(111)

    for i in range(0,N-1):
        op_vector = update_opinion_single_step(n, links, matrix[:,i], mu, tau)
        matrix[:,i+1] = op_vector

    for j in range(0,n):

        ax.plot(np.arange(N),
                 matrix[j,:],
                 color = get_colors(matrix[j,0]),
                 linestyle = 'solid',
                 linewidth = 0.6,
                 )

    plt.ylim(-1.2, 1.2)
    plt.xlim(0,N-1)
    plt.xlabel('Time Periods')
    plt.ylabel('Opinions')

    save_figure('plot_{}.jpg'.format(name_plot))

def make_plots_circle(op, N, name_plot, tau, mu):

    n = 7
    links = {0: [1,6], 1: [0,2], 2: [1,3], 3: [2,4], 4: [3,5], 5: [4,6], 6: [5,0]}
    get_opinion_updated_expresser(n, op, links, N, mu, tau, name_plot)

def make_examples_circle(tau, mu):

    #monotonic ideo. opposed
    op_1 = [-0.5, 0.5, -0.6, 0.6, -0.7, 0.7, -0.8]
    N_1 = 10
    name_plot_1 = 'ideo_opposed'
    make_plots_circle(op_1, N_1, name_plot_1, tau, mu)

    #consensus
    op_2 = [0.3, -0.1, 0.1, 0.2, 0.3, 0.4, 0.5]
    N_2 = 30
    name_plot_2 = 'like_minded'
    make_plots_circle(op_2, N_2, name_plot_2, tau, mu)

    #moderate
    op_3 = [-0.9,-0.6,-0.3,0,0.3,0.6,0.9]
    N_3 = 30
    name_plot_3 = 'moderate'
    make_plots_circle(op_3, N_3, name_plot_3, tau, mu)

    #non-monotonic ideo. opposed
    op_4 = [-0.20, -0.10, 0.50, 0.40, 0.25, -0.70, -0.25]
    N_4 = 30
    name_plot_4 = 'ideo_opposed_non_monotonic'
    make_plots_circle(op_4, N_4, name_plot_4, tau, mu)

def main():

    make_examples_circle(tau = 0.5, mu = 0.2)

if __name__ == '__main__':

    main()
