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

def save_figure(figure_name):

    figure_path = os.path.join('.', 'figures', figure_name)
    plt.savefig(figure_path, bbox_inches='tight', dpi = 300)

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

def update_opinions_two_types(n, links, op, mu, tau, types):

    matrix_op = np.zeros((n,n))
    vector_op = []

    for player in links.keys():
        if types[player] == 'express':
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

        elif types[player] == 'hide':
            for i in links[player]:
                matrix_op[player, i ] = op[i] / (len(links[player])+1)

            b = op[player]/(len(links[player])+1) + sum(matrix_op[player, : ])
            vector_op.append(b)

    return vector_op

def save_list(list, file_name):

    file_path = os.path.join('.', 'data', file_name)
    #file_name has to be .txt

    with open(file_path, "wb") as fp:
        pickle.dump(list, fp)

def read_list(file_name):

    file_path = os.path.join('.', 'data', file_name)

    with open(file_path, "rb") as fp:   # Unpickling
        b = pickle.load(fp)

    return b
