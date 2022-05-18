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
