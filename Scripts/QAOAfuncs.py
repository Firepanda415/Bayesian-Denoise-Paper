# -*- coding: utf-8 -*-
"""
Created on Thu Jun 11 15:19:08 2020

@author: Muqing Zheng
"""
# Numerical/Stats pack
import csv
import pandas as pd
import numpy as np
import scipy.stats as ss
import scipy.linalg as la

import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from qiskit.tools.visualization import *

import networkx as nx

pd.set_option('precision', 6)

# Check endianness, make sure little endian
import sys


def optimalProb(counts, sols):
    shots = 0
    for key in counts:
        shots += counts[key]

    psum = 0
    for x in sols:
        psum += counts[str(x)] / shots
    return psum


# Compute the value of the cost function
def cost_function_C(x, G):
    x = x[::-1]
    E = G.edges()
    if (len(x) != len(G.nodes())):
        return np.nan

    C = 0
    for index in E:
        e1 = index[0]
        e2 = index[1]

        w = G[e1][e2]['weight']
        C = C + w * x[e1] * (1 - x[e2]) + w * x[e2] * (1 - x[e1])

    return C


def QAOAopt(G, counts, shots, ifprint=True):
    avr_C = 0
    max_C = [0, 0]
    hist = {}

    for k in range(len(G.edges()) + 1):
        hist[str(k)] = hist.get(str(k), 0)

    for sample in list(counts.keys()):

        # use sampled bit string x to compute C(x)
        x = [int(num) for num in list(sample)]
        x.append(0)
        tmp_eng = cost_function_C(x, G)

        # compute the expectation value and energy distribution
        avr_C = avr_C + counts[sample] * tmp_eng
        hist[str(round(tmp_eng))] = hist.get(str(round(tmp_eng)),
                                             0) + counts[sample]

        # save best bit string
        if (max_C[1] < tmp_eng):
            max_C[0] = sample
            max_C[1] = tmp_eng

    M1_sampled = avr_C / shots

    if ifprint:
        print('M_sampled = %.04f, solution is %s with C(x*) = %d  \n' %
              (M1_sampled, max_C[0], max_C[1]))
        # plot_histogram(hist,bar_labels = False)
        # plt.show()

    return M1_sampled, max_C, hist
