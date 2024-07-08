import typing
import qokit
import numpy as np
import scipy
import time
import networkx as nx
import os, sys
from qokit.fur.qaoa_simulator_base import QAOAFastSimulatorBase, TermsType
import matplotlib.pyplot as plt

#This is for importing .py's from possibly the parent directory
#But this way it doesn't matter which directory the .py's are in - either current or parent is fine!
# Get the parent directory
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
# Add the parent directory to sys.path
sys.path.insert(0, parent_dir)
# Import the modules
import QAOA_simulator as qs
import scipy_additional_optimizers
import graph_utils

save_data = False

# QAOAの設定
terms = [
    (1.0, [0, 1]),  # 第0キュービットと第1キュービットの間の相互作用
    (-1.0, [1, 2]), # 第1キュービットと第2キュービットの間の相互作用
    (0.5, [2, 3]),  # 第2キュービットと第3キュービットの間の相互作用
]

# Beta and Gamma ranges
beta_range = np.linspace(0, 2*np.pi, 50)
gamma_range = np.linspace(0, 2*np.pi, 50)

# 保存先ディレクトリを指定
if save_data == True:
    save_dir = "data_for_graphs_Predict(4)"
    os.makedirs(save_dir, exist_ok=True)  # ディレクトリが存在しない場合は作成する

# ヒートマップの作成
def create_heatmap(N: int, terms: list, p: int, beta_range: np.ndarray, gamma_range: np.ndarray):
    expectations = np.zeros((len(gamma_range), len(beta_range)))

    for i, gamma in enumerate(gamma_range):
        for j, beta in enumerate(beta_range):
            gamma_vec = np.full(p, gamma)
            beta_vec = np.full(p, beta)
            expectations[i, j] = qs.get_expectation(N, terms, gamma_vec, beta_vec)

    plt.figure(figsize=(8, 6))
    plt.imshow(expectations, aspect='auto', cmap='seismic', origin='lower', extent=[beta_range[0], beta_range[-1], gamma_range[0], gamma_range[-1]])
    plt.colorbar(label='Expectation')
    plt.xlabel('Beta')
    plt.ylabel('Gamma')
    plt.title(f'QAOA Expectation Heatmap (N={N}, p={p})')
    plt.show()
    if save_data == True:
        np.save(f"data_for_graphs_Predict(4)/data_for_QAOA_Expectation_Heatmap_N={N}_p={p}_kanon.npy", expectations)


'''
High-level notes of what I want this code to try: 

I want to "evolve" the graph slowly, using the optimal angle values from the previous graph 
as starting angles for the next one. 
By evolve the graph, I mean that for a target graph G, we start with an easy graph G0. 
We know the optimal angles for G_0, and evolving _0 to G is a sequence of graphs 
G_0, G_1, G_2, ..., G_(T-1), G_T = G, such that ||G_k - G_(k-1)|| is small, maybe 1. 
By the distance I imagine something like sum of edge weights of their symmetric difference.

Idea: Solve for G_k. Use the optimal gamma, beta as initial values for G_(k+1). If it evolves well, 
maybe we can find good angles for G if we start with good values for G_0. 

Kind of like an adiabatic evolution of the actual problem itself!! 

'''
N_values = range(5,6)  


for i, N in enumerate(N_values):
    create_heatmap(N, terms, p=1, beta_range=beta_range, gamma_range=gamma_range)



