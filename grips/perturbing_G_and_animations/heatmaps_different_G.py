#%% imports and setup 
import typing
import qokit
import numpy as np
import scipy
import time
import networkx as nx
import os, sys
from qokit.fur.qaoa_simulator_base import QAOAFastSimulatorBase, TermsType
import matplotlib.pyplot as plt
import matplotlib.animation as animation

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
import seaborn as sns

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
    plt.show(block=False)
    if save_data == True:
        np.save(f"data_for_graphs_Predict(4)/data_for_QAOA_Expectation_Heatmap_N={N}_p={p}_kanon.npy", expectations)

# def create_heatmap2(N: int, terms: list, p: int, beta_range: np.ndarray, gamma_range: np.ndarray):
#     expectations = np.zeros((len(gamma_range), len(beta_range)))

#     for i, gamma in enumerate(gamma_range):
#         for j, beta in enumerate(beta_range):
#             gamma_vec = np.full(p, gamma)
#             beta_vec = np.full(p, beta)
#             expectations[i, j] = qs.get_expectation(N, terms, gamma_vec, beta_vec)

#     # Return the expectations array instead of plotting
#     return expectations

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

#%% 

save_data = False

# # QAOAの設定
# terms = [
#     (1.0, [0, 1]),  # 第0キュービットと第1キュービットの間の相互作用
#     (-1.0, [1, 2]), # 第1キュービットと第2キュービットの間の相互作用
#     (0.5, [2, 3]),  # 第2キュービットと第3キュービットの間の相互作用
# ]


# Beta and Gamma ranges, image size
img_size = 40
beta_range = np.linspace(0, 2*np.pi, img_size)
gamma_range = np.linspace(0, 2*np.pi, img_size)

# # 保存先ディレクトリを指定
# if save_data == True:
#     save_dir = "data_for_graphs_Predict(4)"
#     os.makedirs(save_dir, exist_ok=True)  # ディレクトリが存在しない場合は作成する

#random target graph
N = 9
_, G_target = graph_utils.random_graph(N, 0.5)

#evolve distance and initial graph 
evolve_distance = 1

G_evolving = nx.from_numpy_array(np.zeros(nx.to_numpy_array(G_target).shape, dtype = float))
evolved_graphs_list = [G_evolving]
fully_evolved_flag = False

print(nx.to_numpy_array(G_evolving), '\n')
print(nx.to_numpy_array(G_target), '\n')


while fully_evolved_flag == False: 
    #create_heatmap(N, ising_model, p=1, beta_range=beta_range, gamma_range=gamma_range)
    G_evolving, fully_evolved_flag = graph_utils.evolve_graph(G_evolving, G_target, evolve_distance = evolve_distance)
    evolved_graphs_list.append(G_evolving)

num_graphs = len(evolved_graphs_list)
framecounter =0
expectations = np.zeros((num_graphs, len(gamma_range), len(beta_range)), dtype = float)
for k in range(num_graphs):
    G = evolved_graphs_list[k]
    ising_model = graph_utils.max_cut_terms_for_graph(G)
    terms = graph_utils.max_cut_terms_for_graph(G)
    for i, gamma in enumerate(gamma_range):
        for j, beta in enumerate(beta_range):
            gamma_vec = np.full(1, gamma) #really this 1 should be p, this is temp
            beta_vec = np.full(1, beta)
            expectations[k, i, j] = qs.get_expectation(N, terms, gamma_vec, beta_vec)


print('Evolving graphs:')
for k in range(num_graphs):
    print(nx.to_numpy_array(evolved_graphs_list[k]), '\n')
    print(expectations[k])

print(expectations.shape)

# Define the expectations array with shape (num_frames, nx, ny)
num_frames = num_graphs
# nx, ny = 10, 10
# expectations = np.random.rand(num_frames, nx, ny)  # Replace this with your actual data

# Set up the figure, axis, and plot element
fig, ax = plt.subplots()
cax = ax.matshow(expectations[1], cmap='seismic')

def update(frame):
    cax.set_data(expectations[frame+1])
    cax.set_clim(vmin=np.min(expectations[frame+1]), vmax=np.max(expectations[frame+1]))  # Reset color scale
    return cax,

# Create the animation
ani = animation.FuncAnimation(fig, update, frames=num_frames-1, interval=1000, blit=True)

# Save the animation as a GIF
ani.save('heatmap_animation_N='+str(N)+'.gif', writer='pillow', fps=10)


plt.show()
# %%
