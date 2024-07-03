import typing
import qokit
import numpy as np
import scipy
import time
import networkx as nx
from qokit.fur.qaoa_simulator_base import QAOAFastSimulatorBase, TermsType
import QAOA_simulator as qs
import scipy_additional_optimizers
#必要なライブラリをインポートします。
#これには、QAOAシミュレーターやその他の補助ライブラリが含まれます。
'''
This is a script for testing the QAOA_simulator code. 
In particular, it runs max cut on a small graph to test the QAOA_run function. 
'''

#%% These two functions are to build the ising model of the graph here 

#generate a random graph
def random_graph(N, prob_connect = 0.7):
    A = np.random.choice([0, 1], (N, N), p=[1 - prob_connect, prob_connect])
    np.fill_diagonal(A, 0)  # No self-loops
    A = np.triu(A)  # Use only the upper triangle
    A += A.T  # Make the matrix symmetric
    return (A, nx.from_numpy_array(A))
#N: グラフのノード数。
#prob_connect: エッジが存在する確率。
#この関数は、ランダムな隣接行列 A を生成し、それに基づいて対称な無向グラフ G を作成します。

#build the ising model for a graph to use for QAOA maxcut cost
def max_cut_terms_for_graph(G):
    return list(map((lambda edge : (-0.5, edge)), G.edges)) + [((G.number_of_edges()/2.0), ())]
#この関数は、グラフ G に基づいてMaxCut問題のためのイジングモデルを生成します。
#エッジごとに (-0.5, edge) の形式のタームを作成し、全エッジの半分の数に対応する定数項を追加します。
    
#%% Now build the model and solve with QAOA

#first, set parameters
N = 5 #graph size
p = 3 #circuit depth for QAOA
optimizer_method = scipy_additional_optimizers.spsa_for_scipy#classical optimizer to use
# optimizer_method = 'COBYLA'#classical optimizer to use
init_gamma, init_beta = np.random.rand(2, p) #initial values
(_, G) = random_graph(N, 0.5)  #generate a random graph for G (the '_' we dont need, just networkx syntax)
ising_model = max_cut_terms_for_graph(G) #build the ising model for MaxCut on this graph
sim = qs.get_simulator(N, ising_model) #simulator for this ising model
#N, p, optimizer_method, init_gamma, init_beta などのQAOAのパラメータを設定します。
#ランダムグラフを生成し、そのグラフに対するMaxCut問題のイジングモデルを構築します。
#qs.get_simulator を使用して、イジングモデルに対応するQAOAシミュレーターを取得します。

#now solve with QAOA_run with these parameters
qaoa_result = qs.QAOA_run(
    ising_model,
    N,
    p,
    init_gamma,
    init_beta,
    optimizer_method=optimizer_method)
#qs.QAOA_run 関数を使用して、設定されたパラメータでQAOAを実行します。

#print the results 
print(f'With parameters N = {N}, p = {p}, method {optimizer_method}, we got:\n\n')
#print(f'State was {qaoa_result["state"]}\n') #suppressing this printing since it's noninformative
print(f'Gamma was                {qaoa_result["gamma"]}')
print(f'Beta was                 {qaoa_result["beta"]}')
print(f'Expetation was           {qaoa_result["expectation"]}')
print(f'Overlap was              {qaoa_result["overlap"]}')
print(f'Runtime was              {qaoa_result["runtime"]}')
print(f'Number of QAOA calls was {qaoa_result["num_QAOA_calls"]}\n')
print(f'Success?: {qaoa_result["classical_opt_success"]}\n')
print(f'Optimizer message: {qaoa_result["scipy_opt_message"]}\n')
#QAOAの実行結果を表示します。
#各パラメータ（gamma, beta）、期待値、オーバーラップ、実行時間、QAOA呼び出し回数、最適化の成功フラグ、最適化メッセージなどを出力します。
'''
まとめ
このスクリプトは、QAOAシミュレーターを使用して小さなランダムグラフに対するMaxCut問題を解くためのテストを行います。ランダムグラフを生成し、そのグラフのMaxCut問題に対応するイジングモデルを構築し、QAOAを実行して結果を表示します。これにより、QAOAシミュレーターの機能と性能を確認できます。
'''