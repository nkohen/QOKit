#generate a random graph
import numpy as np
import networkx as nx
def random_graph(N, prob_connect = 0.7):
    A = np.random.choice([0, 1], (N, N), p=[1 - prob_connect, prob_connect])
    np.fill_diagonal(A, 0)  # No self-loops
    A = np.triu(A)  # Use only the upper triangle
    A += A.T  # Make the matrix symmetric
    A = A.astype(float)
    return (A, nx.from_numpy_array(A))


def max_cut_terms_for_graph(G):
    '''
    Takes as input a networkx graph G, and 
    builds the ising model for G to use for QAOA maxcut cost.
    This can be passed to QAOA_run in our QAOA simulator

    Here is an EXAMPLE usage: 
    N = 5 #graph size
    p = 3 
    optimizer_method = scipy_additional_optimizers.spsa_for_scipy
    init_gamma, init_beta = np.random.rand(2, p) #initial values for gamma beta
    (_, G) = random_graph(N, 0.5)  #generate a random graph for G (the '_' we dont need, just networkx syntax)
    ising_model = max_cut_terms_for_graph(G) #HERE build the ising model for MaxCut on this graph
    sim = qs.get_simulator(N, ising_model) #simulator for this ising model

    #now solve with QAOA_run with these parameters, using the ising_model from this function here! 
    qaoa_result = qs.QAOA_run(
        ising_model,
        N,
        p,
        init_gamma,
        init_beta,
        optimizer_method=optimizer_method)
    '''
    return list(map((lambda edge : (-0.5, edge)), G.edges)) + [((G.number_of_edges()/2.0), ())]

#adj_matrix = nx.to_numpy_array(G) is how to get numpy array of a graph 

def evolve_graph(G, G_target, evolve_distance = 1):
    '''
    Takes as input a graph G and a target graph G_target. 
    Both should be networkx graphs, and may be weighted. 
    evolve_distance controls how much the graph G is changed to become more similar to G_target. 

    A total of evolve_distance of edge weights of G will be modified to make it more like G_target. 
    For example, in the unweighted case, if evolve_distance is 2, 
    up to 2 edges in G will be added or deleted to make G match G_target. 
    '''
    graphs_are_equal = False #flag on whether graphs are equal, but careful of float 
    N = len(G)
    adj_G = nx.to_numpy_array(G, dtype = float)
    adj_target = nx.to_numpy_array(G_target, dtype = float)
    exit_flag = False
    amount_evolved = 0.0
    tol = 0.0000001 #for floating point error ==, being careful 
    for i in range(N):
        for j in range(N):
            if np.abs(adj_G[i,j] - adj_target[i,j]) > tol: #this is checking != but controlling for floating point error
                diff = adj_target[i,j] - adj_G[i,j]
                abs_adjustment = max(np.abs(diff), evolve_distance-amount_evolved)
                adj_G[i,j] += np.sign(diff)*abs_adjustment
                amount_evolved += abs_adjustment
                if np.abs(amount_evolved -evolve_distance)< tol: #this is checking == but controlling for floating point error
                    exit_flag = True #exit if we've evolved enough
            if exit_flag: 
                break
        if exit_flag:
            break
    fully_evolved_flag = not(exit_flag) #if we exited, then the graphs weren't approximately equal in the beginning
    return nx.from_numpy_array(adj_G), fully_evolved_flag #returns the evolved G and flag 


