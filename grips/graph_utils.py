#generate a random graph
def random_graph(N, prob_connect = 0.7):
    A = np.random.choice([0, 1], (N, N), p=[1 - prob_connect, prob_connect])
    np.fill_diagonal(A, 0)  # No self-loops
    A = np.triu(A)  # Use only the upper triangle
    A += A.T  # Make the matrix symmetric
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
