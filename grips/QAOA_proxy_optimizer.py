from QAOA_simulator import get_expectation, get_simulator
import numpy as np
import scipy
import qokit.maxcut as mc
from QAOA_proxy import QAOA_proxy_run
import Graph_util
from sklearn import linear_model

def QAOA_proxy_optimize(
    num_constraints: int,
    num_qubits: int,
    p: int,
    init_gamma: np.ndarray,
    init_beta: np.ndarray,
    num_graphs: int = 50,
    optimizer_method: str = "COBYLA",
    optimizer_options: dict | None = None,
    init_tweaks: list[tuple[float, float, float, float]] = [0,0,1,1],
) -> tuple[float, float, float, float]:
    ising_models = []
    sims = []
    for _ in range(num_graphs):
        G = Graph_util.erdos_renyi(num_qubits, num_constraints)
        terms = mc.get_maxcut_terms(G)
        ising_models.append(list(terms))
        sims.append(get_simulator(num_qubits, terms))

    def iof(*args) -> float:
        h, hc, l, r = args[0][0], args[0][1], args[0][2], args[0][3]
        result = QAOA_proxy_run(num_constraints, num_qubits, p, init_gamma, init_beta, optimizer_method, optimizer_options, None, h, hc, l, r)
        expectation_sum = 0
        for i in range(num_graphs):
            ising_model = ising_models[i]
            expectation_sum += get_expectation(num_qubits, ising_model, result["gamma"], result["beta"], sims[i])
        return -expectation_sum

    best_result = None
    best_objective = 0
    for init_h_tweak, init_hc_tweak, init_l_tweak, init_r_tweak in init_tweaks:
        init_freq = np.hstack([init_h_tweak, init_hc_tweak, init_l_tweak, init_r_tweak])
        result = scipy.optimize.minimize(iof, init_freq, args=(), method=optimizer_method, options = optimizer_options, bounds = [(None, 1 << (num_qubits - 4)), (None, None), (0, None), (0, None)])
        result_objective_value = iof(result.x)
        if (result_objective_value < best_objective):
            best_result = result
            best_objective = result_objective_value


    return best_result.x[0], best_result.x[1], best_result.x[2], best_result.x[3]

def find_optimal_parameters(min_N: int, max_N: int, p: int) -> dict:
    init_gamma, init_beta = np.full((2, p), 0.1)
    prev_tweaks = [] # Now only used until we have at least 3 data points
    results = dict()

    for N in range(min_N, max_N + 1):
        print(f"Starting work on N={N}")
        max_edges = N*(N-1)//2

        # For the first run of each new N, set prev_tweaks to be the
        # best parameters in the previous N for nearby number of edges.
        if (N == min_N):
            prev_tweaks = (0, 0, 1, 1)
        elif (N-1, max_edges // 3) in results:
            prev_tweaks = results[N-1, max_edges // 3]
        elif (N-1, (max_edges // 3) - 1) in results:
            prev_tweaks = results[N-1, (max_edges // 3) - 1]
        elif (N-1, (max_edges // 3) - 2) in results:
            prev_tweaks = results[N-1, (max_edges // 3) - 2]
        
        # To make things faster, we only sample every third value of num_edges
        for num_edges in range(max_edges // 3, max_edges - 3, 3):
            inits = []
            if len(results) < 3:
                inits = [prev_tweaks]
            else:
                regr_so_far, _ = linear_regression_for_parameters(results)
                init_guess = regr_so_far.predict(np.array([[N, num_edges]]))
                inits = [(init_guess[0][0], init_guess[0][1], init_guess[0][2], init_guess[0][3])]

            h, hc, l, r = QAOA_proxy_optimize(num_edges, N, p, init_gamma, init_beta, init_tweaks=inits)
            results[N, num_edges] = (h, hc, l, r)
            # Set the initial parameters for the next run
            prev_tweaks = (h, hc, l, r)
    
    return results

def linear_regression_for_parameters(optimal_parameters: dict):
    x = np.array(list(optimal_parameters.keys()))
    y = np.array(list(optimal_parameters.values()))
    regr = linear_model.LinearRegression().fit(x, y)
    return regr, regr.score(x, y)