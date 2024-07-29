from qokit.fur.qaoa_simulator_base import TermsType
from QAOA_simulator import get_expectation, get_simulator
import numpy as np
import scipy
import qokit.maxcut as mc
from QAOA_proxy import QAOA_proxy_run
import Graph_util

def QAOA_proxy_optimize(
    num_constraints: int,
    num_qubits: int,
    p: int,
    init_gamma: np.ndarray,
    init_beta: np.ndarray,
    num_graphs: int = 50,
    optimizer_method: str = "COBYLA",
    optimizer_options: dict | None = None,
    init_tweaks: list[list[float]] = [0,0,1,1],
) -> tuple[float, float, float, float]:
    ising_models = []
    sims = []
    for _ in range(num_graphs):
        G = Graph_util.erdos_reyni(num_qubits, num_constraints)
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
