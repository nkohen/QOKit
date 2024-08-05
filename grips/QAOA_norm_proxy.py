import numpy as np
import math
import typing
import time
import scipy
from scipy.stats import multivariate_normal

"""
This file implements the QAOA proxy algorithm for MaxCut from:
https://journals.aps.org/prresearch/pdf/10.1103/PhysRevResearch.6.023171
"""


###
### The following functions are defined, but QAOA_paper_proxy now uses the julia
### implementations instead of the python ones. So most of the functions in this
### file will never be called. But they can be called using QAOA_paper_proxy_python
###

# P(c') from paper
def prob_cost_norm(cost: int, prob_cost_mean: float, prob_cost_cov: float) -> float:
    return multivariate_normal.pdf(cost, mean = prob_cost_mean, cov = prob_cost_cov)


# N(c') from paper
def number_with_cost_norm_proxy(cost: int, num_qubits: int) -> float:
    scale = 1 << num_qubits
    return prob_cost_norm(cost, num_qubits/2, num_qubits/4) * scale


# N(c'; d, c) from paper
def number_of_costs_at_distance_norm_proxy(cost_1: int, cost_2: int, distance: int, num_qubits: int, N_cost_mean: float, N_cov_1: float, N_cov_2: float) -> float:
    N_distance_mean = num_qubits / 2

    P = np.matrix([[cost_1 - N_cost_mean, N_distance_mean], [-N_distance_mean, cost_1 - N_cost_mean]])
    P_inv = scipy.linalg.inv(P)
    cov_mat = P@np.matrix([[N_cov_1, 0], [0, N_cov_2]])@P_inv
    cov_mat[0, 1] = cov_mat[1, 0] # cov_mat must be symmetric and is prone to floating point error
    return multivariate_normal([N_cost_mean, N_distance_mean], cov_mat).pdf([cost_2, distance])*(1 << num_qubits), cov_mat

# Computes the sum inside the for loop of Algorithm 1 in paper
def compute_amplitude_sum_norm(prev_amplitudes: np.ndarray, gamma: float, beta: float, cost_1: int, num_constraints: int, num_qubits: int, c_mean: float, cov_1: float, cov_2: float) -> complex:
    sum = 0
    for cost_2 in range(num_constraints + 1):
        for distance in range(num_qubits + 1):
            # Should I np-ify all of the stuff here?
            beta_factor = (np.cos(beta) ** (num_qubits - distance)) * ((-1j * np.sin(beta)) ** distance)
            gamma_factor = np.exp(-1j * gamma * cost_2)
            num_costs_at_distance = number_of_costs_at_distance_norm_proxy(cost_1, cost_2, distance, num_qubits, c_mean, cov_1, cov_2)
            sum += beta_factor * gamma_factor * prev_amplitudes[cost_2] * num_costs_at_distance
    return sum


## Uncomment this function and comment out the next function if you want to use the python implementation instead of the julia one.
# TODO: What if instead of optimizing expectation proxy we instead optimize high cost amplitudes (using e.g. exponential weighting)
# Algorithm 1 from paper
# num_constraints = number of edges, and num_qubits = number of vertices
def QAOA_norm_proxy(p: int, gamma: np.ndarray, beta: np.ndarray, num_constraints: int, num_qubits: int, c_mean: float, cov_1: float, cov_2: float, terms_to_drop_in_expectation: int = 0):
    num_costs = num_constraints + 1
    amplitude_proxies = np.zeros([p + 1, num_costs], dtype=complex)
    init_amplitude = np.sqrt(1 / (1 << num_qubits))
    for i in range(num_costs):
        amplitude_proxies[0][i] = init_amplitude

    for current_depth in range(1, p + 1):
        for cost_1 in range(num_costs):
            amplitude_proxies[current_depth][cost_1] = compute_amplitude_sum_norm(
                amplitude_proxies[current_depth - 1], gamma[current_depth - 1], beta[current_depth - 1], cost_1, num_constraints, num_qubits, c_mean, cov_1, cov_2
            )

    expected_proxy = 0
    for cost in range(terms_to_drop_in_expectation, num_costs):
        expected_proxy += number_with_cost_norm_proxy(cost, num_qubits) * (abs(amplitude_proxies[p][cost]) ** 2) * cost

    return amplitude_proxies, expected_proxy



def inverse_norm_proxy_objective_function(num_constraints: int, num_qubits: int, p: int, c_mean: float, cov_1: float, cov_2: float, expectations: list[np.ndarray] | None) -> typing.Callable:
    def inverse_objective(*args) -> float:
        gamma, beta = args[0][:p], args[0][p:]
        _, expectation = QAOA_norm_proxy(p, gamma, beta, num_constraints, num_qubits, c_mean, cov_1, cov_2)
        current_time = time.time()

        if expectations is not None:
            expectations.append((current_time, expectation))

        return -expectation

    return inverse_objective


def QAOA_norm_proxy_run(
    num_constraints: int,
    num_qubits: int,
    p: int,
    init_gamma: np.ndarray,
    init_beta: np.ndarray,
    c_mean: float,
    cov_1: float,
    cov_2: float,
    optimizer_method: str = "COBYLA",
    optimizer_options: dict | None = None,
    expectations: list[np.ndarray] | None = None,
) -> dict:
    init_freq = np.hstack([init_gamma, init_beta])

    start_time = time.time()
    result = scipy.optimize.minimize(
        inverse_norm_proxy_objective_function(num_constraints, num_qubits, p, c_mean, cov_1, cov_2, expectations),
        init_freq,
        args=(),
        method=optimizer_method,
        options=optimizer_options,
    )
    # the above returns a scipy optimization result object that has multiple attributes
    # result.x gives the optimal solutionsol.success #bool whether algorithm succeeded
    # result.message #message of why algorithms terminated
    # result.nfev is number of iterations used (here, number of QAOA calls)
    end_time = time.time()

    def make_time_relative(input: tuple[float, float]) -> tuple[float, float]:
        time, x = input
        return (time - start_time, x)

    if expectations is not None:
        expectations = list(map(make_time_relative, expectations))

    gamma, beta = result.x[:p], result.x[p:]
    _, expectation = QAOA_norm_proxy(p, gamma, beta, num_constraints, num_qubits, c_mean, cov_1, cov_2)

    return {
        "gamma": gamma,
        "beta": beta,
        "expectation": expectation,
        "runtime": end_time - start_time,  # measured in seconds
        "num_QAOA_calls": result.nfev,  # Calls to the proxy of course
        "classical_opt_success": result.success,
        "scipy_opt_message": result.message,
    }