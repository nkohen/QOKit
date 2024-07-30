#Comparison of Multinomial Distribution and Its Approximations
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multinomial, norm, dirichlet, poisson
from math import factorial
import itertools

def edgeworth_multinomial_pmf(counts, probs, n):
    counts = np.array(counts)
    probs = np.array(probs)
    k = len(probs)
    
    # Calculate the mean and variance for the normal approximation
    mu = n * probs
    sigma2 = n * probs * (1 - probs)
    
    # Calculate the standardized counts
    z = (counts - mu) / np.sqrt(sigma2)
    
    # Calculate the skewness and kurtosis
    skewness = (1 - 2 * probs) / np.sqrt(sigma2 / n)
    kurtosis = (1 - 6 * probs * (1 - probs)) / (sigma2 / n)
    
    # Calculate the correction terms for Edgeworth expansion
    phi_z = norm.pdf(z)
    Phi_z = norm.cdf(z)
    
    term1 = skewness / 6 * (z**3 - 3*z) * phi_z
    term2 = kurtosis / 24 * (z**4 - 6*z**2 + 3) * phi_z
    term3 = (skewness**2) / 72 * (z**6 - 15*z**4 + 45*z**2 - 15) * phi_z
    
    correction = Phi_z + term1 + term2 + term3
    
    # Calculate the multinomial coefficient
    multinomial_coeff = factorial(n) / np.prod([factorial(c) for c in counts])
    
    # Calculate the probability using the normal approximation
    normal_approx = np.prod(norm.pdf((counts - mu) / np.sqrt(sigma2)))
    
    # Apply the Edgeworth correction
    pmf = multinomial_coeff * normal_approx * np.prod(correction)
    
    return pmf

# Approximation using normal distribution
def normal_multinomial_pmf(counts, probs, n):
    counts = np.array(counts)
    probs = np.array(probs)
    mu = n * probs
    sigma2 = n * probs * (1 - probs)
    normal_approx = np.prod(norm.pdf((counts - mu) / np.sqrt(sigma2)))
    return normal_approx

# Approximation using Poisson distribution
def poisson_multinomial_pmf(counts, probs, n):
    counts = np.array(counts)
    lambdas = n * probs
    poisson_approx = np.prod([poisson.pmf(counts[i], lambdas[i]) for i in range(len(probs))])
    return poisson_approx

# MCMC Approximation Function
def mcmc_multinomial_pmf(counts, probs, n, num_samples=10000):
    counts = np.array(counts)
    k = len(probs)
    
    samples = np.zeros((num_samples, k))
    
    for i in range(num_samples):
        sample = np.random.multinomial(n, probs)
        samples[i] = sample
    
    sample_pmf = np.mean(np.all(samples == counts, axis=1))
    
    return sample_pmf

# Laplace Approximation Function
def laplace_multinomial_pmf(counts, probs, n):
    counts = np.array(counts)
    probs = np.array(probs)
    k = len(probs)
    
    # Mean and variance
    mu = n * probs
    sigma2 = n * probs * (1 - probs)
    
    # Multinomial coefficient
    multinomial_coeff = factorial(n) / np.prod([factorial(c) for c in counts])
    
    # Normal approximation
    normal_approx = np.prod(norm.pdf((counts - mu) / np.sqrt(sigma2)))
    
    # Laplace correction
    laplace_pmf = multinomial_coeff * normal_approx
    
    return laplace_pmf

'''
# Approximation using Dirichlet distribution
def dirichlet_multinomial_pmf(counts, alpha):
    counts = np.array(counts)
    alpha = np.array(alpha)
    dirichlet_approx = dirichlet.pdf(counts, alpha)
    return dirichlet_approx

alpha = np.random.rand(categories) + 1
dirichlet_pmfs = [dirichlet_multinomial_pmf(counts, alpha) for counts in counts_list]
'''
'''
# Approximation using Bayesian estimation
def bayesian_multinomial_pmf(counts, probs, n):
    counts = np.array(counts)
    alpha = np.ones_like(probs)
    posterior = dirichlet.pdf(probs, alpha + counts)
    bayesian_approx = multinomial.pmf(counts, n, probs) * posterior
    return bayesian_approx

bayesian_pmfs = [bayesian_multinomial_pmf(counts, probs, trials) for counts in counts_list]
'''

def measure_time_and_accuracy(trials, probs, counts_list):
    # Calculate exact PMF and measure time
    start_time = time.time()
    multinomial_pmfs = [multinomial.pmf(counts, n=trials, p=probs) for counts in counts_list]
    multinomial_time = time.time() - start_time
    
    # Calculate approximation PMFs and measure time
    start_time = time.time()
    edgeworth_pmfs = [edgeworth_multinomial_pmf(counts, probs, trials) for counts in counts_list]
    edgeworth_time = time.time() - start_time
    
    start_time = time.time()
    normal_pmfs = [normal_multinomial_pmf(counts, probs, trials) for counts in counts_list]
    normal_time = time.time() - start_time
    
    start_time = time.time()
    poisson_pmfs = [poisson_multinomial_pmf(counts, probs, trials) for counts in counts_list]
    poisson_time = time.time() - start_time
    
    start_time = time.time()
    mcmc_pmfs = [mcmc_multinomial_pmf(counts, probs, trials) for counts in counts_list]
    mcmc_time = time.time() - start_time
    
    start_time = time.time()
    laplace_pmfs = [laplace_multinomial_pmf(counts, probs, trials) for counts in counts_list]
    laplace_time = time.time() - start_time
    
    # Accuracy measures
    def calculate_accuracy(true_pmf, approx_pmf):
        return np.mean(np.abs(true_pmf - approx_pmf))
    
    def calculate_mse(true_pmf, approx_pmf):
        return np.mean((true_pmf - approx_pmf) ** 2)
    
    multinomial_pmf = np.array(multinomial_pmfs)
    edgeworth_accuracy = calculate_accuracy(multinomial_pmf, edgeworth_pmfs)
    normal_accuracy = calculate_accuracy(multinomial_pmf, normal_pmfs)
    poisson_accuracy = calculate_accuracy(multinomial_pmf, poisson_pmfs)
    mcmc_accuracy = calculate_accuracy(multinomial_pmf, mcmc_pmfs)
    laplace_accuracy = calculate_accuracy(multinomial_pmf, laplace_pmfs)
    
    edgeworth_mse = calculate_mse(multinomial_pmf, edgeworth_pmfs)
    normal_mse = calculate_mse(multinomial_pmf, normal_pmfs)
    poisson_mse = calculate_mse(multinomial_pmf, poisson_pmfs)
    mcmc_mse = calculate_mse(multinomial_pmf, mcmc_pmfs)
    laplace_mse = calculate_mse(multinomial_pmf, laplace_pmfs)
    
    return (multinomial_pmfs, multinomial_time, 
            edgeworth_pmfs, edgeworth_time,
            normal_pmfs, normal_time,
            poisson_pmfs, poisson_time,
            mcmc_pmfs, mcmc_time,
            laplace_pmfs, laplace_time,
            edgeworth_accuracy, normal_accuracy, poisson_accuracy, mcmc_accuracy, laplace_accuracy,
            edgeworth_mse, normal_mse, poisson_mse, mcmc_mse, laplace_mse)

# Parameters and user input
categories = 4
trials = int(input("Number of trials: "))

# Generate random probabilities that sum to 1
probs = np.random.rand(categories)
probs /= probs.sum()

# Generate all possible counts
counts_list = list(itertools.product(range(trials + 1), repeat=categories))
counts_list = [counts for counts in counts_list if sum(counts) == trials]
print(counts_list)

'''
# Calculate PMFs
multinomial_pmfs = [multinomial.pmf(counts, n=trials, p=probs) for counts in counts_list]
edgeworth_pmfs = [edgeworth_multinomial_pmf(counts, probs, trials) for counts in counts_list]
normal_pmfs = [normal_multinomial_pmf(counts, probs, trials) for counts in counts_list]
poisson_pmfs = [poisson_multinomial_pmf(counts, probs, trials) for counts in counts_list]
mcmc_pmfs = [mcmc_multinomial_pmf(counts, probs, trials) for counts in counts_list]
laplace_pmfs = [laplace_multinomial_pmf(counts, probs, trials) for counts in counts_list]
'''

# Measure time and accuracy
results = measure_time_and_accuracy(trials, probs, counts_list)
(multinomial_pmfs, multinomial_time, 
 edgeworth_pmfs, edgeworth_time,
 normal_pmfs, normal_time,
 poisson_pmfs, poisson_time,
 mcmc_pmfs, mcmc_time,
 laplace_pmfs, laplace_time,
 edgeworth_accuracy, normal_accuracy, poisson_accuracy, mcmc_accuracy, laplace_accuracy,
 edgeworth_mse, normal_mse, poisson_mse, mcmc_mse, laplace_mse) = results

# Plotting
#labels = [str(counts) for counts in counts_list]
#x = np.arange(len(counts_list))

plt.figure(figsize=(18, 12))

# Probability mass functions comparison
plt.subplot(2, 2, 1)
x = np.arange(len(counts_list))
plt.plot(x, multinomial_pmfs, marker='o', linestyle='-', label='True Multinomial')
plt.plot(x, edgeworth_pmfs, marker='x', linestyle='-', label='Edgeworth Approximation')
plt.plot(x, normal_pmfs, marker='v', linestyle='-', label='Normal Approximation')
plt.plot(x, poisson_pmfs, marker='s', linestyle='-', label='Poisson Approximation')
plt.plot(x, mcmc_pmfs, marker='D', linestyle='-', label='MCMC Approximation')
plt.plot(x, laplace_pmfs, marker='^', linestyle='-', label='Laplace Approximation')
plt.xlabel('Counts')
plt.ylabel('PMF')
plt.title('Probability Mass Functions Comparison')
#plt.xticks(x, [str(counts) for counts in counts_list], rotation=90)
plt.legend()

# Execution time comparison
plt.subplot(2, 2, 2)
methods = ['Multinomial', 'Edgeworth', 'Normal', 'Poisson', 'MCMC', 'Laplace']
times = [multinomial_time, edgeworth_time, normal_time, poisson_time, mcmc_time, laplace_time]
plt.bar(methods, times, color=['black', 'red', 'blue', 'orange', 'green', 'purple'])
plt.xlabel('Method')
plt.ylabel('Execution Time (s)')
plt.title('Execution Time Comparison')

# Accuracy comparison
plt.subplot(2, 2, 3)
methods = ['Edgeworth', 'Normal', 'Poisson', 'MCMC', 'Laplace']
accuracies = [edgeworth_accuracy, normal_accuracy, poisson_accuracy, mcmc_accuracy, laplace_accuracy]
plt.bar(methods, accuracies, color=['red', 'blue', 'orange', 'green', 'purple'])
plt.xlabel('Method')
plt.ylabel('Mean Absolute Error')
plt.title('Mean Absolute Error Comparison')

plt.subplot(2, 2, 4)
methods = ['Edgeworth', 'Normal', 'Poisson', 'MCMC', 'Laplace']
mses = [edgeworth_mse, normal_mse, poisson_mse, mcmc_mse, laplace_mse]
plt.bar(methods, mses, color=['red', 'blue', 'orange', 'green', 'purple'])
plt.xlabel('Method')
plt.ylabel('Mean Squared Error')
plt.title('Mean Squared Error Comparison')

#plt.suptitle(f'Comparison of Multinomial Distribution and Its Approximations (Number of trials: {trials})', fontsize=16)

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()