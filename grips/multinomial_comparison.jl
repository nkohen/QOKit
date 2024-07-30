using Distributions
using Plots

function edgeworth_multinomial_pmf(counts, probs, n)
    k = length(probs)
    
    # Calculate the mean and variance for the normal approximation
    mu = @. n * probs
    sigma2 = @. n * probs * (1 - probs)
    
    # Calculate the standardized counts
    z = @. (counts - mu) / sqrt(sigma2)
    
    # Calculate the skewness and kurtosis
    skewness = @. (1 - 2 * probs) / sqrt(sigma2 / n)
    kurtosis = @. (1 - 6 * probs * (1 - probs)) / (sigma2 / n)
    
    # Calculate the correction terms for Edgeworth expansion
    normal_distribution = Normal()
    phi_z = pdf(normal_distribution, z)
    Phi_z = cdf(normal_distribution, z)
    
    term1 = @. skewness / 6 * (z^3 - 3*z) * phi_z
    term2 = @. kurtosis / 24 * (z^4 - 6*z^2 + 3) * phi_z
    term3 = @. (skewness^2) / 72 * (z^6 - 15*z^4 + 45*z^2 - 15) * phi_z
    
    correction = @. Phi_z + term1 + term2 + term3
    
    # Calculate the multinomial coefficient
    # Can speedup here by not allocating memory
    multinomial_coeff = factorial(n) / prod(factorial.(counts))
    
    # Calculate the probability using the normal approximation
    #normal_approx = prod(norm.pdf((counts - mu) / np.sqrt(sigma2)))
    locations = @. (counts - mu) / sqrt(sigma2)
    normal_approx = prod(pdf(normal_distribution, locations))
    
    # Apply the Edgeworth correction
    pmf = multinomial_coeff * normal_approx * prod(correction)
    
    return pmf
end

# Approximation using normal distribution
function normal_multinomial_pmf(counts, probs, n)
    mu = @. n * probs
    sigma2 = @. n * probs * (1 - probs)
    locations = @. (counts - mu) / sqrt(sigma2)
    normal_distribution = Normal()
    normal_approx = prod(pdf(normal_distribution, locations))
    return normal_approx
end

# Approximation using Poisson distribution
function poisson_multinomial_pmf(counts, probs, n)
    lambdas = @. n * probs
    #poisson_approx = np.prod([poisson.pmf(counts[i], lambdas[i]) for i in range(len(probs))])
    poisson_approx = 1.0
    for (count, lambda) in zip(counts, lambdas)
        poisson_distribution = Poisson(lambda)
        poisson_approx *= pdf(poisson_distribution, count)
    end
    return poisson_approx
end

#=
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
=#

# Laplace Approximation Function
function laplace_multinomial_pmf(counts, probs, n)
    # Mean and variance
    mu = @. n * probs
    sigma2 = @. n * probs * (1 - probs)
    
    # Multinomial coefficient
    # Could speed up by not allocating memory
    multinomial_coeff = factorial(n) / prod(factorial.(counts))
    
    # Normal approximation
    normal_distribution = Normal()
    locations = @. (counts - mu) / sqrt(sigma2)
    normal_approx = prod(pdf(normal_distribution, locations))
    
    # Laplace correction
    laplace_pmf = multinomial_coeff * normal_approx
    
    return laplace_pmf
end

function multinomial_pmf(counts, probs, n)
    multinomial_distribution = Multinomial(n, probs)
    return pdf(multinomial_distribution, collect(counts)) # Multinomial requires vector, not tuple
end

function pmf_wrapper(counts, probs, n, method)
    if (method == "true")
        return multinomial_pmf(counts, probs, n)
    elseif (method == "edgeworth")
        return edgeworth_multinomial_pmf(counts, probs, n)
    elseif (method == "normal")
        return normal_multinomial_pmf(counts, probs, n)
    elseif (method == "poisson")
        return poisson_multinomial_pmf(counts, probs, n)
    elseif (method == "laplace")
        return laplace_multinomial_pmf(counts, probs, n)
    else
        throw("Invalid method name!")
    end

    return nothing
end

function measure_time_and_accuracy(trials, probs, counts_list)

    methods = ["edgeworth", "normal", "poisson", "laplace"]
    elapsed_time_dict = Dict()
    mse_dict = Dict()
    mean_abs_err_dict = Dict()
    pmfs_dict = Dict()

    time = @elapsed pmfs = [pmf_wrapper(counts, probs, trials, "true") for counts in counts_list]
    elapsed_time_dict["true"] = time
    pmfs_dict["true"] = pmfs
    for method in methods
        time = @elapsed pmfs = [pmf_wrapper(counts, probs, trials, method) for counts in counts_list]
        error = pmfs_dict["true"] - pmfs
        mse = mean(error .^ 2)
        mean_abs_err = mean(abs.(error))

        elapsed_time_dict[method] = time
        pmfs_dict[method] = pmfs
        mse_dict[method] = mse
        mean_abs_err_dict[method] = mean_abs_err
    end

    return_dict = Dict()
    return_dict["methods"] = methods
    return_dict["elapsed_time"] = elapsed_time_dict
    return_dict["mse"] = mse_dict
    return_dict["mean_abs_err"] = mean_abs_err_dict
    return_dict["pmfs"] = pmfs_dict

    return return_dict
end


N_categories = 4
N_trials = 20

probabilities = rand(N_categories)
probabilities /= sum(probabilities)

max_indices = Tuple(repeat([N_trials+1], N_categories))
counts_list = typeof(max_indices)[]
for I in CartesianIndices(max_indices)
    counts = Tuple(I) .- 1
    if (sum(counts) == N_trials)
        push!(counts_list, counts)
    end
end

results = measure_time_and_accuracy(N_trials, probabilities, counts_list)
