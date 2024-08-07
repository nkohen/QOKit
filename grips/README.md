# G-RIPS 2024 Mitsubishi B Project

This directory contains work involved as part of [G-RIPS 2024 Sendai](https://www.ipam.ucla.edu/programs/student-research-programs/graduate-level-research-in-industrial-projects-for-students-g-rips-sendai-2024).

The primary focus of our work was to investigate, modify and improve the [Parameter-setting heuristic for the quantum alternating operator ansatz](https://journals.aps.org/prresearch/pdf/10.1103/PhysRevResearch.6.023171).

## Setting up Julia backend
The QAOA proxies have an optional Julia backend implementation, which typically results in an over 100x speedup over the Python implementation.

The Julia backend can be set up by running the python script [setup_juliacall.py](https://github.com/nkohen/QOKit/blob/grips/grips/setup_juliacall.py).
Or, one can install JuliaCall manually by running `pip install julicall` from the terminal, and then installing the required Julia packages by opening a Python session and running:
```
jl.seval("""
import Pkg
Pkg.add("Distributions")
Pkg.add("BenchmarkTools")
Pkg.add("TimerOutputs")
Pkg.add("PythonCall")
""")
```

## Important Files

* [QAOA_Simulator.py](https://github.com/nkohen/QOKit/blob/grips/grips/QAOA_simulator.py)
  * Contains utilities for simulating QAOA on arbitrary Ising model Hamiltonians.
* [plot_utils.py](https://github.com/nkohen/QOKit/blob/grips/grips/plot_utils.py)
  * Contains utilities which delegate to matplotlib but which all have the same interface.
* [QAOA_paper_proxy.py](https://github.com/nkohen/QOKit/blob/grips/grips/QAOA_paper_proxy.py)
  * Implements the original [parameter-setting heuristic](https://journals.aps.org/prresearch/pdf/10.1103/PhysRevResearch.6.023171) from the paper.
  * To enable the Julia backend, make sure `USE_JULIA=True` is set near the top of this file. If `USE_JULIA=False` is set, then the python implementation will be used.
* [QAOA_proxy.py](https://github.com/nkohen/QOKit/blob/grips/grips/QAOA_proxy.py)
  * Implements our parameter-setting heuristic, which is much faster.
  * To enable the Julia backend, make sure `USE_JULIA=True` is set near the top of this file. If `USE_JULIA=False` is set, then the python implementation will be used.
* [setup_juliacall.py](https://github.com/nkohen/QOKit/blob/grips/grips/setup_juliacall.py)
  * Python script for installing the software necessary for the Julia backend for the QAOA proxies. 
* [QAOA_paper_proxy.jl](https://github.com/nkohen/QOKit/blob/grips/grips/QAOA_paper_proxy.jl)
  * Julia implementation of the original [parameter-setting heuristic](https://journals.aps.org/prresearch/pdf/10.1103/PhysRevResearch.6.023171) from the paper.
* [QAOA_proxy.jl](https://github.com/nkohen/QOKit/blob/grips/grips/QAOA_proxy.jl)
  * Julia implementation of our parameter-setting heuristic.

## Example Files

* [QAOA_plots.ipynb](https://github.com/nkohen/QOKit/blob/grips/grips/QAOA_plots.ipynb)
  * Contains example usage of QAOA_Simulator.
* [QAOA_proxy_plots.ipynb](https://github.com/nkohen/QOKit/blob/grips/grips/QAOA_proxy_plots.ipynb)
  * Contains example usage of QAOA_paper_proxy and QAOA_proxy.
* [real_distributions.ipynb](https://github.com/nkohen/QOKit/blob/grips/grips/real_distributions.ipynb)
  * Contains example code for sampling real MaxCut distribution data.

