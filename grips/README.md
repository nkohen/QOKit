# G-RIPS 2024 Mitsubishi B Project

This directory contains work involved as part of [G-RIPS 2024 Sendai](https://www.ipam.ucla.edu/programs/student-research-programs/graduate-level-research-in-industrial-projects-for-students-g-rips-sendai-2024).

The primary focus of our work was to investigate, modify and improve the [Parameter-setting heuristic for the quantum alternating operator ansatz](https://journals.aps.org/prresearch/pdf/10.1103/PhysRevResearch.6.023171).

## Setting up Julia backend

TODO

## Important Files

* [QAOA_Simulator.py](https://github.com/nkohen/QOKit/blob/grips/grips/QAOA_simulator.py)
  * Contains utilities for simulating QAOA on arbitrary Ising model Hamiltonians.
* [plot_utils.py](https://github.com/nkohen/QOKit/blob/grips/grips/plot_utils.py)
  * Contains utilities which delegate to matplotlib but which all have the same interface.
* [QAOA_paper_proxy.py](https://github.com/nkohen/QOKit/blob/grips/grips/QAOA_paper_proxy.py)
  * Implements the original [parameter-setting heuristic](https://journals.aps.org/prresearch/pdf/10.1103/PhysRevResearch.6.023171) from the paper.
* [QAOA_proxy.py](https://github.com/nkohen/QOKit/blob/grips/grips/QAOA_proxy.py)
  * Implements our parameter-setting heuristic, which is much faster.

## Example Files

* [QAOA_plots.ipynb](https://github.com/nkohen/QOKit/blob/grips/grips/QAOA_plots.ipynb)
  * Contains example usage of QAOA_Simulator.
* [QAOA_proxy_plots.ipynb](https://github.com/nkohen/QOKit/blob/grips/grips/QAOA_proxy_plots.ipynb)
  * Contains example usage of QAOA_paper_proxy and QAOA_proxy.
* [real_distributions.ipynb](https://github.com/nkohen/QOKit/blob/grips/grips/real_distributions.ipynb)
  * Contains example code for sampling real MaxCut distribution data.

