# Machine Learning for Dynamic Incentive Problems

This Python-based code repository supplements the work of [Philipp Renner](https://www.lancaster.ac.uk/lums/people/philipp-renner) and [Simon Scheidegger](https://sites.google.com/site/simonscheidegger), titled _[Machine Learning for Dynamic Incentive Problems](#citation)_ (Renner and Scheidegger; 2018), which introduces a highly scalable computational technique to solve dynamic incentive problems (with potentially a substantial amount of heterogeneity).

However, the scope of the method is much broader: as it is a generic framework to compute global solutions to dynamic (stochastic) models with many state variables, it is applicable to almost any dynamic model. The available solution algorithms are based on "value function iteration", as well as "time iteration".


* This repository aims to make our method easily accessible to the computational economics and finance community. 
* The computational framework located [here](dptorch) is extensively documented, leverages [GPyTorch](https://docs.gpytorch.ai/en/v1.5.1/index.html), and combines Gaussian Process regression with performance-boosting options such as Bayesian active learning, the active subspace method, Deep Gaussian processes, and MPI-based parallelism.
* Replication codes for the dynamic incentive problems are provided.
* Furthermore, to demonstrated the broad applicablility of the method, several additional examples are provided. Specifically, a [stochastic optimal growth model](https://www.sciencedirect.com/science/article/pii/S1877750318306161?via%3Dihub) ([solved with value function iteration](dptorch/StochasticOptimalGrowthModel)), and an [international real business cycle model](https://onlinelibrary.wiley.com/doi/abs/10.3982/ECTA12216) ([solved with time iteration](dptorch/IRBC)) are provided.
* In addition, simple code examples to introduce [Gaussian Processe regression](analytical_examples/Gaussian_process_regression_in_1d.ipynb) and [Bayesian active learning](analytical_examples/BAL_with_GPs.ipynb) to new users in a standalone fashion are provided [here](analytical_examples).


### Authors
* [Philipp Renner](https://www.lancaster.ac.uk/lums/people/philipp-renner) (University of Lancaster, Department of Economics)
* [Simon Scheidegger](https://sites.google.com/site/simonscheidegger) (University of Lausanne, Department of Economics)

### Citation

Please cite [Machine Learning for Dynamic Incentive Problems, P. Renner, S. Scheidegger, 2018](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3282487)
in your publications if it helps your research:

```
@article{rennerscheidegger_2018,
  title={Machine learning for dynamic incentive problems},
  author={Renner, Philipp and Scheidegger, Simon},
  year={2018},
  url = "https://ssrn.com/abstract=3282487",
  journal={Available at SSRN 3282487}
}
```

### Illustrative examples

**Introductory example on Gaussian Process Regression:** To illustrate how Gaussian Process regression can be applied to approximate functions, we provide a [simple
notebook](analytical_examples/Gaussian_process_regression_in_1d.ipynb).

**Introductory examples on Bayesian Active Learning:** To illustrate how Bayesian Active Learning in conjunction with Gaussian Process regression can be used to approximate functions, we provide a [1-dimensional example](analytical_examples/BAL_with_GPs.ipynb) as well as a [multi-dimensional example](analytical_examples/MultiDimBAL.ipynb).

**Baseline model by [Fernandes and Phelan
(2000)](https://www.sciencedirect.com/science/article/abs/pii/S0022053199926194):** We provide the code used to solve our [baseline model (section 4.6)](dptorch/AS_golosov_para), including the [result files](dptorch/results_paper/baseline), and an [explanation](dptorch/AS_golosov_para/README.md) on how to run the code.

**Heterogeneous agent model:** We provide the code used to solve our [adverse selection model with heterogeneous agents (section 5)](dptorch/AS_het_agent), including the [result files](dptorch/results_paper/het_agent), and an [explanation](dptorch/AS_het_agent/README.md) on how to run the code.


## Usage
We provide implementations which use python 3.


## Support

This work was generously supported by grants from the Swiss National Supercomputing Centre (CSCS) under project IDs s885, s995, the Swiss Platform for Advanced Scientific Computing (PASC) under project ID ["Computing equilibria in heterogeneous agent macro models on contemporary HPC platforms"](https://www.pasc-ch.org/projects/2017-2020/call-for-pasc-hpc-software-development-project-proposals), the [Swiss National Science Foundation](https://www.snf.ch) under project IDs “New methods for asset pricing with frictions”, "Can economic policy mitigate climate change", and the [Enterprise for Society (E4S)](https://e4s.center).
