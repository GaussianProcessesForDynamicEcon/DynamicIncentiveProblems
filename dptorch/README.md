# Dynamic programming with GPyTorch

Dependencies:

- pytorch (for MPI support compiled from source - see Usage section below)
- gpytorch (go to https://pypi.org/project/gpytorch; pip install gpytorch)
- hydra-core >= 1.1 (goto: https://hydra.cc/docs/intro; pip install hydra-core --upgrade)
- cyipopt (go to https://pypi.org/project/ipopt; pip install ipopt)

It's easiest to install the above via conda. If you require MPI support, please read below as PyTorch needs to be compiled from source.
## Approach

A generic DPGP (Dynamic Programming with Gaussian Processes) has the following structure (see. e.g `StochasticOptimalGrowthModel/Model.py`):

1) A way to generate sample points
2) A way to fit a GP to value-function estimates
3) A way to propagate states forward and calculate expectations of the VF
4) A way to solve the Bellman-Equation (VFI)

**Note**: The library also support FOC based models without VFI, see a few sections below (and see. e.g `SJE_BKS/Model.py`).

The `DPGPModel` abstract base-class is designed to encapsulate this behaviour. In particular it accepts a Gpytorch GPModel and implements 2) and the calculation of 3) given an integration method.

It has also some additional utility functions:
- Exposes the gradients of expected Value-Function that can be used for subclasses.
- Handles MPI actions

To create a fully specified model, certain methods need to be specified (e.g. utility function, solve, etc... ) in derived classes.

The library will perform the following iteration steps:

1) Require a VF sample (or initial guess function) to start the iteration
2) Solve the VF iteration problem to generate a new set of VF samples and policy estimates
3) If doing a checkpoint (specified via `CHECKPOINT_INTERVAL` flag), then also run a GP fit on the policy samples before the checkpoint - can be slow! (unless explicitly disabled via `DISABLE_POLICY_FIT` flag)

### Ipopt based models

For Ipopt based models, there is a specific subclass `DPGPIpoptModel`, which implements the solve method using Ipopt. This class requires certain other methods to be specified (e.g. lb, ub, etc..)

For an example on how to create a specific model from this base class, check StochasticOptimalGrowthModel/Model.py and the `SpecifiedModel` class.

### Mixed state handling

The library supports mixed state models, it will fit a separate GP for each value / policy function for each discrete state. The fitting of the models will be parallelized across MPI workers if available (both for discrete state / policy dimension).

Discrete states are simply enumerated in the last column of the tensor `DPGPModel.state_sample`. Quantities depending on the discrete state are best enumerated as a list with the same dimension as number of discrete states, with each element of the list defining the complete state information (e.g. in the `SJE_BKS` model each element of the TFP variable will represent the TFP across both regions for the given discrete state).

**Note**: Continuous-state only models should be represented by setting `discrete_state_dim=1` in the model constructor. The discrete state value should be set to 0 in all sampling routines.

### FOC based models

In some cases a model solution can be defined without a Value-Function, just purely based on equilibrium FOCs (similar to DEQN). This is also supported by the library, in this case the `model.ONLY_POLICY_ITER` config value should be set to true. The behaviour of the library in this case will be to:

1) Require a policy sample to start the iteration
2) At the beginning of each cycle do a GP fit on the policy sample (the value functions will not be estimated and will have a dummy value of zero stored)
3) Solve the equilibrium conditions to get a new policy sample

For a reference implementation, check the `SJE_BKS` folder.

## Usage

```
python run_dpgp.py
```

or can be also executed via MPI, e.g:

```
mpirun -np 4 python run_dpgp.py hydra.verbose=True
```

**NOTE**: For MPI support PyTorch needs to be compiled form source using the guide at https://pytorch.org/tutorials/intermediate/dist_tuto.html#communication-backends (easiest to work in a conda environment). Note use the build command `env MAX_JOBS=8 python setup.py build`

In this case the learning rate is not adjusted as only a single node (master) is used for learning the GP params after gathering the solutions.

### Configuration

There are 2 components to configuring a model run:

1) config/model/<MODEL_NAME>.yaml (see for example config/model/SOGM.yaml)

This file contains the basic settings for the model:

Details:
 - model.MODEL_NAME: the folder containing the model definition and parameters
 - model.GP_MODEL.name: the module (in the GPModels folder) containing the Gaussian-Process specification
 - model.EXPECT_OP: the specified expectation method (see ExpectOps folder for available methods) => this is only relevant for certain models, for complex cases a specific implementation of state_iterate_exp might be needed.
 - model.params: Any additional parameters used in the model / optimizer, etc... This is passed to all instance of DPGPModel.

**Hint**: A "deep" pre-processor to a low-dimensional GP seems to generally work quite well (see e.g `SOGM_deep.yaml`), just make sure the learning rate / number of iterations is not too high (and the number of sample points not too low) to avoid converging to some wiggly local solution.

2) Params.py

A dynamic_params function can be placed in this module. This will be applied to the parameteres passed via the yaml file. It can be used to calculate dynamic transforms or create Python objects not representable via yaml types.

#### Optional Flags

See the config/ folder and subfolders for configuration options. These should be altered with the appropriate namespacing (e.g. `torch_optim.config.lr=0.001` or `model.GP_MODEL.config.active_dim=3`).

To restart from a specific checkpoint, you can set the `STARTING_POINT=<checkpoint file>` config value.

To restart from the latest checkpoint of a given model, you can set the `STARTING_POINT=LATEST` config value. In this case, the run folders of the model specified in the config will be scanned for the latest checkpoint file.
If none is found, the computation will start from scratch.

Setting `STARTING_POINT=NEW` will also start the computation from scratch.

##### Logging

When running the code with MPI, the output can become hard to read due to the many processes emitting logs at the same time.
To help with this situation, there are two flags that control the log level of all non-zero rank MPI worker simultaneously (the rank 0
process will always log at the level set by hydra).

Additionally, there are two loggers set up in DPGPModel.py: the "main" logger and the "train" logger. The train logger will log event related to GP training, while the main logger will log the rest of the event.

You can set the log level of both independently with the `WORKER_MAIN_LOG_LEVEL` and `WORKER_TRAIN_LOG_LEVEL` flags. For example, to log only the
train events on non-0 rank workers, you can use the setting

```
WORKER_MAIN_LOG_LEVEL: ERROR
WORKER_TRAIN_LOG_LEVEL: INFO
```

Note that these settings only set the log level of the logger objects. If the main hydra level is set higher (e.g. INFO, the base value), you will not see log messages from lower levels in the logs (e.g. DEBUG). To make DEBUG level logs appear, use `hydra.verbose=True` when starting the script.

### GPU support

**TODO**: Fix GPU support as currently GPU support is incomplete and probably doesn't work in the most recent version. In past versions total GPU performance wasn't better than CPU due to Ipopt being CPU implemented.

If a GPU is available, it will be used to evaluate the GP (this hasn't been tested fully in each model, so might require additional work). However the rest of the functions are currently assumed to run on CPU (e.g. utility function), to minimize copying between CPU and GPU (since Ipopt is running on CPU, we need to copy the function evaluations, gradients, etc.. always back to CPU anyway).

### Bayesian Active Learning

BAL can be turned on in `config.yaml` by setting `BAL.enabled=true`. In this case additional parmeters specifying frequency, size and max points need to be specified.

In addition, multiple targets can be set, where each target is independently evaluated. A target specifies which policy (0=VF), which discrete state (not specified means optimal points are looked across discrete states) and preference (rho, beta) should be specified.

The total number of points (not necessarily distinct!) added in each iteration = [ number of targets ] x points_per_iter

For an example see below:

```
BAL:
  enabled: true
  points_per_iter: 5
  epoch_freq: 1
  max_points: -1 # no limit
  targets:
  - policy: 0 # best across all discrete states
    rho: 0.0
    beta: 1.0
  - discrete_state: 0 # only look at discrete state 0
    policy: 1
    rho: 0.0
    beta: 1.0
```

### Dynamic resampling for BAL

States can be resampled in 3 ways: disabled, random or dynamic, which can be set in `config.yaml`, by setting `resample_method`.
- 'disabled' means the initial sample in the first iteration will be used in all subsequent iterations
- 'random' means each interation will generate its own random state sample
- 'dynamic' means each state sample in an iteration is fed to `state_iterate_exp` to generate a list of possible future states along with their corresponding probabilities. Then, `dynamic_resample_num_new` states are drawn from the list of possible future dates, according to their probabilites (set it to -1 to use all possible future states). The new state sample will be the cumulative sum of such drawn states. If BAL is enabled, it'll use this new state sample for each defined BAL target.

### Optimizers

Different torch optimizers can be defined by creating optimizer profiles in the `torch_optim` folder (any valid Torch optimizer is accepted, with the config structure being passed to the optimizer definition)

If `torch_optim=bfgs` is selected, then `PyTorch-LBFGS` is used.

The flag `reset_optimizer_every_timestep` controls if the GP models and their optimizers should be reinitalized on each timestep.
This can help in avoiding getting stuck in local minima.

### Kernels

Kernels can be specified by adding additional Kernel configurations in the `kernel` directory or simply overriding `kernel.name` on the command-line. The config structure can be used to pass parameters to the GPyTorch kernels.

## Post-processing

To run post-processing, the `post_process.py` script can be used. An example is given here:

```
python post_process.py RUN_DIR=StochasticOptimalGrowthModel/2021-06-16/11-53-17
```

This will load the latest saved checkpoint (from the `runs/{RUN_DIR}` file) and will run the `PostProcess.py` module from the `{MODEL_NAME}` folder, passing the loaded model to the module. The output of the module will be in `postprocess/{RUN_DIR}`.

A specific checkpoint file can also loaded via specifying a `CHECKPOINT_FILE` parameter:

```
python post_process.py RUN_DIR=StochasticOptimalGrowthModel/2021-06-16/11-53-17 CHECKPOINT_FILE=Iter_9.pth
```

Each model can have its own post-processing script and also post-processing outputs will be in a matching folder compared to the run outputs.

An example post-processing script can be found in `StochasticOptimalGrowthModel/PostProcess.py` which will plot the estimated VF (+confidence bounds) as a function of State 0.

> **_NOTE:_**  The output logs will contain the full command needed to run the post-processing, so it's not needed in general to manually copy the RUN_DAY, etc.. from the folder names.

In the PostProcessing script for Ipopt based models one can invoke the `DPGPIpoptModel.generate_trajectory`, which will generate a trajectory of states, controls and `eval_g` values. This does the following:
1. Starts from a random state
2. Uses the GPs to generate a policy prediction
3. Evaluates the `eval_g` function using state + policies
4. Uses `state_iterate_expectation` to generate a new set of candidate samples and samples them according to the associated weights returned by the same method.
5. Proceeds from 2. using the new sample

The generated values can be saved to csv, see for examples `StochasticOptimalGrowthModel/PostProcess.py`.

The `generate_trajectory` function accepts a `from_sample` argument, which if True will use a random sample to evaluate `eval_g` instead of iteratively generating a trajectory.

## Adding a new model

To add a new model, create a new folder with the model name in the directory of this code. There need to be 3 files:


- `Model.py`: This needs to contain a class named `SpecifiedModel` which is a subclass of the DPGPModel / DPGPIpoptModel class
- `Params.py`: This is used to dynamically update parameters
- `PostProcess.py`: This will be called by the post-processing code. You can copy it from an existing model and customize it as necessary.

The class in `Model.py` needs to implement the following methods:

  - `sample`: This function generates new samples for the model. The last column is always interpreted as an **enumeration** of discrete states and should be set to zero even for continuous only models
  - `state_iterate_exp`: This function should take a state value and return a tuple of weights and future states. The conditional expectation will be a weighted sum of the value function on future states, weighted by the weight values.
  - `u`: The 'value' function of the dynamic programming problem

For ipopt based models additional methods need to be implemented:
  - `lb`, `ub`, `cu`, `cl`, `eval_g`: the lower / upper bounds, the constrain lower/upper bounds, equality constraint function

**Note**: It is recommended to initially use the derivative checker of ipopt (second-order) to see if the right gradients are returned by torch autograd. The derivative checker can be turned on by adding the CLI argument: `ipopt.derivative_test=second-order` (the perturbation size can be set via `ipopt.derivative_test_perturbation`).

If not, then verify that __all calculations depending on the control are saved to a tensor type before the final results are calculated, as this is required for Torch to track gradients__.

### Referencing states / policies

States and Policies can be referenced by name if the `state_names` and `policy_names` vectors are provided to the `DPGPModel` constructor. See an example of this in `SJE_BKS/Model.py`. Note that if estimated policies need to be called, those can be done via `self.M[d][1 + policy_index]` where `d` is the discrete state dimension and `policy_index` references the policy number (see `SJE_BKS/Model.py` on how to use `policy_names` to get the `policy_index` via `self.P`). The offset of `1+` is needed as the 0-th index element is always the estimated value function.

### Expectation calculation

In certain cases it might be simpler to separate the dynamics of the state from the expectation calculation. It is possible to only define a `state_iterate` method in the model, in which case an expectation operator will need to be applied (i.e. how do we generate samples 'around' the propagated state).

So the state propagation logic is as follows:
1) Given today's state and controls calculate the conditional expectation of tomorrow's states using `state_iterate`
2) For continuous states, apply shocks around the propagated state, using the Expectation Operator:
  - SinglePoint
  - MCAdditiveStandardNormal: Scaled standard normal shocks with parameters:
     - mc_std: The standard deviation of each continous state, should be a vector
     - n_mc_sample: The number of samples to draw for each continuous state, defaults to 10
  - Monomial: Monomial integration rule:
     - mc_std: The standard deviation of each continous state, should be a vector

For an example on how to configure the expectation operator check the `config/model/SOGM_deep.yaml` and the `StochasticOptimalGrowthModel/Params.py` file.

3) Expand the generated weights and points to include all discrete states and multiply each with the transition probability from today's discrete state. So the full number of points evaluated will be `n_mc_sample x discrete_state_dim` in the case of MC Normal integration for example.

For mixed state models, an additional method called `pi` should be added for step 3) above (taking inputs current state), which specifies the transition probabilities (the full matrix). It can be made conditional on state if necessary, but in most cases should return a constant matrix and `state_iterate_exp` will then automatically select the right transition probabilities. The transition probabilities should be as `pi_i_j = The transition probability to state j, given today's state is i`. Defaults to square matrix with all entries = `1/discrete_state_dim`.

If provided, it can be overridden by specifying the `state_iterate_exp` method which will then be used.
