## How to run the dynamic adverse selection model with heterogeneous agents.
To start the run from scratch:
- In config/kernel/PiecewisePolynomialKernel.yaml set q:0
- In config/config.yaml set
  - scipyopt: AS_het_agent
  - model: AS_het_agent
  - STARTING_POINT: NEW
  - no_samples: 512

To load the run from the paper:
- In config/config.yaml set
  - dir: results_paper/baseline/highrisk
  - model: AS_het_agent
  - STARTING_POINT: Iter_1355.pth
  - BAL: enabled: false
  
Run `python run_dpgp.py`

## How to analyze the pre-computed solutions
- In config/kernel/PiecewisePolynomialKernel.yaml set q:0
- In config/config.yaml set
  - dir: runs/${model.MODEL_NAME}/highrisk
  - scipyopt: AS_het_agent
  - model: AS_het_agent
  
Run `python post_process.py RUN_DIR=AS_het_agent/model CHECKPOINT_FILE=Iter_1355.pth`
