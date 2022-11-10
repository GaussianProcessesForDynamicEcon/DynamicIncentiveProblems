## How to run the dynamic adverse selection model with heterogeneous agents.
- In config/kernel/PiecewisePolynomialKernel.yaml set q:0
- In config/config.yaml set
  - scipyopt: AS_het_agent
  - model: AS_het_agent
  - STARTING_POINT: NEW
  - no_samples: 512
To load the run from the paper
- In config/config.yaml set
  - dir: results_paper/baseline/highrisk
  - model: AS_golosov_para
  - STARTING_POINT: Iter_259.pth
  - BAL: enabled: false
  
Run python run_dpgp.py

## How to analyze the pre-computed solutions
- In config/kernel/PiecewisePolynomialKernel.yaml set q:0
- In config/config.yaml set
  - dir: runs/${model.MODEL_NAME}/highrisk
  - scipyopt: AS_het_agent
  - scipyopt: default
  - model: AS_golosov_para
  
Run python post_process.py RUN_DIR=results_paper/baseline/highrisk CHECKPOINT_FILE=Iter_259.pth
