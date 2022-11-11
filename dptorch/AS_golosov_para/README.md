## How to run the Fernandes and Phelan (2000) benchmark model
To start the computation from scratch:
- In config/kernel/PiecewisePolynomialKernel.yaml set q:1
- In config/model/AS_golosov_para.yaml  set params: AS_highrisk
- In config/config.yaml set
  - scipyopt: default
  - model: AS_golosov_para
  - STARTING_POINT: NEW
  - no_samples: 128

To load the run from the paper:
- In config/config.yaml set
  - dir: runs/${model.MODEL_NAME}/highrisk
  - model: AS_golosov_para
  - STARTING_POINT: Iter_259.pth
  - BAL: enabled: false
  
Run `python run_dpgp.py`

## How to analyze the pre-computed solutions
- In config/kernel/PiecewisePolynomialKernel.yaml set q:1
- In config/model/AS_golosov_para.yaml  set params: AS_highrisk
- In config/config.yaml set
  - dir: runs/${model.MODEL_NAME}/highrisk
  - scipyopt: default
  - model: AS_golosov_para
  
Run `python post_process.py RUN_DIR=AS_golosov_para/highrisk CHECKPOINT_FILE=Iter_259.pth`



