import os
import glob
import importlib
import hydra
import logging
import logging.config
import torch
from DPGPModel import DPGPModel
from DPGPIpoptModel import DPGPIpoptModel
import cyipopt
from omegaconf import OmegaConf

logger = logging.getLogger(__name__)


#### Configuration setup
@hydra.main(
    config_path="config",
    config_name="postprocess.yaml",
)
def set_conf(cfg):
    logger.info(OmegaConf.to_yaml(cfg))
    cfg_run = OmegaConf.load(
        hydra.utils.to_absolute_path(f"runs/{cfg.RUN_DIR}/.hydra/config.yaml")
    )
    logger.info("Original configuration:")
    logger.info(OmegaConf.to_yaml(cfg_run))
    model = importlib.import_module(cfg_run.model.MODEL_NAME + ".Model")
    gp_model = importlib.import_module("GPModels." + cfg_run.model.GP_MODEL.name)
    expect = importlib.import_module("ExpectOps." + cfg_run.model.EXPECT_OP.name)

    # RNG
    torch.manual_seed(0)

    # get checkpoints
    checkpoints = list(
        sorted(
            glob.glob(f"{hydra.utils.get_original_cwd()}/runs/{cfg.RUN_DIR}/*.pth"),
            key=os.path.getmtime,
        )
    )

    if cfg.CHECKPOINT_FILE == "LATEST":
        CHECKPOINT_FILE = checkpoints[-1]
        if len(checkpoints) > 1:
            CHECKPOINT_FILE_PREV = checkpoints[-2]
        else:
            CHECKPOINT_FILE_PREV = checkpoints[-1]
    else:
        indx_pos = -1
        for indx in range(len(checkpoints)):
            if checkpoints[indx].endswith(cfg.CHECKPOINT_FILE):
                indx_pos = indx

        if indx_pos == -1:
            raise FileNotFoundError("Specified checkpoint file not found")
        elif indx_pos == 0:
            CHECKPOINT_FILE = checkpoints[0]
            CHECKPOINT_FILE_PREV = CHECKPOINT_FILE
        else:
            CHECKPOINT_FILE = checkpoints[indx_pos]
            CHECKPOINT_FILE_PREV = checkpoints[indx_pos-1]

    logger.info(f"Loading checkpoint file: {CHECKPOINT_FILE}")

    # load the specified model
    m = model.SpecifiedModel.load(
        path=CHECKPOINT_FILE,
        # no override, use saved params
        cfg_override={"distributed": False, "init_with_zeros": False},
        Model=gp_model.GPModel,
        ExpectOps=expect.ExpectOps,
    )
    m_prev = model.SpecifiedModel.load(
        path=CHECKPOINT_FILE_PREV,
        # no override, use saved params
        cfg_override={"distributed": False, "init_with_zeros": False},
        Model=gp_model.GPModel,
        ExpectOps=expect.ExpectOps,
    )

    logging.getLogger("DPGPModel").setLevel(30)

    # call the post_processing script for the model
    pp = importlib.import_module(cfg_run.model.MODEL_NAME + ".PostProcess")

    #pp.logger.setLevel(logging.WARNING)

    pp.process(m, cfg)
    
    err_out = open(f"V_func_error.txt", 'w')
    err_out.write("#T1 T2 L2 LInf \n")   
    err_out.close()    
    # iterate over checkpoints
    if len(checkpoints) > 1:
        for i in range(len(checkpoints) - 1):
            m1 = model.SpecifiedModel.load(
                path=checkpoints[i],
                # no override, use saved params
                cfg_override={"distributed": False, "init_with_zeros": False},
                Model=gp_model.GPModel,
                ExpectOps=expect.ExpectOps,
            )
            m2 = model.SpecifiedModel.load(
                path=checkpoints[i + 1],
                # no override, use saved params
                cfg_override={"distributed": False, "init_with_zeros": False},
                Model=gp_model.GPModel,
                ExpectOps=expect.ExpectOps,
            )
            pp.compare(m1, m2, cfg)

    pp.simulate(m,m_prev, cfg, model)


set_conf()
