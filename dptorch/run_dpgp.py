import torch
import hydra
from omegaconf import OmegaConf
import os
import sys
import shutil
import logging
import importlib
import pathlib
from hydra.utils import get_original_cwd

#import traceback
#import warnings
#import sys

#def warn_with_traceback(message, category, filename, lineno, file=None, line=None):

    #log = file if hasattr(file,'write') else sys.stderr
    #traceback.print_stack(file=log)
    #log.write(warnings.formatwarning(message, category, filename, lineno, line))

#warnings.showwarning = warn_with_traceback

# disable cyipopt INFO logs which are extremely verbose
logging.getLogger("cyipopt").setLevel(30)
rank = int(os.environ.get("OMPI_COMM_WORLD_RANK", "-1"))
if rank != -1:
    logger = logging.getLogger(f"{__name__}.Rank{rank}")
else:
    logger = logging.getLogger(f"{__name__}")


torch.set_num_threads(1)
torch.set_num_interop_threads(1)

#### Configuration setup
@hydra.main(
    config_path="config",
    config_name="config.yaml",
)
def set_conf(cfg):
    seed_offset = 0

    # any override parameters
    cfg = OmegaConf.to_container(cfg)
    cfg.update(
        {
            "cwd": os.getcwd(),
        }
    )

    # distributed setup
    if os.getenv("OMPI_COMM_WORLD_SIZE"):
        import torch.distributed as dist

        cfg["distributed"] = True
        dist.init_process_group(backend="mpi", rank=0, world_size=0)
    else:
        cfg["distributed"] = False

    # RNG
    cfg["mc_seed"] = cfg["seed"] + int(os.environ.get("OMPI_COMM_WORLD_RANK", 0))
    torch.manual_seed(cfg["mc_seed"])

    model = importlib.import_module(cfg["model"]["MODEL_NAME"] + ".Model")

    try:
        # dynamic parameters
        cfg = importlib.import_module(
            cfg["model"]["MODEL_NAME"] + ".Params"
        ).dynamic_params(cfg)
        try:
            # adjust learning rate
            if 'lr' in cfg['torch_optim']['config'] and cfg['override_checkpoint_configs_optimizer']:
                cfg['torch_optim']['config']['lr'] /= cfg['no_samples']
                logger.info(f"Learning rate adjusted with the number of points. Effective LR: {cfg['torch_optim']['config']['lr']}")
        except:
            logger.debug("No learning rate detected, not adjusted.")
    except FileNotFoundError:
        logger.warning(
            "No Params.py found - continuing with using only parameters from yaml file."
        )
    except AttributeError:
        logger.warning(
            "No method named `dynamic_params` found in Params.py - continuing using only parameters from yaml file."
        )

    logger.info("Running with parameters: " + str(cfg))

    gp_model = importlib.import_module("GPModels." + cfg["model"]["GP_MODEL"]["name"])

    expect = importlib.import_module("ExpectOps." + cfg["model"]["EXPECT_OP"]["name"])

    save_file = None
    if cfg["STARTING_POINT"] == "NEW":
        pass
    elif cfg["STARTING_POINT"] == "LATEST":
        model_dir = f"{get_original_cwd()}/runs/{cfg['model']['MODEL_NAME']}"
        date_dirs = sorted([elem for elem in pathlib.Path(model_dir).iterdir() if elem.is_dir()], key=lambda elem: elem.name, reverse=True)
        for date_dir in date_dirs:
            time_dirs = sorted([elem for elem in date_dir.iterdir() if elem.is_dir()], key=lambda elem: elem.name, reverse=True)
            for time_dir in time_dirs:
                logger.debug(f"Searching folder for latest save file: {time_dir.resolve()}")
                dir_contents = sorted([elem for elem in time_dir.iterdir()], key=os.path.getmtime, reverse=True)
                save_file = next((pth for pth in dir_contents if pth.name.endswith(".pth")), None)
                if save_file:
                    break
            if save_file:
                break
    else:
        save_file = cfg["STARTING_POINT"]

    if save_file:
        cfg.update({"init_with_zeros": False})
        # load
        logger.info(f"Resuming from state {save_file}")
        m = model.SpecifiedModel.load(
            path=save_file,
            cfg_override=cfg,
            Model=gp_model.GPModel,
            ExpectOps=expect.ExpectOps,
        )
    else:
        logger.info(f"Starting from scratch")
        m = model.SpecifiedModel(
            Model=gp_model.GPModel,
            ExpectOps=expect.ExpectOps,
            cfg=cfg,
            beta=cfg["model"]["params"]["beta"],
        )

    for i in range(cfg["num_cycles"]):
        #logger.info(f"Starting iteration: {i}")
        m.iterate(cfg["torch_optim"]["iter_per_cycle"])



set_conf()
