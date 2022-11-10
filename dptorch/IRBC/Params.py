from math import exp
import numpy as np

def dynamic_params(cfg):
    """Dynamic parameter transforms"""
    cfg["model"]["params"]["A_tfp"] = (1.0 - cfg["model"]["params"]["beta"]*(1.0 - cfg["model"]["params"]["delta"])) / (
        cfg["model"]["params"]["alpha"] * cfg["model"]["params"]["beta"]
    )

    cfg["model"]["params"]["pareto"] = cfg["model"]["params"]["A_tfp"]**(1.0/cfg["model"]["params"]["gamma"])

    cfg["model"]["params"]["aMin"] = np.exp(-0.8*cfg["model"]["params"]["sigE"]/(1.0-cfg["model"]["params"]["rhoZ"]))
    cfg["model"]["params"]["aMax"] = np.exp(0.8*cfg["model"]["params"]["sigE"]/(1.0-cfg["model"]["params"]["rhoZ"]))

    cfg["model"]["params"]["nShocks"] = cfg["model"]["params"]["nCountries"]+1

    cfg["model"]["params"]["nPols"] = cfg["model"]["params"]["nCountries"]+1

    return cfg
