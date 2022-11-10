import numpy as np

def dynamic_params(cfg):
    """Dynamic parameter transforms"""

    l_inp = np.array(cfg["model"]["params"]["l"])
    A = cfg["model"]["params"]["A"]
    l = np.interp([x * (12 / A) for x in range(A)], range(12),l_inp)
    
    multiplier = (
        1 - cfg["model"]["params"]["beta"] * (1 - cfg["model"]["params"]["delta"])
    ) / (cfg["model"]["params"]["alpha"] * cfg["model"]["params"]["beta"])
    
    cfg["model"]["params"]["tfp"] = [
        [y * multiplier for y in x] for x in cfg["model"]["params"]["tfp"]
    ]

    cfg["model"]["params"]["l"] = [
        x * A / sum(l)
        for x in l
    ]

    cfg["model"]["params"]["L"] = [
        x * A for x in cfg["model"]["params"]["L"]
    ]

    return cfg
