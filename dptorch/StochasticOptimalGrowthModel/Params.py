def dynamic_params(cfg):
    """Dynamic parameter transforms"""
    cfg["model"]["params"]["big_A"] = (1.0 - cfg["model"]["params"]["beta"]) / (
        cfg["model"]["params"]["psi"] * cfg["model"]["params"]["beta"]
    )
    if not "no_samples" in cfg:
        cfg["no_samples"] = 10 * cfg["model"]["params"]["n_agents"]

    # used for expectations
    if cfg["model"]["EXPECT_OP"]["name"] != "SinglePoint":
        cfg["model"]["EXPECT_OP"]["config"]["mc_std"] = [
            cfg["model"]["EXPECT_OP"]["config"]["sigma"]
        ] * cfg["model"]["params"]["n_agents"]

    return cfg
