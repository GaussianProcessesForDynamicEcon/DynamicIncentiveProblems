import importlib
import torch

def dynamic_params(cfg):
    """Dynamic parameter transforms"""
    model = importlib.import_module(
            cfg["model"]["MODEL_NAME"] + ".Model"
        )
    cfg["model"]["params"]["shock_vec"] = torch.tensor([0.1 , 0.35])
    cfg["model"]["params"]["trans_mat"] = torch.tensor([
            [cfg["model"]["params"]["Pi_1_1"],cfg["model"]["params"]["Pi_1_2"]],
            [cfg["model"]["params"]["Pi_2_1"],cfg["model"]["params"]["Pi_2_2"]]])
    cfg["model"]["params"]["trans_mat_inv"] = torch.inverse(cfg["model"]["params"]["trans_mat"])
    shock_vec = cfg["model"]["params"]["shock_vec"]
    reg_c = cfg["model"]["params"]["reg_c"]
    sigma = cfg["model"]["params"]["sigma"]
    upperb = cfg["model"]["params"]["upperb"]
    lowerb = cfg["model"]["params"]["lowerb"]
    n_agents = cfg["model"]["params"]["n_agents"]
    beta = cfg["model"]["params"]["beta"]
    cfg["model"]["params"]["discrete_state_dim"] = n_agents

    upper_w = model.utility_ind(upperb, reg_c, sigma)#/(1-beta)
    cfg["model"]["params"]["upper_w"] = torch.tensor([upper_w,upper_w])

    lower_w = model.utility_ind(lowerb, reg_c, sigma)#/(1-beta)
    #low_u = model.utility_ind(cfg["model"]["params"]["shock_vec"][1] - cfg["model"]["params"]["shock_vec"][0], reg_c, sigma) #last ineq enforces this
    low_u = model.utility_ind(lowerb, reg_c, sigma) #last ineq enforces this
    cfg["model"]["params"]["lower_w"] = torch.tensor([lower_w,lower_w])

    cfg["model"]["params"]["lower_V"] =  torch.tensor(-upperb+cfg["model"]["params"]["lowerh"])/(1-beta)
    cfg["model"]["params"]["upper_V"] =  shock_vec[1]/(1-beta)
    cfg["model"]["params"]["lower_V_vec"] =  torch.tensor([
        (0.5*-upperb+cfg["model"]["params"]["lowerh"]) + (0.5*-upperb+cfg["model"]["params"]["upperh"]),
        (0.5*-upperb+cfg["model"]["params"]["lowerh"]) + (0.5*-upperb+cfg["model"]["params"]["upperh"]),
        ])/(1-beta)

    cfg["model"]["params"]["GP_offset"] = cfg["model"]["params"]["lower_V"] - 0.1/(1-beta) #translate the gp by this amount
    cfg["model"]["params"]["GP_min"] = torch.tensor(-2.00)/(1-beta)

    if not "no_samples" in cfg:
        cfg["no_samples"] = 10 * n_agents

    # used for expectations
    if cfg["model"]["EXPECT_OP"]["name"] != "SinglePoint":
        cfg["model"]["EXPECT_OP"]["config"]["mc_std"] = [
            cfg["model"]["EXPECT_OP"]["config"]["sigma"]
        ] * cfg["model"]["params"]["n_agents"]

    return cfg
