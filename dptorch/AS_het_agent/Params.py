import importlib
import numpy as np
import torch

def dynamic_params(cfg):
    """Dynamic parameter transforms"""
    model = importlib.import_module(
            cfg["model"]["MODEL_NAME"] + ".Model"
        )
    reg_c = cfg["model"]["params"]["reg_c"]
    sigma = cfg["model"]["params"]["sigma"]
    upperb = torch.tensor(cfg["model"]["params"]["upperb"])
    lowerb = torch.tensor(cfg["model"]["params"]["lowerb"])
    n_agents = cfg["model"]["params"]["n_agents"]
    n_shocks = cfg["model"]["params"]["n_shocks"]
    cfg["model"]["params"]["n_types"] = n_agents*n_shocks
    n_types = cfg["model"]["params"]["n_types"]

    ds_lst = [(indxa,indxs) for indxa in range(n_agents) for indxs in range(n_shocks)] #list of discrete states
    assert len(ds_lst)==n_types

    cfg["model"]["params"]["discrete_state_dim"] = len(ds_lst)
    cfg["model"]["params"]["ds_list"] = ds_lst
    beta = cfg["model"]["params"]["beta"]

    ##############################################################
    # markov chain with matrix  [[1-alpha,alpha],[beta,1-beta]]
    # has stationary dist       [p,1-p]  with p = beta/(beta+alpha)
    # pick beta such that p = 0.4
    trans_mat_ag = torch.tensor(
    [
        [
            [0.9,0.1],
            [0.1,0.9]
        ],
        [
            [0.9,0.1],
            [0.1*4/6,1-0.1*4/6]
        ],])

    cfg["model"]["params"]["agents_trans_mat"] = trans_mat_ag
    
    assert cfg["model"]["params"]["agents_trans_mat"].shape[0] == n_agents, f"we need {n_agents} transition matrices"
    assert cfg["model"]["params"]["agents_trans_mat"].shape[1] == n_shocks and cfg["model"]["params"]["agents_trans_mat"].shape[2] == n_shocks, f"we need {n_shocks} sized transition matrices"
    # mapping indivuduals trans prob to large trans prob
    trans_mat = torch.zeros([n_types,n_types])
    for indxa in range(n_agents):
        for indxr in range(n_shocks):
            for indxc in range(n_shocks):
                indxrow = ds_lst.index((indxa,indxr))
                indxcol = ds_lst.index((indxa,indxc))
                trans_mat[indxrow,indxcol] = cfg["model"]["params"]["agents_trans_mat"][indxa,indxr,indxc]

    cfg["model"]["params"]["trans_mat"] = trans_mat
    cfg["model"]["params"]["trans_mat_inv"] = torch.inverse(trans_mat)

    shock_mat = torch.tensor(
        [[cfg["model"]["params"]["lowerh"], cfg["model"]["params"]["upperh"]],
        [cfg["model"]["params"]["lowerh"], cfg["model"]["params"]["upperh"]]])

    
    shock_vec = torch.zeros(n_types)
    for indxt in range(n_types):
        shock_vec[indxt] = shock_mat[ds_lst[indxt][0],ds_lst[indxt][1]]
    assert shock_vec.shape[0] == n_types, "Number of shocks and types must match"
    cfg["model"]["params"]["shock_vec"] = shock_vec

    #promise utility setup
    state_names = [f"w_{i+1}" for i in range(n_types)]    
    cfg["model"]["params"]["state_names"] = state_names

    upper_w = model.utility_ind(upperb, reg_c, sigma)#/(1-beta)
    cfg["model"]["params"]["upper_w"] = torch.tensor(
        list([upper_w for indxs in range(n_types)]))
    
    lower_w = model.utility_ind(lowerb, reg_c, sigma)#/(1-beta)
    cfg["model"]["params"]["lower_w"] = torch.tensor(
        list([lower_w for indxs in range(n_types)]))

   
    #policy function names and scaling
    cfg["model"]["params"]["policy_dim"] = 4 * cfg["model"]["params"]["n_types"] + (cfg["model"]["params"]["n_types"])**2 #dimension of policy vector; has to be larger than control vector
    cfg["model"]["params"]["control_dim"] = cfg["model"]["params"]["policy_dim"]
    assert cfg["model"]["params"]["policy_dim"] >= cfg["model"]["params"]["control_dim"]
    policy_names = []
    policy_names += [f"c_{i+1}" for i in range(n_types)]
    policy_names += [f"u_{i+1}" for i in range(n_types)]
    policy_names += [
        f"fut_util_{indxr+1}_{indxc+1}" 
        for indxc in range(n_types) 
        for indxr in range(n_types)]

    policy_names += [f"pen_{i+1}" for i in range(n_types)]
    policy_names += [f"pen_u_{i+1}" for i in range(n_types)]
    cfg["model"]["params"]["policy_names"] = policy_names
    P = {key: val for val, key in enumerate(policy_names)}
    assert len(policy_names) == cfg["model"]["params"]["policy_dim"] 

    scale_pol_vec = torch.ones(cfg["model"]["params"]["policy_dim"])
    # for indxr in range(n_types):
    #     scale_pol_vec[P[f"u_{indxr+1}"]] = scale_pol_vec[P[f"u_{indxr+1}"]]/(1-beta)
    #     scale_pol_vec[P[f"c_{indxr+1}"]] = scale_pol_vec[P[f"c_{indxr+1}"]]/(1-beta)
    #     scale_pol_vec[P[f"pen_u_{indxr+1}"]] = scale_pol_vec[P[f"pen_u_{indxr+1}"]]/(1-beta)
    #     scale_pol_vec[P[f"pen_{indxr+1}"]] = scale_pol_vec[P[f"pen_{indxr+1}"]]/(1-beta)
    #     for indxc in range(n_types):
    #         scale_pol_vec[P[f"fut_util_{indxr+1}_{indxc+1}"]] = scale_pol_vec[P[f"fut_util_{indxr+1}_{indxc+1}"]]/(1-beta)

    cfg["model"]["params"]["scale_pol_vec"] =  scale_pol_vec
    
    #value function bounds and scaling
    cfg["model"]["params"]["lower_V"] =  (torch.tensor(-upperb+cfg["model"]["params"]["lowerh"]))/(1-beta)
    cfg["model"]["params"]["lower_V_vec"] =  torch.tensor([
        (0.5*-upperb+cfg["model"]["params"]["lowerh"]) + (0.5*-upperb+cfg["model"]["params"]["upperh"]),
        (0.5*-upperb+cfg["model"]["params"]["lowerh"]) + (0.5*-upperb+cfg["model"]["params"]["upperh"]),
        (0.4*-upperb+cfg["model"]["params"]["lowerh"]) + (0.6*-upperb+cfg["model"]["params"]["upperh"]),
        (0.4*-upperb+cfg["model"]["params"]["lowerh"]) + (0.6*-upperb+cfg["model"]["params"]["upperh"]),
        ])/(1-beta)

    cfg["model"]["params"]["upper_V"] =  torch.tensor(cfg["model"]["params"]["upperh"])/(1-beta)

    cfg["model"]["params"]["GP_offset"] = cfg["model"]["params"]["lower_V"] - 0.1/(1-beta) #translate the gp by this amount
    cfg["model"]["params"]["GP_min"] = torch.tensor(-4.00)/(1-beta)


    if not "no_samples" in cfg:
        cfg["no_samples"] = 10 * n_agents

    # used for expectations
    if cfg["model"]["EXPECT_OP"]["name"] != "SinglePoint":
        cfg["model"]["EXPECT_OP"]["config"]["mc_std"] = [
            cfg["model"]["EXPECT_OP"]["config"]["sigma"]
        ] * cfg["model"]["params"]["n_agents"]

    return cfg
