from matplotlib import pyplot as plt
import torch
import gpytorch
import matplotlib
import importlib
import logging
import pandas as pd
import numpy as np
from Utils import NonConvergedError

matplotlib.use("Agg")

logger = logging.getLogger(__name__)


def process(m, cfg):

    lower_V = m.cfg["model"]["params"]["lower_V"]
    gp_offset = m.cfg["model"]["params"]["GP_offset"]
    sigma_util = m.cfg["model"]["params"]["sigma"]
    beta = m.cfg["model"]["params"]["beta"]
    n_agents = m.cfg["model"]["params"]["n_agents"]
    n_types = m.cfg["model"]["params"]["n_types"]
    lower_w = m.cfg["model"]["params"]["lower_w"]
    upper_w = m.cfg["model"]["params"]["upper_w"]
    scale_pol_vec = m.cfg["model"]["params"]["scale_pol_vec"] 

    # deleting the error file
    f = open("V_func_error.txt", 'w')
    n_plot_pts = 4000
    plot_sample_01 = (
        torch.rand(
            [100*n_plot_pts, m.cfg["model"]["params"]["n_types"]]
        )
    )

    mask_feas = torch.ones(plot_sample_01.shape[0], dtype=torch.bool)
    for indxa in range(n_agents):
        indxh = m.set_disc_state(indxa,1)
        indxl = m.set_disc_state(indxa,0)
        mask_feas = torch.logical_and(mask_feas,plot_sample_01[:,indxh] >= plot_sample_01[:,indxl]) #only upper traingle is feasible

    plot_sample_01= plot_sample_01[mask_feas,:]
    if plot_sample_01.shape[0] > n_plot_pts:
        plot_sample_01 = plot_sample_01[:n_plot_pts,:]
    else:
        assert False, "Not enough sample points"


    for indxd in range(n_types):
        plot_sample = (upper_w - lower_w) * plot_sample_01 + lower_w
        for indx in range(plot_sample.shape[0]):
            tmp_state = torch.zeros(plot_sample.shape[1] + 1)
            tmp_state[:-1] = plot_sample[indx, :]
            tmp_state[-1] = 1. * indxd
            plot_sample[indx, :] = tmp_state[:-1]

        d = indxd
        mask = m.state_sample[:, -1] == d * torch.tensor(1.)
        max_v_ind = torch.argmax(m.V_sample[mask] + gp_offset)
        min_v_ind = torch.argmin(m.V_sample[mask] + gp_offset)
        V = m.V_sample[mask] + gp_offset
        sample = m.state_sample_all[mask,:]
        logger.info(f">>>  maximal value in state {d} is {V[max_v_ind]} min is {V[min_v_ind]} state {sample[max_v_ind,:]}")    
        #index_lst = (mask.type(torch.IntTensor))
        eval_samples = plot_sample  # m.state_sample[mask,:-1]

        V_fun = m.M[d][0].eval()
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            V_pred = (V_fun(eval_samples).mean + gp_offset) 

        top_out = "# "
        for key, val in m.S.items():
            top_out = top_out + key + " "

        top_out = top_out + "VF "

        #deleting prev file and write out the header
        file_out = open(f"V_func_all_{m.epoch}_{indxd}.txt", 'w')
        file_out.write(top_out + "\n")
        file_out.close()
        file_out = open(f"V_func_all_{m.epoch}_{indxd}.txt", 'a')

        plot_pts = torch.cat(
            ((
                torch.cat(
                    (eval_samples,
                     indxd * torch.ones(
                         (eval_samples.shape[0],1))
                     ),
                    dim=1)[:,:-1]) *
                torch.tensor((1 - sigma_util)), #scale back to Fernandes Pheland's values
                torch.unsqueeze(V_pred,1)),
            dim=1)

        np.savetxt(
            file_out,
            plot_pts.numpy(),
            fmt='%.18e',
            delimiter=' ',
            newline='\n',
            header='',
            footer='',
            comments='# ',
            encoding=None)

        file_out.close()

        #deleting prev file and write out the header
        file_out = open(f"V_func_feas_{m.epoch}_{indxd}.txt", 'w')
        file_out.write(top_out + "\n")
        file_out.close()
        file_out = open(f"V_func_feas_{m.epoch}_{indxd}.txt", 'a')

        mask_feas = V_pred >= lower_V 
        feas_samples = eval_samples[mask_feas, :]
        V_feas = V_pred[mask_feas]

        plot_pts = torch.cat(
            ((
                torch.cat(
                    (feas_samples,
                     indxd * torch.ones((feas_samples.shape[0],1))),
                    dim=1)[:,:-1]) *
                torch.tensor((1 -sigma_util) ), #scale back to Fernandes Pheland's values
                torch.unsqueeze(V_feas,1)),
            dim=1)

        np.savetxt(
            file_out,
            plot_pts.numpy(),
            fmt='%.18e',
            delimiter=' ',
            newline='\n',
            header='',
            footer='',
            comments='# ',
            encoding=None)

        file_out.close()

        top_out = "# "
        for key, val in m.S.items():
            top_out = top_out + key + " "

        top_out = top_out + "VF "
        for key, val in m.P.items():
            top_out = top_out + key + " "

        #deleting prev file and write out the header
        file_out = open(f"V_comp_{m.epoch}_{indxd}.txt", 'w')
        file_out.write(top_out + "\n")
        file_out.close()
        file_out = open(f"V_comp_{m.epoch}_{indxd}.txt", 'a')

        pts_out = torch.cat(
            (
                (m.state_sample[mask, :-1]) * torch.tensor((1 - sigma_util) / (1 - beta)), #scale back to Fernandes Pheland's values
                torch.unsqueeze((m.V_sample[mask] + gp_offset), dim=-1),
                m.policy_sample[mask, :]
            ), dim=1)
        np.savetxt(
            file_out,
            pts_out.numpy(),
            fmt='%.18e',
            delimiter=' ',
            newline='\n',
            header='',
            footer='',
            comments='# ',
            encoding=None)
        
        file_out.close()

    top_out = "# "
    for key, val in m.S.items():
        top_out = top_out + key + " "

    top_out = top_out + "VF "
    for key, val in m.P.items():
        top_out = top_out + key + " "

    #deleting prev file and write out the header
    file_out = open(f"V_comp_{m.epoch}.txt", 'w')
    file_out.write(top_out + "\n")
    file_out.close()
    file_out = open(f"V_comp_{m.epoch}.txt", 'a')
    pts_out = torch.cat(
        (
            (m.state_sample[:, :-1]) * torch.tensor((1 - sigma_util) / (1 - beta)), #scale back to Fernandes Pheland's values
            torch.unsqueeze(m.state_sample[:, -1],dim=-1),
            torch.unsqueeze((m.V_sample[:] + gp_offset), dim=-1),
            m.policy_sample[:, :]
        ), dim=1)
    np.savetxt(
        file_out,
        pts_out.numpy(),
        fmt='%.18e',
        delimiter=' ',
        newline='\n',
        header='',
        footer='',
        comments='# ',
        encoding=None)
    
    file_out.close()

def compare(m1, m2, cfg):
    """Compare two GP-s"""

    n_test_pts = 1000
    state_sample_01 = (
        torch.rand(
            [n_test_pts, m1.cfg["model"]["params"]["n_types"]]
        )
    )

    lower_w = m1.cfg["model"]["params"]["lower_w"]
    upper_w = m1.cfg["model"]["params"]["upper_w"]

    error_vec = np.zeros((1, 2))
    # error_feas_vec = np.zeros((1, 2))
    for indxd in range(m1.cfg["model"]["params"]["discrete_state_dim"]):
        state_sample = ((upper_w) - (
            lower_w)) * state_sample_01 + (lower_w)
        for indx in range(state_sample.shape[0]):
            tmp_state = torch.zeros(state_sample.shape[1] + 1)
            tmp_state[:-1] = state_sample[indx, :]
            tmp_state[-1] = 1. * indxd
            state_sample[indx, :] = tmp_state[:-1]

        V1 = m1.M[indxd][0].eval()
        V2 = m2.M[indxd][0].eval()
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            V_p1 = V1(state_sample).mean
            V_p2 = V2(state_sample).mean

        error_vec += np.array([[(1 / V_p1.shape[0]**0.5 * torch.linalg.norm((V_p1 - V_p2)/(1 + torch.abs(V_p2)))).numpy(),
                                (torch.linalg.norm((V_p1 - V_p2)/(1 + torch.abs(V_p2)),ord=float('inf'))).numpy()]])
        # mask_feas = V_p2 + \
        #     m2.cfg["model"]["params"]["GP_offset"] > m2.cfg["model"]["params"]["lower_V"]
        # n_pts = torch.sum(mask_feas)
        # error_feas_vec += np.array([[(1 / n_pts**0.5 * torch.linalg.norm(V_p1[mask_feas] - V_p2[mask_feas])).numpy(),
        #                              (torch.linalg.norm(V_p1[mask_feas] - V_p2[mask_feas],
        #                             ord=float('inf'))).numpy()]])

    error_out = np.zeros((1, 4))
    error_out[0, 2:] = error_vec[0, :]/m1.cfg["model"]["params"]["discrete_state_dim"]
    error_out[0,0] = m1.epoch
    error_out[0,1] = m2.epoch
    # error_out[0, 2:] = error_feas_vec[0, :]
    with open("V_func_error.txt", 'a') as csvfile:
        np.savetxt(
            csvfile,
            error_out,
            delimiter=' ',
            newline='\n',
            header='',
            footer='',
            comments='# ',
            encoding=None)

    n_disc_states = m2.cfg["model"]["params"]["discrete_state_dim"]
    gp_offset = m2.cfg["model"]["params"]["GP_offset"]
    max_v = torch.zeros(n_disc_states)
    min_v = torch.zeros(n_disc_states)
    for indxs in range(n_disc_states):
        mask_state = m2.state_sample[:,-1] == 1.*indxs
        max_v[indxs] = torch.max(m2.V_sample[mask_state] + gp_offset)
        min_v[indxs] = torch.min(m2.V_sample[mask_state] + gp_offset)
    
    n_feas_pts = torch.sum(m2.feasible_all)
    logger.info(
        f"Difference epochs {m1.epoch} and {m2.epoch} in prediction (norm): L2: {1 / V_p1.shape[0]**0.5 * torch.linalg.norm((V_p1 - V_p2)/(1 + torch.abs(V_p2)))}; max by state {max_v} min {min_v} no feas pts {n_feas_pts}"
    )


def simulate(m, m_prev, cfg, model):

    optimize = False
    n_sim_steps = 1000
    n_agents = m.cfg["model"]["params"]["n_agents"]
    n_types = m.cfg["model"]["params"]["n_types"]
    trans_mat = m.cfg["model"]["params"]["trans_mat"]
    lower_V = m.cfg["model"]["params"]["lower_V"]
    beta = m.cfg["model"]["params"]["beta"]
    sigma_util = m.cfg["model"]["params"]["sigma"]
    gp_offset = m.cfg["model"]["params"]["GP_offset"]
    scale_pol_vec = m.cfg["model"]["params"]["scale_pol_vec"] 

    # fitting policy in case we have not done so yet
    if m.cfg.get("DISABLE_POLICY_FIT"):
        m.policy_fit(m.cfg["torch_optim"]["iter_per_cycle"])
        
    logger.info("Training of policies done now simulate")

    # setting to evaluate
    for d in range(m.discrete_state_dim):
        for p in range(m.policy_dim + 1):
            m.M[d][p].eval()

    start_pt = torch.zeros((n_agents,m.state_sample.shape[-1]))
    obj_value = -1e10*torch.ones(n_agents)
    for indxp in range(m.state_sample.shape[0]):
        indxa,indxs = m.get_disc_state(int(m.state_sample[indxp,-1].item()))
        if m.V_sample[indxp] > obj_value[indxa]:
            obj_value[indxa] = m.V_sample[indxp]
            start_pt[indxa,:] = m.state_sample[indxp,:]

    for indxa in range(n_agents):
        torch.manual_seed(1054211) #set seed for reproducible results

        # deleting previous file
        sim_out = open(f"simulation_{m.epoch}_{indxa}.txt", 'w')
        sim_out.close()

        # opening for appending
        sim_out = open(f"simulation_{m.epoch}_{indxa}.txt", 'a')
        top_out = "# "
        for key, val in m.S.items():
            top_out = top_out + key + " "

        top_out = top_out + "DS VF CB_L CB_U R_DIFF "
        for key, val in m.P.items():
            top_out = top_out + key + " "

        sim_out.write(top_out + "\n")

        current_state = torch.unsqueeze(start_pt[indxa,:],dim=0)

        dim_state = start_pt.shape[1]
        out_np = np.zeros([1, m.policy_dim + 4 + dim_state])
        abs_diff_vec = np.zeros([n_sim_steps])
        for indxt in range(n_sim_steps):

            params = m.get_params(current_state[0, :],None)

            out_np[0, 0:dim_state] = (
                current_state[0, :]).numpy()  # current state
            current_disc_state = int(current_state[0, -1].item())
            with torch.no_grad(), gpytorch.settings.fast_pred_var():
                pred = m.M[current_disc_state][0](
                    current_state[:, :-1])
                out_np[0, dim_state] = (pred.mean + gp_offset).numpy()  # value function

                lower,upper = pred.confidence_region() #confidence interval
                out_np[0, dim_state+1] = lower.numpy()
                out_np[0, dim_state+2] = upper.numpy()

                abs_diff_vec[indxt] = (
                    np.abs((m_prev.M[current_disc_state][0](
                    current_state[:, :-1]).mean + gp_offset).numpy() - out_np[0, dim_state])/
                    (1 + np.abs(out_np[0, dim_state])))  # relative abs of difference of vf in this and the prior epoch

                out_np[0, dim_state+3] = abs_diff_vec[indxt]

                pol_out = torch.zeros(m.control_dim)
                LB_pol = m.lb(current_state[0, :],params)
                UB_pol = m.ub(current_state[0, :],params)
                for indxp in range(1, m.policy_dim + 1):
                    pol_out[indxp - 1] = torch.minimum(
                        torch.tensor(UB_pol[indxp - 1]),
                        torch.maximum(
                            torch.tensor(LB_pol[indxp - 1]),
                            m.M[current_disc_state][indxp](current_state[:, :-1]).mean/scale_pol_vec[indxp-1]))[0]
                    out_np[0, dim_state + 4 + indxp - 1] = pol_out[indxp - 1].numpy()  # indxp th policy

                for indxtype in range(n_types):
                    pol_out[m.P[f"pen_{indxtype+1}"]] = 0.
                    pol_out[m.P[f"pen_u_{indxtype+1}"]] = 0.

            if optimize:

                bellman_eq_err = abs( (m.eval_f(current_state[0, :],params,pol_out)).detach().numpy() - out_np[0, dim_state])/(1 + np.abs(out_np[0, dim_state]))
                if bellman_eq_err >= 0.01:
                    try:

                        m.cfg["ipopt"]["no_restarts"] = 5
                        v, p = m.solve(current_state[0, :],pol_out)
                        out_np[0, dim_state + 4:] = p[:]
                        out_np[0, dim_state] = v.detach().numpy()  + gp_offset.detach().numpy()
                        pol_out = (p)


                    except NonConvergedError as e:
                        for indxp in range(1, m.policy_dim + 1):
                            out_np[0, dim_state + 4 + indxp - 1] = pol_out[indxp - 1].numpy()  # indxp th policy
                else:
                    for indxp in range(1, m.policy_dim + 1):
                        out_np[0, dim_state + 4 + indxp - 1] = pol_out[indxp - 1].numpy()  # indxp th policy                   
                    

            else:
                    for indxp in range(1, m.policy_dim + 1):
                        out_np[0, dim_state + 4 + indxp - 1] = pol_out[indxp - 1].numpy()  # indxp th policy

            bellman_eq_err = abs( (m.eval_f(current_state[0, :],params,pol_out)).detach().numpy() - out_np[0, dim_state])/(1 + np.abs(out_np[0, dim_state]))
            #scale back to Fernandes Pheland's values
            out_np[0, dim_state] = out_np[0, dim_state]
            out_np[0, :dim_state - 1] = out_np[0,:dim_state - 1] * (1 - sigma_util) / (1 - beta)
            np.savetxt(
                sim_out,
                out_np,
                fmt='%.18e',
                delimiter=' ',
                newline='\n',
                header='',
                footer='',
                comments='# ',
                encoding=None)

            target = m.cfg["BAL"]["targets"][0]
            bal_util = m.bal_utility_func(current_state[:,:-1],current_disc_state,0,target.get("rho"),target.get("beta"))
            logger.info(f"iteration {indxt} output {out_np[0,0:]} \nBAL: {bal_util} \nBellman error {bellman_eq_err}")
            sim_out.flush()

            cat_dist = torch.distributions.categorical.Categorical(
                trans_mat[current_disc_state,:])
            next_disc_state = int((cat_dist.sample()).item())
            current_state = torch.unsqueeze(m.state_next(
                current_state[0, :], params, pol_out, next_disc_state), 0)

        l2_err = np.linalg.norm(abs_diff_vec/n_sim_steps,ord=2)
        linf_err = np.quantile(abs_diff_vec,0.99)
        logger.info(f"Relative error along simulatin path: L2 error {l2_err} Linf error {linf_err}")
        sim_out.write(f"#Relative error along simulatin path: L2 error {l2_err} Linf error {linf_err}")
        sim_out.close()
