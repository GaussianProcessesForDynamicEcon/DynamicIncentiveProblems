import torch
import gpytorch
import numpy as np
from DPGPIpoptModel import DPGPIpoptModel,IpoptModel
from DPGPScipyModel import DPGPScipyModel
import importlib
import logging
import sys

from Utils import NonConvergedError
import os

if os.getenv("OMPI_COMM_WORLD_SIZE"):
    import torch.distributed as dist

logger = logging.getLogger(__name__)
torch.set_default_dtype(torch.float64)

# ======================================================================
#
#     sets the parameters for the model
#     "APS_nocharge"
#
#     Philipp Renner, 09/21
# ======================================================================

# ======================================================================


def utility_ind(cons, reg_c, sigma): #utility function
   return ((cons + reg_c)**(1 - sigma)/(1 - sigma) - (reg_c)**(1 - sigma)/(1 - sigma))

def inv_utility_ind(util, reg_c, sigma): #inverse utility function
   return ((1-sigma)*(util + (reg_c)**(1 - sigma)/(1 - sigma)))**(1/(1 - sigma)) - reg_c


# ======================================================================


# V infinity
def V_INFINITY(model,state,cfg):
    beta = cfg["model"]["params"]["beta"]
    shock_vec= cfg["model"]["params"]["shock_vec"]
    n_agents= cfg["model"]["params"]["n_agents"]
    ds_lst= cfg["model"]["params"]["ds_list"]
    n_types= cfg["model"]["params"]["n_types"]
    trans_mat= cfg["model"]["params"]["trans_mat"]
    gp_offset= cfg["model"]["params"]["GP_offset"]
    reg_c= cfg["model"]["params"]["reg_c"]
    sigma= cfg["model"]["params"]["sigma"]
    disc_state = int(state[-1].item())
    sum_tmp = torch.zeros(1)

    for indxt in range(n_types):
        sum_tmp += trans_mat[disc_state,indxt]*(
            shock_vec[indxt] - inv_utility_ind(state[model.S[f"w_{indxt+1}"]], reg_c, sigma))/(1-beta)

    v_infinity = -gp_offset + sum_tmp
    return v_infinity


# ======================================================================
#   Equality constraints during the VFI of the model


#def EV_G_ITER(model, state, control, cfg, S, P, overreporting):
def EV_G_ITER(model, state, params, control, cfg, S, P):

    beta = cfg["model"]["params"]["beta"]
    shock_vec= cfg["model"]["params"]["shock_vec"]
    n_agents= cfg["model"]["params"]["n_agents"]
    n_shocks= cfg["model"]["params"]["n_shocks"]
    n_types = cfg["model"]["params"]["n_types"]
    reg_c= cfg["model"]["params"]["reg_c"]
    sigma= cfg["model"]["params"]["sigma"]
    trans_mat= cfg["model"]["params"]["trans_mat"]


    M = 2*n_types + n_agents + n_agents*(n_agents-1)*3  # number of constraints

    G = torch.empty(M)

    #equality constraints
    counter = 0
    for indxs in range(n_types):
        G[counter] = utility_ind(control[P[f"c_{indxs+1}"]], reg_c, sigma) - control[P[f"u_{indxs+1}"]]
        counter += 1

    #promise keeping
    for indx in range(n_types):
        G[counter] = state[S[f"w_{indx+1}"]] + control[P[f"pen_{indx+1}"]] - control[P[f"pen_u_{indx+1}"]] - \
                sum(
                    [
                        ((1-beta)*control[P[f"u_{indxs+1}"]] + beta*(state[S[f"w_{indxs+1}"]] + control[P[f"fut_util_{indxs+1}_{indxs+1}"]]))*trans_mat[indx,indxs]  
                        for indxs in range(n_types) ])
        counter += 1


    #inequality constraints
    #incentive constraints
    for indxa in range(n_agents):
        #agent cannot report low shock if high shock
        indxsh = model.set_disc_state(indxa,1)
        indxsl = model.set_disc_state(indxa,0)
        G[counter] = (1-beta)*control[P[f"u_{indxsh+1}"]] + beta*(state[S[f"w_{indxsh+1}"]] + control[P[f"fut_util_{indxsh+1}_{indxsh+1}"]]) - \
            ( (1-beta)*utility_ind(shock_vec[indxsh] + control[P[f"c_{indxsl+1}"]] - shock_vec[indxsl], reg_c, sigma) + beta*(state[S[f"w_{indxsh+1}"]] + control[P[f"fut_util_{indxsl+1}_{indxsh+1}"]]))
        counter += 1

        #agent cannot report to be a different type of agent
        for indxa2 in range(n_agents):
            if indxa2 != indxa:
                indxa2sh = model.set_disc_state(indxa2,1)
                indxa2sl = model.set_disc_state(indxa2,0)
                G[counter] = (1-beta)*control[P[f"u_{indxsh+1}"]] + beta*(state[S[f"w_{indxsh+1}"]] + control[P[f"fut_util_{indxsh+1}_{indxsh+1}"]]) - \
                    ((1-beta)*utility_ind(shock_vec[indxsh] + control[P[f"c_{indxa2sl+1}"]] - shock_vec[indxa2sl], reg_c, sigma) + beta*(state[S[f"w_{indxsh+1}"]] + control[P[f"fut_util_{indxa2sl+1}_{indxsh+1}"]]))
                assert shock_vec[indxsh] - shock_vec[indxa2sl] >= 0, "shock difference needs to be non-negative"
                counter += 1
                G[counter] = (1-beta)*control[P[f"u_{indxsh+1}"]] + beta*(state[S[f"w_{indxsh+1}"]] + control[P[f"fut_util_{indxsh+1}_{indxsh+1}"]]) - \
                    ((1-beta)*control[P[f"u_{indxa2sh+1}"]] + beta*(state[S[f"w_{indxsh+1}"]] + control[P[f"fut_util_{indxa2sh+1}_{indxsh+1}"]]))
                counter += 1
                G[counter] = (1-beta)*control[P[f"u_{indxsl+1}"]] + beta*(state[S[f"w_{indxsl+1}"]] + control[P[f"fut_util_{indxsl+1}_{indxsl+1}"]]) - \
                    ((1-beta)*control[P[f"u_{indxa2sl+1}"]] + beta*(state[S[f"w_{indxsl+1}"]] + control[P[f"fut_util_{indxa2sl+1}_{indxsl+1}"]]))
                counter += 1

    return G


# ======================================================================
class SpecifiedModel(DPGPScipyModel):
    def __init__(self, V_guess=V_INFINITY, cfg={}, **kwargs):

        n_types = cfg["model"]["params"]["n_types"]
        n_agents = cfg["model"]["params"]["n_agents"]
        policy_names = cfg["model"]["params"]["policy_names"]

        state_names = cfg["model"]["params"]["state_names"]

        # for faster indexing
        self.fut_util_mask = torch.tensor(
            [x.startswith("fut_util_") for x in policy_names]
        )
        self.pen_mask = torch.tensor(
            [x.startswith("pen_") for x in policy_names]
        )

        super().__init__(
            V_guess=lambda state: V_INFINITY(
                self,
                state,
                self.cfg
            ),
            cfg=cfg,
            policy_names=policy_names,
            state_names=state_names,
            policy_dim = cfg["model"]["params"]["policy_dim"],
            discrete_state_dim = cfg["model"]["params"].get("discrete_state_dim", 1),
            control_dim= cfg["model"]["params"]["control_dim"],
            **kwargs
        )


    def get_disc_state(self,disc_state):
        return self.cfg["model"]["params"]["ds_list"][int(disc_state)]

    def set_disc_state(self,agent,state):
        for indxs in range(len(self.cfg["model"]["params"]["ds_list"])):
            if self.cfg["model"]["params"]["ds_list"][indxs][0] == int(agent) and self.cfg["model"]["params"]["ds_list"][indxs][1] == int(state):
                return indxs

        return None


    def sample(self,no_samples=None):
        if no_samples is None:
            no_samples = self.cfg["no_samples"]

        n_types = self.cfg["model"]["params"]["n_types"]
        n_agents = self.cfg["model"]["params"]["n_agents"]
        n_shocks = self.cfg["model"]["params"]["n_shocks"]
        shock_vec= self.cfg["model"]["params"]["shock_vec"]
        dim_size = n_types
        assert no_samples % self.discrete_state_dim == 0, "no. of samples should be divisble by no of discrete states because of uniform distribution of samples"
        self.state_sample = (
            #torch.tile(
                torch.rand(
                    [int(no_samples/self.discrete_state_dim), dim_size]
                )#,[1,n_agents])
        )

        #corner points
        # self.state_sample = torch.cat(
        #     (
        #         self.state_sample,
        #         torch.tensor([
        #             [0., 0.,0., 0.],
        #             [1., 1.,1., 1.],
        #             [0.5, 0.5,0.5, 0.5]])
        #     ),
        #     dim=0,
        # )
        # for indxdim in range(1,dim_size):
        #     indx_mat = torch.combinations(torch.arange(dim_size),r=indxdim)
        #     no_pts = indx_mat.shape[0]
        #     for indxr in range(no_pts):
        #         corner_vec = torch.zeros([1,dim_size])
        #         for indx in range(indxdim):
        #             corner_vec[0,indx] = 1.0

        #         self.state_sample = torch.cat(
        #             (
        #                 self.state_sample,
        #                 corner_vec
        #             ),
        #             dim=0,
        #         )
                
        # mask_feas = torch.ones(self.state_sample.shape[0], dtype=torch.bool)
        # for indxa in range(n_agents):
        #     indxh = self.set_disc_state(indxa,1)
        #     indxl = self.set_disc_state(indxa,0)
        #     assert shock_vec[indxh] - shock_vec[indxl] >= 0, "shock difference needs to be non-negative"
        #     mask_feas = torch.logical_and(mask_feas,self.state_sample[:,indxh] + 0.1 >= self.state_sample[:,indxl]) #only upper traingle is feasible
        #     for indxa2 in range(n_agents):
        #         if indxa != indxa:
        #             indxa2h = self.set_disc_state(indxa2,1)
        #             indxa2l = self.set_disc_state(indxa2,0)
        #             assert shock_vec[indxa2h] - shock_vec[indxl] >= 0, "shock difference needs to be non-negative"
        #             mask_feas = torch.logical_and(mask_feas,self.state_sample[:,indxh] + 0.1 >= self.state_sample[:,indxa2l]) #only upper traingle is feasible

        # self.state_sample = self.state_sample[mask_feas,:]
        # if self.state_sample.shape[0] > int(no_samples/self.discrete_state_dim):
        #     self.state_sample = self.state_sample[:int(no_samples/self.discrete_state_dim),:]
        # else:
        #     assert False, "Not enough sample points"



        no_samples = self.state_sample.shape[0]

        self.state_sample = torch.cat(
            (
                torch.repeat_interleave(self.state_sample,self.discrete_state_dim,dim=0),
                torch.tile(torch.unsqueeze(torch.arange(0.,self.discrete_state_dim),dim=1),[no_samples,1]),
            ),
            dim=1,
        )

        no_samples = self.state_sample.shape[0]

        #compute the state bounds
        lower_w = self.cfg["model"]["params"]["lower_w"]
        upper_w = self.cfg["model"]["params"]["upper_w"]
        LB_state = torch.zeros(dim_size)
        UB_state = torch.zeros(dim_size)

        for indxt in range(n_types):
            LB_state[self.S[f"w_{indxt+1}"]] = lower_w[indxt]
            UB_state[self.S[f"w_{indxt+1}"]] = upper_w[indxt]

        self.state_sample[:,:-1] = torch.unsqueeze(UB_state - LB_state,dim=0)*self.state_sample[:,:-1] + torch.unsqueeze(LB_state,dim=0)

        #add in a guess of an optimal point
        # self.state_sample = torch.cat(
        #     (
        #         self.state_sample,
        #         torch.tensor([
        #             [4.9176, 7.3888,3.5366, 6.9523, 0.0],
        #             [3.9846, 5.4949,3.2650, 4.9301, 1.0],
        #             [4.9176, 7.3888,3.5366, 6.9523, 2.0],
        #             [3.9846, 5.4949,3.2650, 4.9301, 3.0]])
        #     ),
        #     dim=0,
        # )

        self.feasible = torch.ones(self.state_sample.shape[0])
        self.combined_sample = torch.zeros([self.state_sample.shape[0], 1+self.policy_dim])

    @torch.no_grad()
    def sample_start_pts(self, state, params, policy, n_restarts):
        S = self.S
        n_agents = self.cfg["model"]["params"]["n_agents"]
        beta = self.cfg["model"]["params"]["beta"]
        reg_c = self.cfg["model"]["params"]["reg_c"]
        sigma = self.cfg["model"]["params"]["sigma"]
        n_types = self.cfg["model"]["params"]["n_types"]
        shock_vec = self.cfg["model"]["params"]["shock_vec"]
        trans_mat_ag = self.cfg["model"]["params"]["agents_trans_mat"]
        scale_pol_vec = self.cfg["model"]["params"]["scale_pol_vec"] 

        disc_state = int(state[-1].item())

        inv_trans_mat = self.cfg["model"]["params"]["trans_mat_inv"]

        LB = torch.from_numpy(self.lb(state, params))
        UB = torch.from_numpy(self.ub(state, params))
        n_pts = 1*n_restarts
        policy_sample = (
            torch.rand(
                [n_pts, self.control_dim]
            )
            * (
                UB - LB
            )
            + LB
        )

        # val_lst = torch.zeros(n_pts)
        for indxp in range(n_pts):
            control = torch.zeros(policy_sample.shape[-1])
            control[:] = policy_sample[indxp,:]
            for indxt in range(n_types):
                control[self.P[f"pen_{indxt+1}"]] = 0.
                control[self.P[f"pen_u_{indxt+1}"]] = 0.

            for indxr in range(n_types):

                control[self.P[f"c_{indxr+1}"]] = inv_utility_ind(control[self.P[f"u_{indxr+1}"]] ,reg_c,sigma)

                # control[self.P[f"fut_util_{indxr+1}_{indxr+1}"]] = -(1 - beta) * control[self.P[f"u_{indxr+1}"]]/beta - state[S[f"w_{indxr+1}"]]
                # for indxc in range(n_types):
                #     control[self.P[f"fut_util_{indxr+1}_{indxr+1}"]] += inv_trans_mat[indxr,indxc]*state[self.S[f"w_{indxc+1}"]]/beta

            policy_sample[indxp,:] = control[:]
            # val_lst[indxp] = self.eval_f(state,params,control)
        
        # indx_lst = torch.argsort(val_lst, descending=True)
        policy_sample_out = policy_sample#[indx_lst[:n_restarts],:]
        policy_sample_out[-1,:] = policy

        if not self.cfg.get("DISABLE_POLICY_FIT"):
            pol_inter = torch.zeros(policy_sample.shape[-1])
            eval_pt = torch.unsqueeze(state[:-1],dim=0)
            for indxp in range(policy_sample.shape[-1]):
                try:
                    pol_inter[indxp] = torch.minimum(
                        UB[indxp],
                        torch.maximum(
                            LB[indxp],
                            self.M[disc_state][1 + indxp](eval_pt).mean/scale_pol_vec[indxp]
                        ),
                    )
                    policy_sample_out[-2,indxp] = pol_inter[indxp]
                except:
                    logger.info(f"state {disc_state} policy {indxp} evaluation failed in sample_stps.")

        return policy_sample_out

    @torch.no_grad()
    def get_params(self,state,policy):

        lower_V = self.cfg["model"]["params"]["lower_V"]
        gp_offset = self.cfg["model"]["params"]["GP_offset"]
        n_types = self.cfg["model"]["params"]["n_types"]
        params = torch.zeros(2 + n_types)
        current_v = gp_offset + (self.M[int(state[-1].item())][0](
            torch.unsqueeze(state[:-1],dim=0)).mean)
        params[1] = current_v[0]
        if current_v < lower_V:
            params[0] = 0.   #infeasible point
        else:
            params[0] = 1.   # possible feasible point

        for indxt in range(n_types):
            state_sample = self.state_sample_all[:self.V_sample_all.shape[0],:]
            mask_type = state_sample[:,-1] == 1.*indxt
            assert mask_type.shape[0] > 0, "empty number of samples"
            params[2+indxt] = torch.max(self.V_sample_all[mask_type] + gp_offset) #store largest function value

        return params

    def is_feasible(self,state,value,control):
        error_tol = 1.0e-3
        n_types = self.cfg["model"]["params"]["n_types"]
        pen_opt_vf = self.cfg["model"]["params"]["pen_opt_vf"]
        total_pen = sum([ control[self.P[f"pen_{indxt}"]]*pen_opt_vf for indxt in range(1,n_types+1)]) + \
            sum([ control[self.P[f"pen_u_{indxt}"]]*pen_opt_vf for indxt in range(1,n_types+1)])
        gp_offset = self.cfg["model"]["params"]["GP_offset"]
        lower_V = self.cfg["model"]["params"]["lower_V"]
        if value  <= 0.0:
        # if total_pen >= error_tol and value <= 0.0:            
            return 0.0
        else:
            return 1.0

    def post_process_optimization(self, state, params, control, value):
        beta = self.cfg["model"]["params"]["beta"]
        error_tol = 1.0e-3
        n_agents = self.cfg["model"]["params"]["n_agents"]
        n_types = self.cfg["model"]["params"]["n_types"]
        GP_min = self.cfg["model"]["params"]["GP_min"]
        pen_opt_vf = self.cfg["model"]["params"]["pen_opt_vf"]
        pen_vf = self.cfg["model"]["params"]["pen_vf"]
        gp_offset = self.cfg["model"]["params"]["GP_offset"]
        upper_V = self.cfg["model"]["params"]["upper_V"]
        lower_V = self.cfg["model"]["params"]["lower_V"]
        lower_V_vec = self.cfg["model"]["params"]["lower_V_vec"]
        pen_opt_barrier = self.cfg["model"]["params"]["pen_opt_barrier"]
        pen_barrier = self.cfg["model"]["params"]["pen_barrier"]
        scale_pol_vec = self.cfg["model"]["params"]["scale_pol_vec"] 

        is_feas = self.is_feasible(state,value - gp_offset,control)
        disc_state_in = int(state[-1].item())

        tmp_sum = value
        for indxa in range(n_types):
            tmp_sum += control[self.P[f"pen_{indxa+1}"]]*pen_opt_vf + control[self.P[f"pen_u_{indxa+1}"]]*pen_opt_vf

        total_pen = sum([ control[self.P[f"pen_{indxt}"]]*pen_opt_vf for indxt in range(1,n_types+1)]) + \
            sum([ control[self.P[f"pen_u_{indxt}"]]*pen_opt_vf for indxt in range(1,n_types+1)])
        if total_pen >= error_tol:
            for indxa in range(n_types):
                tmp_sum +=  - control[self.P[f"pen_{indxa+1}"]]*(pen_vf) - control[self.P[f"pen_u_{indxa+1}"]]*(pen_vf)

        weights, points = self.state_iterate_exp(state, params, control)
        for i in range(len(weights)):
            p_i = torch.unsqueeze(points[i, :-1], 0)
            obj_val = (self.M[int(points[i, -1].item())][0](p_i).mean + gp_offset)

            # obj_val_capped = torch.minimum(params[2 + i],obj_val)
            # tmp_sum += beta*(obj_val_capped[0] - obj_val[0]) * weights[i]
            if obj_val < lower_V_vec[disc_state_in] and params[0] == 1.0:
                tmp_sum +=  beta*(lower_V_vec[disc_state_in] - obj_val[0])**2*pen_opt_barrier - beta*(lower_V_vec[disc_state_in] - obj_val[0])**2*pen_barrier

        control = scale_pol_vec * control
        return control, torch.minimum(
            upper_V - gp_offset,
            torch.maximum(
                GP_min - gp_offset,
                tmp_sum  - gp_offset))


    def u(self, state, params, control):
        n_agents = self.cfg["model"]["params"]["n_agents"]
        n_shocks = self.cfg["model"]["params"]["n_shocks"]
        n_types = self.cfg["model"]["params"]["n_types"]
        disc_state_in = int(state[-1].item())
        trans_mat = self.cfg["model"]["params"]["trans_mat"]
        shock_vec = self.cfg["model"]["params"]["shock_vec"]
        GP_offset = self.cfg["model"]["params"]["GP_offset"]
        ds_list = self.cfg["model"]["params"]["ds_list"]
        beta = self.cfg["model"]["params"]["beta"]
        pen_opt_vf = self.cfg["model"]["params"]["pen_opt_vf"]

        total = torch.tensor(0.)   #lowest value will be greater than zero, useful for GP approximation of VF
        for indxt in range(n_types):
            total += trans_mat[disc_state_in,indxt]*(shock_vec[indxt] - control[self.P[f"c_{indxt+1}"]]) - control[self.P[f"pen_{indxt+1}"]]*pen_opt_vf - control[self.P[f"pen_u_{indxt+1}"]]*pen_opt_vf
        
        return total



    def E_V(self, state, params, control):
        """Caclulate the expectation of V"""
        # if not VFI, then return a differentiable zero
        if self.cfg["model"].get("ONLY_POLICY_ITER"):
            return torch.sum(control) * torch.zeros(1)

        e_v_next = 0
        gp_offset = self.cfg["model"]["params"]["GP_offset"]
        lower_V = self.cfg["model"]["params"]["lower_V"]
        lower_V_vec = self.cfg["model"]["params"]["lower_V_vec"]
        pen_opt_vf = self.cfg["model"]["params"]["pen_opt_vf"]
        pen_opt_barrier = self.cfg["model"]["params"]["pen_opt_barrier"]

        disc_state_in = int(state[-1].item())

        weights, points = self.state_iterate_exp(state, params, control)
        for i in range(len(weights)):
            p_i = torch.unsqueeze(points[i, :-1], 0)
            # obj_val = torch.minimum(params[2 + i],self.M[int(points[i, -1].item())][0](p_i).mean + gp_offset)
            obj_val = (self.M[int(points[i, -1].item())][0](p_i).mean + gp_offset)
            e_v_next += (obj_val)  * weights[i]
            if obj_val < lower_V_vec[disc_state_in] and params[0] == 1.0:
                e_v_next += - (lower_V_vec[disc_state_in] - obj_val)**2*pen_opt_barrier
                
        return e_v_next


    def state_next(self, state, params, control, zpy):
        """Return next periods states, given the controls of today and the random discrete realization"""
        
        S = self.S

        s = state.clone()
        
        # update discrete state
        s[-1] = 1.*zpy
        
        n_types = self.cfg["model"]["params"]["n_types"]
        n_agents = self.cfg["model"]["params"]["n_agents"]

        for indx in range(n_types):
            s[S[f"w_{indx+1}"]] = state[self.S[f"w_{indx+1}"]] + control[self.P[f"fut_util_{int(zpy)+1}_{indx+1}"]]

        return s
 
    def state_iterate_exp(self, state, params, control):
        """How are future states generated from today state and control"""
        
        n_types = self.cfg["model"]["params"]["n_types"]
        disc_state = int(state[-1].item())
        trans_mat = self.cfg["model"]["params"]["trans_mat"]
        weights = torch.tensor([trans_mat[disc_state,indx]  for indx in range(n_types)])
        
        points = torch.cat(
            tuple(
                torch.unsqueeze(self.state_next(state, params, control, z), dim=0)
                for z in range(n_types)
            ),
            dim=0,
        )

        return weights, points




    def lb(self, state, params):
        S = self.S
        n_agents = self.cfg["model"]["params"]["n_agents"]
        n_types = self.cfg["model"]["params"]["n_types"]
        lowerb = self.cfg["model"]["params"]["lowerb"]
        beta = self.cfg["model"]["params"]["beta"]
        lower_w = self.cfg["model"]["params"]["lower_w"]
        upper_w = self.cfg["model"]["params"]["upper_w"]
        shock_val= self.cfg["model"]["params"]["shock_vec"]
        reg_c = self.cfg["model"]["params"]["reg_c"]
        sigma= self.cfg["model"]["params"]["sigma"]

        X_L = np.zeros(self.control_dim)
        for indxa in range(n_types):
            for indxa2 in range(n_types):
                X_L[self.P[f"fut_util_{indxa+1}_{indxa2+1}"]] = lower_w[indxa2] - state[self.S[f"w_{indxa2+1}"]]


        return X_L

    def ub(self, state, params):
        S = self.S
        n_agents = self.cfg["model"]["params"]["n_agents"]
        n_types = self.cfg["model"]["params"]["n_types"]
        upperb = self.cfg["model"]["params"]["upperb"]
        beta = self.cfg["model"]["params"]["beta"]
        upper_w = self.cfg["model"]["params"]["upper_w"]
        shock_val= self.cfg["model"]["params"]["shock_vec"]
        reg_c = self.cfg["model"]["params"]["reg_c"]
        sigma= self.cfg["model"]["params"]["sigma"]

        X_U = np.empty(self.control_dim)
        for indxa in range(n_types):
            X_U[self.P[f"u_{indxa+1}"]] = utility_ind(upperb, reg_c, sigma)
            X_U[self.P[f"pen_{indxa+1}"]] = utility_ind(upperb, reg_c, sigma)#/(1-beta)
            X_U[self.P[f"pen_u_{indxa+1}"]] = utility_ind(upperb, reg_c, sigma)#/(1-beta)
            X_U[self.P[f"c_{indxa+1}"]] = upperb
            for indxa2 in range(n_types):
                X_U[self.P[f"fut_util_{indxa+1}_{indxa2+1}"]] = upper_w[indxa2] - state[self.S[f"w_{indxa2+1}"]]



        return X_U

    def cl(self, state, params):
        n_agents = self.cfg["model"]["params"]["n_agents"]
        n_types = self.cfg["model"]["params"]["n_types"]        
        M =  2*n_types + n_agents + n_agents*(n_agents-1)*3  # number of constraints
        G_L = np.empty(M)
        G_L[:] =  0.0
        return G_L

    def cu(self, state, params):
        n_agents = self.cfg["model"]["params"]["n_agents"]
        n_types = self.cfg["model"]["params"]["n_types"]
        M =  2*n_types + n_agents + n_agents*(n_agents-1)*3  # number of constraints
        G_U = np.empty(M)
        # Set bounds for the constraints
        G_U[:] = 0.0
        G_U[2*n_types:] = 1.0e10

        return G_U

    def eval_g(self, state, params, control):

        return EV_G_ITER(
            self,
            state,
            params,
            control,
            self.cfg,
            self.S,
            self.P, #overreporting
        )

    def scaling_vector(self):
        n_types = self.cfg["model"]["params"]["n_types"]
        beta = self.cfg["model"]["params"]["beta"]
        scale_vec = torch.ones(self.control_dim)
        for indxr in range(n_types):
            for indxc in range(n_types):
                scale_vec[self.P[f"fut_util_{indxr+1}_{indxc+1}"]] = 1

        return scale_vec

##############################################################################################################################
###########              Functions used for Bayesian active learning                                            ##############
##############################################################################################################################

    def bal_utility_func(self,eval_pt,discrete_state,target_p,rho,beta,pen_val=torch.tensor([-1.0e10])):
        lower_V = (self.cfg["model"]["params"]["lower_V"])
        gp_offset = (self.cfg["model"]["params"]["GP_offset"])

        self.M[discrete_state][target_p].eval()
        self.likelihood[discrete_state][target_p].eval()
        pred = self.M[discrete_state][target_p](
                        eval_pt
                    )

        var_v = pred.variance
        mean_v = pred.mean

        if var_v < 0.1:
            return pen_val
        else:
            return rho * (mean_v + gp_offset  ) + beta / 2.0 * torch.log(var_v + 1e-10)


    def BAL(self):
        new_sample = torch.empty((0,self.state_sample_all.shape[1]))
        for target in self.cfg["BAL"]["targets"]:
            # calculate BAL utility for each of the new sample
            bal_utility = -1.0e10*torch.ones(self.state_sample.shape[0])
            non_empty_vec = True

            target_d = target.get("discrete_state", -1)
            d_range = [target_d] if target_d != -1 else range(self.discrete_state_dim)
            target_p = target.get("policy")

            for discrete_state in d_range:
                self.M[discrete_state][target_p].eval()
                self.likelihood[discrete_state][target_p].eval()

            for s_ind in range(self.state_sample.shape[0]):
                    discrete_state = int(self.state_sample[s_ind, -1].item())
                    if discrete_state in d_range:
                        non_empty_vec = True
                        eval_pt = self.state_sample.to(self.device)[s_ind : (s_ind + 1), :-1]
                        bal_utility[s_ind] = self.bal_utility_func(eval_pt,discrete_state,target_p,target.get("rho"),target.get("beta"))

            if non_empty_vec:
                new_sample = torch.cat(
                    (
                        new_sample,
                        torch.index_select(
                            self.state_sample,
                            0,
                            torch.argsort(bal_utility, descending=True)[
                                : self.cfg["BAL"].get("points_per_iter")
                            ],
                        ),
                    ),
                    dim=0,
                )

        self.state_sample = torch.cat(
            (
                    self.prev_state_sample,
                    new_sample,
            ),
            dim=0,
        )
        self.feasible = torch.cat(
            (
                    self.prev_feasible,
                    torch.ones(new_sample.shape[0]),
            ),
            dim=0,
        )
        self.combined_sample = torch.cat(
            (
                    self.prev_combined_sample,
                    torch.zeros([new_sample.shape[0],1+self.policy_dim]),
            ),
            dim=0,
        )


    def sample_all(self, init_sample=None):
        if init_sample is None:
            if self.epoch == 0:
                self.sample()
            else:
                self.prev_state_sample = self.state_sample_all.clone()
                self.prev_combined_sample = self.combined_sample_all.clone()
                self.prev_feasible = self.feasible_all.clone()
                if self.cfg.get("resample_method") == "random":
                    self.sample(self.cfg.get("resample_num_new",1000))
                elif self.cfg.get("resample_method") == "dynamic":
                    # Pull a new state accoriding to weights randomly
                    # Use method: https://discuss.pytorch.org/t/torch-equivalent-of-numpy-random-choice/16146/14
                    # sample next state
                    if self.cfg.get("distributed") and dist.get_rank() == 0:
                        new_state_sample = torch.tensor([])
                        for s_ind in range(self.state_sample_all.shape[0]):
                            if self.prev_feasible[s_ind] == 1:
                                params = self.get_params(
                                    self.state_sample_all[s_ind],
                                    self.policy_sample_all[s_ind][: self.policy_dim]
                                    )
                                weights, points = self.state_iterate_exp(
                                    self.state_sample_all[s_ind],
                                    params,
                                    self.policy_sample_all[s_ind][: self.policy_dim],
                                )
                                if self.cfg.get("resample_num_new") < 0:
                                    new_state_sample = torch.cat(
                                        (new_state_sample, points), dim=0
                                    )
                                elif (
                                    self.cfg.get("resample_num_new") <= weights.shape[0]
                                ):
                                    sample_idx = weights.multinomial(
                                        num_samples=self.cfg.get("resample_num_new"),
                                        replacement=False,
                                    )
                                    new_state_sample = torch.cat(
                                        (new_state_sample, points[sample_idx]), dim=0
                                    )
                                else:
                                    sys.stderr.write(
                                        f"Config value of resample_num_new must not exceed the number of possible future states ({int(weights.shape[0])})"
                                    )
                                    sys.exit(1)

                        self.state_sample = new_state_sample
                elif self.cfg.get("resample_method") == "disabled":
                    self.state_sample = self.state_sample_all
                    self.feasible = self.feasible_all
                    self.combined_sample = self.combined_sample_all
                else:
                    sys.stderr.write(
                        f'Unrecognised config value for resample_method: {self.cfg.get("resample_method")}'
                    )
                    sys.exit(1)
                # do Bayesian Active Learning
                if self.cfg["BAL"].get("enabled"):
                    if (
                        self.epoch % self.cfg["BAL"].get("epoch_freq", 5) == 0
                        and (
                            self.cfg["BAL"]["max_points"] < 0
                            or self.cfg["BAL"]["max_points"]
                            > self.prev_state_sample.shape[0]
                        )
                    ) and self.epoch > 1:
                        if self.cfg.get("DISABLE_POLICY_FIT"):
                            self.policy_fit_trans()
                        # if self.cfg.get("distributed") and dist.get_rank() != 0:
                        #     logger.debug(
                        #         "Skipping Bayesian-Active learning sampling..."
                        #     )
                        # else:
                        #     logger.info(
                        #         "Starting Bayesian-Active learning sampling..."
                        #     )
                        #     self.BAL()
                        self.simulate()
                    else:
                        # if not doing BAL accumulation, then retain sample
                        self.state_sample = self.prev_state_sample
                        self.combined_sample = self.prev_combined_sample
                        self.feasible = self.prev_feasible
                else:
                    self.feasible = self.prev_feasible.to(self.device)
                    self.state_sample = self.prev_state_sample.to(self.device)
                    self.combined_sample = self.prev_combined_sample.to(self.device)

        else:
            self.state_sample = init_sample.to(self.device)
            self.feasible = torch.ones(init_sample.shape[0]).to(self.device)
            self.combined_sample = torch.empty([self.state_sample.shape[0],1 + self.policy_dim])

        # the sampling is always for the complete population
        self.state_sample_all = self.state_sample.to(self.device)

        self.combined_sample_all = self.combined_sample.to(self.device)

        if self.epoch % 20 == 0:
            tmp = self.feasible.to(self.device)
            tmp[:] = 1.
            self.feasible_all = tmp
        else:
            self.feasible_all = self.feasible.to(self.device)

        # non-convered points
        self.non_converged_all = torch.zeros(self.state_sample.shape[0]).to(self.device)

        # distribute the samples
        self.scatter_sample()

    def policy_fit_trans(self):
        S = self.S
        P = self.P
        n_types = self.cfg["model"]["params"]["n_types"]
        self.fit_GP(
            self.cfg["torch_optim"]["iter_per_cycle"],
            [
                (d, 1+P[f"fut_util_{p1}_{p2}"])
                for p1 in range(1, n_types + 1)
                for p2 in range(1, n_types + 1)
                for d in range(self.discrete_state_dim)
            ],
        )
        self.fit_GP(
            self.cfg["torch_optim"]["iter_per_cycle"],
            [
                (d, 1+P[f"c_{p}"])
                for p in range(1, n_types + 1)
                for d in range(self.discrete_state_dim)
            ],
        )

    def simulate(self):

        n_sim_steps = 1000
        P = self.P
        n_types = self.cfg["model"]["params"]["n_types"]
        trans_mat = self.cfg["model"]["params"]["trans_mat"]
        lower_V = self.cfg["model"]["params"]["lower_V"]
        lower_V_vec = self.cfg["model"]["params"]["lower_V_vec"]
        target = self.cfg["BAL"]["targets"][0]
        gp_offset = self.cfg["model"]["params"]["GP_offset"]
        dim_state = self.state_sample_all.shape[1]
        beta = self.cfg["model"]["params"]["beta"]
        scale_pol_vec = self.cfg["model"]["params"]["scale_pol_vec"] 

        #compute the state bounds
        lower_w = self.cfg["model"]["params"]["lower_w"]
        upper_w = self.cfg["model"]["params"]["upper_w"]
        LB_state = torch.zeros(dim_state - 1)
        UB_state = torch.zeros(dim_state - 1)

        for indxt in range(n_types):
            LB_state[self.S[f"w_{indxt+1}"]] = lower_w[indxt]
            UB_state[self.S[f"w_{indxt+1}"]] = upper_w[indxt]


        #new bal points are stored here with their bal util
        out_tmp = torch.zeros([n_types,1 + dim_state])
        pen_val = -1.0e10
        out_tmp[:,-1] = pen_val

        #setting to evaluate
        for d in range(self.discrete_state_dim):
            self.M[d][0].eval()
            for p1 in range(1,n_types+1):
                self.M[d][1+P[f"c_{p1}"]].eval()
                for p2 in range(1,n_types+1):
                    self.M[d][1+P[f"fut_util_{p1}_{p2}"]].eval()


        #pick point with maximal vf value to start sim from
        start_pt = torch.zeros([2*n_types,dim_state])
        for indxd in range(n_types):

            mask = self.state_sample_all[:, -1] == indxd * torch.tensor(1.)
            V_sample = self.V_sample_all[mask]
            max_ind_opt = torch.argmax(V_sample)

            sample_tmp = self.state_sample_all[mask,:]
            try:
                obj,point = self.find_max_of_vf(indxd, sample_tmp[max_ind_opt,:-1],LB_state,UB_state)            
                test_pt = torch.unsqueeze(torch.from_numpy(point),dim=0)
                bal_util = self.bal_utility_func(test_pt,indxd,0,target.get("rho"),target.get("beta"), pen_val=torch.tensor([pen_val]))
                out_tmp[indxd,-1] = bal_util[0]
                out_tmp[indxd,:(dim_state-1)] = test_pt[0,:]
                out_tmp[indxd,-2] = 1.*indxd
            except:
                logger.info(f"failed to converge when finding max gp val in state {indxd}")
                obj = -1e10
                point = torch.empty(1)

            start_pt[indxd,:-1] = sample_tmp[max_ind_opt,:-1]


            start_pt[indxd,-1] = indxd
            start_pt[n_types + indxd,:] = start_pt[indxd,:]

        if not self.cfg.get("distributed") or dist.get_rank() == 0:
            logger.info(f"Starting BAL simulation at points {start_pt}")

        tasks_per_worker = 2*n_types
        if self.cfg.get("distributed"):
            torch.manual_seed(1211 + dist.get_rank()*123 + self.epoch*3654)
            # allocate fitting across workers
            tasks_per_worker = n_types / dist.get_world_size()
            worker_slice = [
                A
                for A in range(2*n_types)
                if int(A / tasks_per_worker) == dist.get_rank()
            ]
        else:
            torch.manual_seed(1054211+ self.epoch*3654)
            worker_slice = list(range(tasks_per_worker))

        for indx_ in worker_slice: #start a simulation for each types's maximum starting point
            indx_type = (indx_%int(n_types))
            current_state = torch.unsqueeze(start_pt[indx_type,:],dim=0)

            for indxt in range(n_sim_steps):
                if indx_ < n_types:
                    current_state[0,-1] = 1.*indx_type

                current_disc_state = int(current_state[0,-1].item())

                #compute state transition
                pol_out = torch.zeros(self.control_dim)
                params = self.get_params(current_state[0,:],None)
                LB_pol = self.lb(current_state[0,:],params)
                UB_pol = self.ub(current_state[0,:],params)
                try:
                    for p1 in range(1,n_types+1):
                        with torch.no_grad():
                            next_pol_tmp = self.M[current_disc_state][1+P[f"c_{p1}"]](current_state[:,:-1]).mean/scale_pol_vec[P[f"c_{p1}"]]

                        pol_out[P[f"c_{p1}"]] = torch.minimum(
                            torch.tensor(UB_pol[P[f"c_{p1}"]]),
                            torch.maximum(
                                torch.tensor(LB_pol[P[f"c_{p1}"]]),
                                next_pol_tmp))[0]

                        for p2 in range(1,n_types+1):
                            with torch.no_grad():
                                next_pol_tmp = self.M[current_disc_state][1+P[f"fut_util_{p1}_{p2}"]](current_state[:,:-1]).mean/scale_pol_vec[P[f"fut_util_{p1}_{p2}"]]

                            pol_out[P[f"fut_util_{p1}_{p2}"]] = torch.minimum(
                                torch.tensor(UB_pol[P[f"fut_util_{p1}_{p2}"]]),
                                torch.maximum(
                                    torch.tensor(LB_pol[P[f"fut_util_{p1}_{p2}"]]),
                                    next_pol_tmp))[0]
                except:
                    logger.info(f"Evaluation failed in state {current_disc_state}; aborting simulation.")
                    break

                #compare bal utility
                with torch.no_grad():
                    obj_val = self.eval_f(current_state[0, :],params,pol_out)
                    bal_util = self.bal_utility_func(current_state[:,:-1],current_disc_state,0,target.get("rho"),target.get("beta"), pen_val=torch.tensor([pen_val]))
                    v = self.M[current_disc_state][0](current_state[:,:-1]).mean

                if v + gp_offset > 1.1*params[2 + current_disc_state] or v + gp_offset < lower_V_vec[current_disc_state]: #if at any point we exceed the max value of all interpolpts then prioritze them when adding pts
                     bal_util += 100.

                if bal_util > out_tmp[current_disc_state,-1]:
                    out_tmp[current_disc_state,:-1] = current_state[0,:]
                    out_tmp[current_disc_state,-1] = bal_util[0]

                #check of we need to abort sim because we walked somewhere nonsensical
                if v + gp_offset < lower_V_vec[current_disc_state]:
                    break

                cat_dist = torch.distributions.categorical.Categorical(trans_mat[current_disc_state,:])
                next_disc_state = int((cat_dist.sample()).item())
                current_state = torch.unsqueeze(self.state_next(current_state[0,:], params, pol_out, next_disc_state),0)

        #gather simulation results on rank 0
        if self.cfg.get("distributed"):
            cand_pts_gather = (
                torch.cat(self.gather_tensors(out_tmp))
                .clone()
                .detach()
                .to(self.device)
            )
        else:
            cand_pts_gather = out_tmp

        #we only compute and return the results for rank 0
        if not self.cfg.get("distributed") or dist.get_rank() == 0:
            cand_pts = torch.zeros([n_types,1 + dim_state])
            cand_pts[:,-1] = pen_val
            for indxt in range(n_types):
                for indxp in range(cand_pts_gather.shape[0]):
                    if cand_pts_gather[indxp,-2].item() == 1.*indxt:
                        if cand_pts_gather[indxp,-1] > cand_pts[indxt,-1]:
                            cand_pts[indxt,:] = cand_pts_gather[indxp,:]

            n_pts = 0
            for indxd in range(self.cfg["model"]["params"]["discrete_state_dim"]):
                if cand_pts[indxd,-1] > pen_val:
                    n_pts += 1

            out = torch.zeros([n_pts,dim_state])
            indxp = 0
            for indxd in range(self.cfg["model"]["params"]["discrete_state_dim"]):
                if cand_pts[indxd,-1] > pen_val:
                    out[indxp,:] = cand_pts[indxd,:-1]
                    indxp+=1

            logger.info(f"BAL added points {out} after iteration {indxt}")
            new_sample = out
            self.state_sample = torch.cat(
                (
                        self.prev_state_sample,
                        new_sample,
                ),
                dim=0,
            )
            self.feasible = torch.cat(
                (
                        self.prev_feasible,
                        torch.ones(new_sample.shape[0]),
                ),
                dim=0,
            )
            self.combined_sample = torch.cat(
                (
                        self.prev_combined_sample,
                        torch.zeros([new_sample.shape[0],1+self.policy_dim]),
                ),
                dim=0,
            )

    def create_model(self, d, p, train_x, train_y):
        if self.cfg.get('use_fixed_noise',True):
            # if p == 0:
            #     noise_vec = torch.zeros(train_y.shape[0])
            #     feas_mask = train_y[:] > 0.
            #     noise_vec[feas_mask] = self.cfg["gpytorch"].get("likelihood_noise_feas", 1e-6)
            #     infeas_mask = train_y[:] == False
            #     noise_vec[infeas_mask] = self.cfg["gpytorch"].get("likelihood_noise_infeas", 1.0)
            # else:
            noise_vec = torch.ones(train_y.shape[0])*self.cfg["gpytorch"].get("likelihood_noise_feas", 1e-6)

            self.likelihood[d][p] = gpytorch.likelihoods.FixedNoiseGaussianLikelihood(
                        noise_vec,
                        learn_additional_noise=False
                    ).to(self.device)

        else:
            self.likelihood[d][p] = gpytorch.likelihoods.GaussianLikelihood(
                        noise_constraint=gpytorch.constraints.GreaterThan(1e-7)
                    ).to(self.device)

        model = self.Model(
                    train_x,
                    train_y,
                    self.likelihood[d][p],
                    self.cfg,
                ).to(self.device)

        self.mll[d][p] = gpytorch.mlls.ExactMarginalLogLikelihood(
                    self.likelihood[d][p], model
                )

        return model

