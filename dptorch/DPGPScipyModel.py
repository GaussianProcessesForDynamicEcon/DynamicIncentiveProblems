from tracemalloc import start
from scipy import optimize
import gpytorch
import logging
import torch
from DPGPModel import DPGPModel
from abc import abstractmethod
from Utils import NonConvergedError
import numpy as np

logger = logging.getLogger(__name__)

torch.set_default_dtype(torch.float64)

class ScipyModel:
    def __init__(self, DPGM, state, params, eval_g_eq, eval_g_ineq, control_dim):
        self._DPGM = DPGM
        self._state = state
        self._params = params
        self._eval_g_eq = eval_g_eq
        self._eval_g_ineq = eval_g_ineq
        self._control_dim = control_dim


    def eval_hessian_f(self, x):
        h = torch.autograd.functional.hessian(
            lambda y: self._DPGM.eval_f(self._state,self._params, y),
            x,
            vectorize=True,
        )
        return h

    # cpyipopt requirements
    def objective(self, x):
        """Returns the scalar value of the objective given x."""
        return (
            -1*self._DPGM.eval_f(self._state,self._params, torch.from_numpy(x)).detach().cpu().numpy()
        )

    def gradient_obj(self, x):
        """Returns the gradient of the objective with respect to x."""
        return (
            -1*self._DPGM.eval_grad_f(self._state,self._params, torch.from_numpy(x))
            .detach()
            .cpu()
            .numpy()
        )

    def hessian_obj(self, x):
        """Returns the gradient of the objective with respect to x."""
        return (
            -1*self.eval_hessian_f(torch.from_numpy(x))
            .detach()
            .cpu()
            .numpy()
        )

    def eval_jac_g_eq(self, x):
        jac = torch.autograd.functional.jacobian(self._eval_g_eq, x)
        return jac

    def eval_hessian_g_eq(self, x, lagrange):
        L = torch.from_numpy(lagrange)
        h = torch.autograd.functional.hessian(
            lambda y: torch.dot(
                self._eval_g_eq(y),
                L,
            ),
            x,
            vectorize=True,
        )
        return h

    def constraints_eq(self, x):
        """Returns the constraints."""
        return self._eval_g_eq(torch.from_numpy(x)).detach().numpy()

    def jacobian_constraints_eq(self, x):
        """Returns the Jacobian of the constraints with respect to x."""
        return self.eval_jac_g_eq(torch.from_numpy(x)).detach().cpu().numpy()

    def hessian_constraints_eq(self, x, lagrange):
        h = (
            self.eval_hessian_g_eq(torch.from_numpy(x), lagrange)
            .detach()
            .cpu()
            .numpy()
        )
        return h


    def eval_jac_g_ineq(self, x):
        jac = torch.autograd.functional.jacobian(self._eval_g_ineq, x)
        return jac

    def eval_hessian_g_ineq(self, x, lagrange):
        L = torch.from_numpy(lagrange)
        h = torch.autograd.functional.hessian(
            lambda y: torch.dot(
                self._eval_g_ineq(y),
                L,
            ),
            x,
            vectorize=True,
        )
        return h

    def constraints_ineq(self, x):
        """Returns the constraints."""
        return self._eval_g_ineq(torch.from_numpy(x)).detach().numpy()

    def jacobian_constraints_ineq(self, x):
        """Returns the Jacobian of the constraints with respect to x."""
        return self.eval_jac_g_ineq(torch.from_numpy(x)).detach().cpu().numpy()

    def hessian_constraints_ineq(self, x, lagrange):
        h = (
            self.eval_hessian_g_ineq(torch.from_numpy(x), lagrange)
            .detach()
            .cpu()
            .numpy()
        )
        return h

class ScipyModel_max_vf:
    def __init__(self, DPGM, disc_state, control_dim):
        self._DPGM = DPGM
        self._disc_state = int(disc_state)
        self._control_dim = control_dim


    # cpyipopt requirements
    def objective(self, x):
        """Returns the scalar value of the objective given x."""
        return (
            self._DPGM.eval_f_max_vf(self._disc_state,torch.from_numpy(x)).detach().cpu().numpy()
        )

    def gradient_obj(self, x):
        """Returns the gradient of the objective with respect to x."""
        return (
            self._DPGM.eval_grad_f_max_vf(self._disc_state,torch.from_numpy(x))
            .detach()
            .cpu()
            .numpy()
        )



    

class DPGPScipyModel(DPGPModel):
    def __init__(self, control_dim, **kwargs):
        super().__init__(**kwargs)
        self.control_dim = control_dim

    @abstractmethod
    def eval_g(self, state, params, control):
        pass

    @abstractmethod
    def lb(self, state, params):
        pass

    @abstractmethod
    def ub(self, state, params):
        pass

    @abstractmethod
    def cl(self, state, params):
        pass

    @abstractmethod
    def cu(self, state, params):
        pass

    def x_init(self, state):
        return 0.5 * self.lb(state) + 0.5 * self.ub(state)

    def get_params(self,state,policy):
        return torch.zeros(1)

    def solve(self, state, policy):
        params = self.get_params(state,policy)
        X_L = self.lb(state,params)
        X_U = self.ub(state,params)
        G_L = self.cl(state,params)
        G_U = self.cu(state,params)
        mask_eq = G_L == G_U
        mask_ineq = np.logical_not(mask_eq)

        scipy_obj = ScipyModel(
                self,
                state,
                params,
                lambda c: self.eval_g(state.detach(), params.detach(), c)[mask_eq],
                lambda c: self.eval_g(state.detach(), params.detach(), c)[mask_ineq],
                self.control_dim
            )        

        if sum(mask_eq) > 0:
            nonlinear_constraints_eq = optimize.NonlinearConstraint(
                scipy_obj.constraints_eq,
                G_L[mask_eq],
                G_U[mask_eq],
                jac=scipy_obj.jacobian_constraints_eq,
                hess=scipy_obj.hessian_constraints_eq,
                )
        if sum(mask_ineq) > 0:
            nonlinear_constraints_ineq = optimize.NonlinearConstraint(
                scipy_obj.constraints_ineq,
                G_L[mask_ineq],
                G_U[mask_ineq],
                jac=scipy_obj.jacobian_constraints_ineq,
                hess=scipy_obj.hessian_constraints_ineq,
                )
        if sum(mask_eq) > 0 and sum(mask_ineq) > 0:
            constr_lst = [nonlinear_constraints_eq,nonlinear_constraints_ineq]
        elif sum(mask_eq) == 0 and sum(mask_ineq) > 0:
            constr_lst = nonlinear_constraints_ineq
        elif sum(mask_eq) > 0 and sum(mask_ineq) == 0:         
            constr_lst = nonlinear_constraints_eq
        else:
            constr_lst = []

        var_bounds = optimize.Bounds(lb=X_L,ub=X_U,keep_feasible=True)

        options_dict = {}

        n_restarts = self.cfg["scipyopt"].get("no_restarts", 1)
        
        start_pts = self.sample_start_pts(state, params, policy ,n_restarts)

        x_vec = -1e10*np.ones((n_restarts,self.control_dim+2))
        info_vec = -1*np.ones(n_restarts,dtype=np.int32)

        n_iter_vec = np.zeros(n_restarts)
        prim_feas_vec = np.zeros(n_restarts)
        for indxit in range(n_restarts):
            x_init = start_pts[indxit,:].clone().detach().numpy()
            res = optimize.minimize(
                scipy_obj.objective,
                x_init,
                method=self.cfg["scipyopt"].get("method", 'SLSQP'), #'trust-constr', 
                jac=scipy_obj.gradient_obj, 
                hess=scipy_obj.hessian_obj, 
                bounds=var_bounds, 
                constraints=constr_lst, 
                tol=self.cfg["scipyopt"].get("tol", 1e-6),
                options=options_dict)

            info_vec[indxit] = res.success
            n_iter_vec[indxit] =  res.nit
            prim_feas_vec[indxit] = 0.#res.constr_violation
            x_vec[indxit,0] = -1*res.fun
            x_vec[indxit,2:] = res.x
            if info_vec[indxit] == 1:
                x_vec[indxit,1] = -1*res.fun
        
        mask = info_vec==1
        x_converg = x_vec[mask,:]
        if x_converg.shape[0] == 0:
            logger.info(f"Scipy failed to converge in state {state} with return codes {info_vec} no iterations {n_iter_vec} prim feas {prim_feas_vec}")
            raise NonConvergedError("Scipy did not converge (to acceptable tolerance)")

        max_indx = np.argmax(x_vec[:,1])
        # logger.info(f"Scipy converged in state {state} with return codes {info_vec[max_indx]} no iterations {n_iter_vec[max_indx]} prim feas {prim_feas_vec[max_indx]}")
        logger.debug(state)
        logger.debug(info_vec)
        logger.debug(x_vec)
        control,value = self.post_process_optimization(state, params, torch.from_numpy(x_vec[max_indx,2:]), torch.tensor(x_vec[max_indx,0]))
        info_out = {}
        info_out["obj_val"] = value
        info_out["x"] = control

        min_val = self.cfg["model"]["params"].get("min_val", -1.0e15)
        return max(min_val,info_out["obj_val"]), info_out["x"]


    def eval_f_max_vf(self, disc_state, control):
        out = torch.zeros(1)
        out += self.M[disc_state][0](torch.unsqueeze(control,dim=0)).mean
        return out
        

    def eval_grad_f_max_vf(self, disc_state, control):
        c = control.clone().detach().requires_grad_(True)
        u = self.eval_f_max_vf(disc_state, c)
        u.backward()
        return c.grad

    def eval_g_zero(self,control):
        return torch.tensor([])

    def find_max_of_vf(self, disc_state, start_pt,LB_var,UB_var):
        X_L = LB_var
        X_U = UB_var

        scipy_obj = ScipyModel_max_vf(
                self,
                disc_state,
                len(X_L)
            )  

        options_dict = {}

        var_bounds = optimize.Bounds(lb=X_L,ub=X_U,keep_feasible=True)
        
        x_init = start_pt.clone().detach().numpy()
        res = optimize.minimize(
            scipy_obj.objective,
            x_init,
            method=self.cfg["scipyopt"].get("method_gpopt", 'L-BFGS-B'), 
            jac=scipy_obj.gradient_obj, 
            bounds=var_bounds, 
            tol=self.cfg["scipyopt"].get("tol_gpopt", 1e-6),
            options=options_dict)

        if res.success:
            logger.debug(res)
        else:
            logger.info(res)
            raise NonConvergedError("Scipy did not converge (to acceptable tolerance)")

        return -res.fun, res.x


    def generate_trajectory(self, trajectory_length=None, from_sample=False):
        if not from_sample:
            state_trajectory = [torch.mean(self.state_sample_all, dim=0)]
        else:
            self.cfg['no_samples'] = trajectory_length
            self.sample_all()
            state_trajectory = self.state_sample_all
        policy_trajectory = []
        eval_g_trajectory = []
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            for i in range(trajectory_length):
                policy_trajectory.append(self.predict_policies(state_trajectory[i]))
                eval_g_trajectory.append(
                    self.eval_g(state_trajectory[i], policy_trajectory[i][:self.control_dim])
                )
                # sample next state
                weights, points = self.state_iterate_exp(
                    state_trajectory[i], policy_trajectory[i][:self.control_dim]
                )
                if not from_sample:
                    state_trajectory.append(
                        points[int(torch.multinomial(weights, num_samples=1).item()), :]
                    )
        return state_trajectory, policy_trajectory, eval_g_trajectory
