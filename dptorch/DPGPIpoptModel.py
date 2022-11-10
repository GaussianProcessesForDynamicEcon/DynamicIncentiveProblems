from tracemalloc import start
import cyipopt
import gpytorch
import logging
import torch
from DPGPModel import DPGPModel
from abc import abstractmethod
from Utils import NonConvergedError
import numpy as np

logger = logging.getLogger(__name__)

torch.set_default_dtype(torch.float64)

class IpoptModel:
    def __init__(self, DPGM, state, params, eval_g, control_dim):
        self._DPGM = DPGM
        self._state = state
        self._params = params
        self._eval_g = eval_g
        self._control_dim = control_dim
        self.n_iter = -1
        self.dual_feas = 1.0e4
        self.primal_feas = 1.0e4

    def eval_jac_g(self, x):
        return torch.autograd.functional.jacobian(self._eval_g, x)

    def eval_hessian_g(self, x, lagrange, obj_factor):
        L = torch.cat(
            (obj_factor.unsqueeze(0), lagrange)
        )
        h = torch.autograd.functional.hessian(
            lambda y: torch.dot(
                torch.cat((self._DPGM.eval_f(self._state,self._params, y), self._eval_g(y))),
                L,
            ),
            x,
            vectorize=True,
        )
        return h

    # cpyipopt requirements
    def objective(self, x):
        """Returns the scalar value of the objective given x."""
        return (
            self._DPGM.eval_f(self._state,self._params, torch.from_numpy(x)).detach().cpu().numpy()
        )

    def gradient(self, x):
        """Returns the gradient of the objective with respect to x."""
        return (
            self._DPGM.eval_grad_f(self._state,self._params, torch.from_numpy(x))
            .detach()
            .cpu()
            .numpy()
        )


    def constraints(self, x):
        """Returns the constraints."""
        return self._eval_g(torch.from_numpy(x)).detach().numpy()

    def jacobian(self, x):
        """Returns the Jacobian of the constraints with respect to x."""
        return self.eval_jac_g(torch.from_numpy(x)).detach().cpu().numpy().flatten()

    def hessianstructure(self):
        return np.tril_indices_from(np.eye(self._control_dim))

    def hessian(self, x, lagrange, obj_factor):
        h = (
            self.eval_hessian_g(torch.from_numpy(x), torch.from_numpy(lagrange), torch.tensor(obj_factor))
            .detach()
            .cpu()
            .numpy()
        )
        return h[np.tril_indices_from(h)]

    def intermediate(
            self,
            alg_mod,
            iter_count,
            obj_value,
            inf_pr,
            inf_du,
            mu,
            d_norm,
            regularization_size,
            alpha_du,
            alpha_pr,
            ls_trials
            ):

        #
        # Example for the use of the intermediate callback.
        #
        self.n_iter = iter_count
        self.dual_feas = inf_du
        self.primal_feas = inf_pr
        # print("Objective value at iteration #%d is - %g" % (iter_count, obj_value))


class IpoptModel_max_vf:
    def __init__(self, DPGM, disc_state, eval_g, control_dim):
        self._DPGM = DPGM
        self._eval_g = eval_g
        self._disc_state = int(disc_state)
        self._control_dim = control_dim
        self.n_iter = -1
        self.dual_feas = 1.0e4
        self.primal_feas = 1.0e4


    def eval_jac_g(self, x):
        return torch.autograd.functional.jacobian(self._eval_g, x)

    def eval_hessian_g(self, x, lagrange, obj_factor):
        L = torch.cat(
            (obj_factor.unsqueeze(0), lagrange)
        )
        h = torch.autograd.functional.hessian(
            lambda y: torch.dot(
                torch.cat((self._DPGM.eval_f_max_vf(self._disc_state,y), self._eval_g(y))),
                L,
            ),
            x,
            vectorize=True,
        )
        return h

    # cpyipopt requirements
    def objective(self, x):
        """Returns the scalar value of the objective given x."""
        return (
            self._DPGM.eval_f_max_vf(self._disc_state,torch.from_numpy(x)).detach().cpu().numpy()
        )

    def gradient(self, x):
        """Returns the gradient of the objective with respect to x."""
        return (
            self._DPGM.eval_grad_f_max_vf(self._disc_state,torch.from_numpy(x))
            .detach()
            .cpu()
            .numpy()
        )


    def constraints(self, x):
        """Returns the constraints."""
        return self._eval_g(torch.from_numpy(x)).detach().numpy()

    def jacobian(self, x):
        """Returns the Jacobian of the constraints with respect to x."""
        return self.eval_jac_g(torch.from_numpy(x)).detach().cpu().numpy().flatten()

    def hessianstructure(self):
        return np.tril_indices_from(np.eye(self._control_dim))

    def hessian(self, x, lagrange, obj_factor):
        h = (
            self.eval_hessian_g(torch.from_numpy(x), torch.from_numpy(lagrange), torch.tensor(obj_factor))
            .detach()
            .cpu()
            .numpy()
        )
        return h[np.tril_indices_from(h)]

    def intermediate(
            self,
            alg_mod,
            iter_count,
            obj_value,
            inf_pr,
            inf_du,
            mu,
            d_norm,
            regularization_size,
            alpha_du,
            alpha_pr,
            ls_trials
            ):

        #
        # Example for the use of the intermediate callback.
        #
        self.n_iter = iter_count
        self.dual_feas = inf_du
        self.primal_feas = inf_pr
        # print("Objective value at iteration #%d is - %g" % (iter_count, obj_value))


class DPGPIpoptModel(DPGPModel):
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

    def x_init(self, state, params):
        return 0.5 * self.lb(state,params) + 0.5 * self.ub(state,params)

    def get_params(self,state,policy):
        return torch.zeros(1)

    def solve(self, state, policy):
        params = self.get_params(state,policy)
        X_L = self.lb(state,params)
        X_U = self.ub(state,params)
        G_L = self.cl(state,params)
        G_U = self.cu(state,params)

        ipopt_obj = IpoptModel(
                self,
                state,
                params,
                lambda c: self.eval_g(state.detach(), params.detach(), c),
                self.control_dim
            )
        nlp = cyipopt.Problem(
            n=len(X_L),
            m=len(G_L),
            problem_obj=ipopt_obj,
            lb=X_L,
            ub=X_U,
            cl=G_L,
            cu=G_U,
        )
        nlp.add_option("obj_scaling_factor", -1.0)
        nlp.add_option("tol", self.cfg["ipopt"].get("tol", 1e-6))
        nlp.add_option("acceptable_tol", self.cfg["ipopt"].get("acceptable_tol", 1e-5))
        nlp.add_option("max_iter", self.cfg["ipopt"].get("max_iterations", 1000))
        nlp.add_option("constr_viol_tol", self.cfg["ipopt"].get("constr_viol_tol", 1e-8))
        nlp.add_option("bound_relax_factor",0.0)
        nlp.add_option("honor_original_bounds",'yes')
        # nlp.add_option("linear_solver",'ma57')   
        # nlp.add_option("nlp_scaling_method","equilibration-based")
        if self.cfg["ipopt"].get("derivative_test"):
            nlp.add_option("derivative_test", self.cfg["ipopt"].get("derivative_test"))
            nlp.add_option(
                "derivative_test_perturbation",
                self.cfg["ipopt"].get("derivative_test_perturbation", 1e-6),
            )
            # update print level
            self.cfg["ipopt"]["print_level"] = max(
                5, self.cfg["ipopt"].get("print_level", 1)
            )

        nlp.add_option(
            "hessian_approximation",
            self.cfg["ipopt"].get("hessian_approximation", "exact"),
        )

        nlp.add_option("max_cpu_time", self.cfg["ipopt"].get("max_cpu_time", 600.0))
        nlp.add_option("print_level", self.cfg["ipopt"].get("print_level", 1))
        n_restarts = self.cfg["ipopt"].get("no_restarts", 1)
        scale_vec = self.scaling_vector()
        nlp.set_problem_scaling(x_scaling=scale_vec)
        
        if n_restarts == 1:
            x, info = nlp.solve(self.x_init(state,params).clone().detach().numpy())
            if info["status"] != 0 and info["status"] != 1:
                logger.debug(info)
                raise NonConvergedError("Ipopt did not converge (to acceptable tolerance)")
            else:
                logger.debug(state)
                logger.debug(info)

            info_out = {}
            info_out["obj_val"] = torch.tensor(info["obj_val"])
            info_out["x"] = torch.from_numpy(info["x"])
        else:
            start_pts = self.sample_start_pts(state, params, policy ,n_restarts)

            x_vec = -1e10*np.ones((n_restarts,self.control_dim+2))
            g_vec = np.zeros((n_restarts,len(G_L)))
            info_vec = -1*np.ones(n_restarts,dtype=np.int32)

            n_iter_vec = np.zeros(n_restarts)
            prim_feas_vec = np.zeros(n_restarts)
            dual_feas_vec = np.zeros(n_restarts)
            for indxit in range(n_restarts):
                x_init = start_pts[indxit,:].clone().detach().numpy()
                x, info = nlp.solve(x_init)
                n_iter_vec[indxit] = ipopt_obj.n_iter
                prim_feas_vec[indxit] = ipopt_obj.primal_feas
                dual_feas_vec[indxit] = ipopt_obj.dual_feas
                info_vec[indxit] = info["status"]
                x_vec[indxit,0] = info["obj_val"]
                x_vec[indxit,2:] = info["x"]
                g_vec[indxit,:] = info["g"]
                if info_vec[indxit] == 0 or info_vec[indxit] == 1:
                    x_vec[indxit,1] = info["obj_val"]
            
            mask = np.logical_or(info_vec==-2,np.logical_or(info_vec==1,info_vec==0))
            x_converg = x_vec[mask,:]
            if x_converg.shape[0] == 0:
                logger.info(f"Ipopt failed to converge in state {state} with return codes {info_vec} no iterations {n_iter_vec} prim feas {prim_feas_vec} dual feas {dual_feas_vec}")
                raise NonConvergedError("Ipopt did not converge (to acceptable tolerance)")

            max_indx = np.argmax(x_vec[:,1])
            # logger.info(f"Ipopt converged in state {state} with return codes {info_vec[max_indx]} no iterations {n_iter_vec[max_indx]} prim feas {prim_feas_vec[max_indx]} dual feas {dual_feas_vec[max_indx]}")
            logger.debug(state)
            logger.debug(info_vec)
            logger.debug(x_vec)
            logger.debug(g_vec)
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
        G_L = torch.zeros(1)
        G_U = torch.zeros(1)

        nlp = cyipopt.Problem(
            n=len(X_L),
            m=0,
            problem_obj=IpoptModel_max_vf(
                self,
                disc_state,
                lambda c: self.eval_g_zero(c),
                len(X_L)
            ),
            lb=X_L,
            ub=X_U,
            cl=G_L,
            cu=G_U,
        )
        nlp.add_option("obj_scaling_factor", -1.0)
        nlp.add_option("tol", self.cfg["ipopt"].get("tol", 1e-6))
        nlp.add_option("acceptable_tol", self.cfg["ipopt"].get("acceptable_tol", 1e-5))
        nlp.add_option("max_iter", self.cfg["ipopt"].get("max_iterations", 1000))
        nlp.add_option("constr_viol_tol", self.cfg["ipopt"].get("constr_viol_tol", 1e-8))
        # nlp.add_option("linear_solver",'ma57')
        nlp.add_option("bound_relax_factor",0.0)
        nlp.add_option("honor_original_bounds",'yes')
        if self.cfg["ipopt"].get("derivative_test"):
            nlp.add_option("derivative_test", self.cfg["ipopt"].get("derivative_test"))
            nlp.add_option(
                "derivative_test_perturbation",
                self.cfg["ipopt"].get("derivative_test_perturbation", 1e-6),
            )
            # update print level
            self.cfg["ipopt"]["print_level"] = max(
                5, self.cfg["ipopt"].get("print_level", 1)
            )

        nlp.add_option(
            "hessian_approximation",
            self.cfg["ipopt"].get("hessian_approximation", "exact"),
        )

        nlp.add_option("max_cpu_time", self.cfg["ipopt"].get("max_cpu_time", 600.0))
        nlp.add_option("print_level", self.cfg["ipopt"].get("print_level", 1))
        
        x, info = nlp.solve(start_pt.clone().detach().numpy())
        if info["status"] != 0 and info["status"] != 1:
            logger.info(f"failed to converge when finding max GP value with info {info}")
        else:
            logger.debug(info)
            logger.debug(x)

        return info["obj_val"], info["x"]


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
