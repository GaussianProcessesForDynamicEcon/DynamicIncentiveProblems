import torch
import numpy as np
from DPGPIpoptModel import DPGPIpoptModel
import cyipopt

# ======================================================================
#
#     sets the parameters for the model
#     "Growth Model"
#
#     The model is described in Scheidegger & Bilionis (2017)
#     https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2927400
#
#     Simon Scheidegger, 01/19
# ======================================================================

# ======================================================================


def utility(cons, lab, big_A, gamma, psi, eta):
    sum_util = torch.zeros(1)
    n = cons.shape[0]
    for i in range(n):
        nom1 = (cons[i] / big_A) ** (1.0 - gamma) - 1.0
        den1 = 1.0 - gamma

        nom2 = (1.0 - psi) * ((lab[i] ** (1.0 + eta)) - 1.0)
        den2 = 1.0 + eta

        sum_util += nom1 / den1 - nom2 / den2

    return sum_util


# ======================================================================
# output_f


def output_f(kap, lab, big_A, psi, d):
    theta = [1.0, 0.9, 0.95, 1.05, 1.10]
    fun_val = theta[int(d.item())] * big_A * (kap ** psi) * (lab ** (1.0 - psi))
    return fun_val


# V infinity
def V_INFINITY(k, big_A, gamma, psi, eta, beta):
    d = k[-1]
    k = k[:-1]
    e = torch.ones_like(k)
    c = output_f(k, e, big_A, psi, d)
    v_infinity = utility(c, e, big_A, gamma, psi, eta) / (1 - beta)
    return v_infinity


# ======================================================================
#   Equality constraints during the VFI of the model


def EV_G_ITER(X, k_init, n_agents, big_A, psi, zeta, delta):
    d = k_init[-1]
    k_init = k_init[:-1]
    M = 3 * n_agents + 1  # number of constraints
    G = torch.empty(M)

    # Extract Variables
    cons = X[:n_agents]
    lab = X[n_agents : 2 * n_agents]
    inv = X[2 * n_agents : 3 * n_agents]

    # first n_agents equality constraints
    for i in range(n_agents):
        G[i] = cons[i]
        G[i + n_agents] = lab[i]
        G[i + 2 * n_agents] = inv[i]

    f_prod = output_f(k_init, lab, big_A, psi, d)

    Gamma_adjust = 0.5 * zeta * k_init * ((inv / k_init - delta) ** 2.0)
    sectors_sum = cons + inv - delta * k_init - (f_prod - Gamma_adjust)
    G[3 * n_agents] = sectors_sum.sum()

    return G


# ======================================================================
class SpecifiedModel(DPGPIpoptModel):
    def __init__(self, V_guess=V_INFINITY, cfg={}, **kwargs):
        super().__init__(
            V_guess=lambda x: V_INFINITY(
                x,
                big_A=cfg["model"]["params"]["big_A"],
                gamma=cfg["model"]["params"]["gamma"],
                psi=cfg["model"]["params"]["psi"],
                eta=cfg["model"]["params"]["eta"],
                beta=cfg["model"]["params"]["beta"],
            ),
            cfg=cfg,
            policy_dim=3 * cfg["model"]["params"]["n_agents"],
            discrete_state_dim=cfg["model"]["params"].get("discrete_state_dim", 1),
            control_dim=3 * cfg["model"]["params"]["n_agents"],
            **kwargs
        )

    def sample(self):
        self.state_sample = (
            torch.rand(
                [self.cfg["no_samples"], self.cfg["model"]["params"]["n_agents"]]
            )
            * (
                self.cfg["model"]["params"]["k_up"]
                - self.cfg["model"]["params"]["k_bar"]
            )
            + self.cfg["model"]["params"]["k_bar"]
        )
        self.state_sample = torch.cat(
            (
                self.state_sample,
                torch.randint(
                    0,
                    self.discrete_state_dim,
                    size=torch.Size([self.state_sample.shape[0], 1]),
                    dtype=torch.float64,
                ),
            ),
            dim=1,
        )

    def u(self, state, control):
        cons = control[0 : self.cfg["model"]["params"]["n_agents"]]
        lab = control[
            self.cfg["model"]["params"]["n_agents"] : 2
            * self.cfg["model"]["params"]["n_agents"]
        ]
        return utility(
            cons,
            lab,
            big_A=self.cfg["model"]["params"]["big_A"],
            gamma=self.cfg["model"]["params"]["gamma"],
            psi=self.cfg["model"]["params"]["psi"],
            eta=self.cfg["model"]["params"]["eta"],
        )

    def state_iterate(self, state, control):
        return torch.cat(
            (
                (1 - self.cfg["model"]["params"]["delta"]) * state[:-1]
                + control[
                    2
                    * self.cfg["model"]["params"]["n_agents"] : 3
                    * self.cfg["model"]["params"]["n_agents"]
                ],
                state[-1:],
            )
        )

    def eval_g(self, state, control):
        return EV_G_ITER(
            control,
            state,
            self.cfg["model"]["params"]["n_agents"],
            big_A=self.cfg["model"]["params"]["big_A"],
            psi=self.cfg["model"]["params"]["psi"],
            zeta=self.cfg["model"]["params"]["zeta"],
            delta=self.cfg["model"]["params"]["delta"],
        )

    def lb(self, state):
        X_L = np.empty(3 * self.cfg["model"]["params"]["n_agents"])
        X_L[: self.cfg["model"]["params"]["n_agents"]] = self.cfg["model"]["params"][
            "c_bar"
        ]
        X_L[
            self.cfg["model"]["params"]["n_agents"] : 2
            * self.cfg["model"]["params"]["n_agents"]
        ] = self.cfg["model"]["params"]["l_bar"]
        X_L[
            2
            * self.cfg["model"]["params"]["n_agents"] : 3
            * self.cfg["model"]["params"]["n_agents"]
        ] = self.cfg["model"]["params"]["inv_bar"]
        return X_L

    def ub(self, state):
        X_U = np.empty(3 * self.cfg["model"]["params"]["n_agents"])
        X_U[: self.cfg["model"]["params"]["n_agents"]] = self.cfg["model"]["params"][
            "c_up"
        ]
        X_U[
            self.cfg["model"]["params"]["n_agents"] : 2
            * self.cfg["model"]["params"]["n_agents"]
        ] = self.cfg["model"]["params"]["l_up"]
        X_U[
            2
            * self.cfg["model"]["params"]["n_agents"] : 3
            * self.cfg["model"]["params"]["n_agents"]
        ] = self.cfg["model"]["params"]["inv_up"]
        return X_U

    def cl(self, state):
        n_agents = self.cfg["model"]["params"]["n_agents"]
        M = 3 * n_agents + 1
        G_L = np.empty(M)
        G_L[:n_agents] = self.cfg["model"]["params"]["c_bar"]
        G_L[n_agents : 2 * n_agents] = self.cfg["model"]["params"]["l_bar"]
        G_L[2 * n_agents : 3 * n_agents] = self.cfg["model"]["params"]["inv_bar"]
        G_L[3 * n_agents] = 0.0
        return G_L

    def cu(self, state):
        n_agents = self.cfg["model"]["params"]["n_agents"]
        M = 3 * n_agents + 1
        # number of constraints
        G_U = np.empty(M)
        # Set bounds for the constraints
        G_U[:n_agents] = self.cfg["model"]["params"]["c_up"]
        G_U[n_agents : 2 * n_agents] = self.cfg["model"]["params"]["l_up"]
        G_U[2 * n_agents : 3 * n_agents] = self.cfg["model"]["params"]["inv_up"]
        G_U[3 * n_agents] = 0.0
        return G_U
