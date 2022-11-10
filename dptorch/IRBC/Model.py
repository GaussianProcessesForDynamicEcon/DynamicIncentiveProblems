import torch
import numpy as np
from DPGPIpoptModel import DPGPIpoptModel
from DPGPScipyModel import DPGPScipyModel


################################################################################
#                           Growth function                                #
################################################################################


def growth(k_next, k_prev):
    k_next = torch.max(k_next, torch.tensor([1e-6]))
    k_prev = torch.max(k_prev, torch.tensor([1e-6]))

    return k_next / k_prev - 1.0


################################################################################
#                           Production function                                #
################################################################################


def F(k_t, a_t, A_tfp, alpha):

    FF = a_t * A_tfp * torch.max(k_t, torch.tensor([1e-6])) ** alpha

    return FF


################################################################################
#                      Marginal product of capital                             #
################################################################################


def Fk(k_tp1, a_tp1, A_tfp, alpha):

    F_k = (
        a_tp1 * A_tfp * alpha * torch.max(k_tp1, torch.tensor([1e-6])) ** (alpha - 1.0)
    )

    return F_k


################################################################################
#                          Capital adjustment cost                             #
################################################################################


def AdjCost(k_t, k_tp1, phi):
    g_tp1 = growth(k_tp1, k_t)
    Adj_cost = 0.5 * phi * g_tp1 * g_tp1 * k_t

    return Adj_cost


################################################################################
#        Derivative of capital adjustment cost w.r.t today's cap stock         #
################################################################################


def AdjCost_k(k_tp1, k_tp2, phi):
    g_tp2 = growth(k_tp2, k_tp1)
    g_tp2_p2 = growth(k_tp2, k_tp1) + 2.0
    AdjCostk = (-0.5) * phi * g_tp2 * g_tp2_p2

    return AdjCostk


################################################################################
#      Derivative of capital adjustment cost w.r.t tomorrows's cap stock       #
################################################################################


def AdjCost_ktom(k_t, k_tp1, phi):
    j = growth(k_tp1, k_t)
    AdjCostktom = phi * j

    return AdjCostktom


################################################################################
#      Shocked a state variable     #
################################################################################


def shocks_MC(MC_N, nShocks):
    return torch.randn(MC_N, nShocks)


def shocks_monomial(nShocks):
    return np.sqrt(nShocks / 2) * torch.cat(
        (torch.diag(torch.ones(nShocks)), -1 * torch.diag(torch.ones(nShocks)))
    )


def next_a(a_t, shocks, rhoZ, sigma, aMin, aMax):
    shock_global = shocks[0]
    country_shocks = shocks[1:]
    res = torch.abs(a_t) ** rhoZ * torch.exp(sigma * (country_shocks + shock_global))

    return torch.max(torch.tensor([aMin]), torch.min(torch.tensor([aMax]), res))


def a_tp1s_weights_MC(nCountries, nShocks, MC_N, a_t, rhoZ, sigma, aMin, aMax):
    a_tp1s = torch.zeros(MC_N, nCountries)
    weights = torch.ones(MC_N) / MC_N
    shocks = shocks_MC(MC_N, nShocks)
    for idx, sh in enumerate(shocks):
        a_tp1s[idx, :] = next_a(a_t, sh, rhoZ, sigma, aMin, aMax)
    return a_tp1s, weights


def a_tp1s_weights_monomial(nCountries, nShocks, a_t, rhoZ, sigma, aMin, aMax):
    a_tp1s = torch.zeros(2 * nShocks, nCountries)
    weights = torch.ones(2 * nShocks) / (2 * nShocks)
    shocks = shocks_monomial(nShocks)
    for idx, sh in enumerate(shocks):
        a_tp1s[idx, :] = next_a(a_t, sh, rhoZ, sigma, aMin, aMax)
    return a_tp1s, weights


# ======================================================================
class SpecifiedModel(DPGPScipyModel):
    def __init__(self, cfg={}, **kwargs):
        self.no_samples = cfg["no_samples"]
        self.nCountries = cfg["model"]["params"]["nCountries"]
        self.nPols = cfg["model"]["params"]["nPols"]
        self.nShocks = cfg["model"]["params"]["nShocks"]
        self.beta = cfg["model"]["params"]["beta"]
        self.delta = cfg["model"]["params"]["delta"]
        self.pareto = cfg["model"]["params"]["pareto"]
        self.gamma = cfg["model"]["params"]["gamma"]
        self.rhoZ = cfg["model"]["params"]["rhoZ"]
        self.phi = cfg["model"]["params"]["phi"]
        self.A_tfp = cfg["model"]["params"]["A_tfp"]
        self.alpha = cfg["model"]["params"]["alpha"]
        self.sigma = cfg["model"]["params"]["sigE"]
        self.aMin = cfg["model"]["params"]["aMin"]
        self.aMax = cfg["model"]["params"]["aMax"]
        self.lMin = cfg["model"]["params"]["lMin"]
        self.lMax = cfg["model"]["params"]["lMax"]
        self.kMin = cfg["model"]["params"]["kMin"]
        self.kMax = cfg["model"]["params"]["kMax"]
        self.MC_N = cfg["model"]["params"]["MC_N"]
        self.typeInt = cfg["model"]["params"]["typeInt"]
        super().__init__(
            V_guess=lambda x: 0,
            cfg=cfg,
            policy_dim=cfg["model"]["params"]["nPols"],
            control_dim=cfg["model"]["params"]["nPols"],
            discrete_state_dim=1,
            **kwargs,
        )
        self.gp_fit_start_epoch = 1

    # Initial residuals function
    def EV_G(self, X, state):
        # Solver variables
        k_tp1 = X[0 : self.nCountries]
        lambda_t = X[self.nCountries]

        # Previous state variables
        a_t = state[0 : self.nCountries]
        k_t = state[self.nCountries : -1]
        # Initial lambda is set to the steady state 1.0
        lambda_init = 1.0

        # Next state variables in the first step have no shocks
        a_tp1 = torch.absolute(a_t) ** self.rhoZ
        # Residuals
        res = torch.zeros(self.nCountries + 1)

        # Euler equations
        for ires in range(self.nCountries):
            res[ires] = lambda_t * (
                1 + self.phi * growth(k_tp1[ires], k_t[ires])
            ) - self.beta * lambda_init * (
                Fk(k_tp1[ires], a_tp1[ires], self.A_tfp, self.alpha) + 1.0 - self.delta
            )

        # Aggregate resource constraint
        for ires2 in range(self.nCountries):
            res[self.nCountries] = res[self.nCountries] + (
                F(k_t[ires2], a_t[ires2], self.A_tfp, self.alpha)
                + (1.0 - self.delta) * k_t[ires2]
                - k_tp1[ires2]
                - AdjCost(k_t[ires2], k_tp1[ires2], self.phi)
                - (lambda_t / self.pareto) ** (-1.0 / self.gamma)
            )

        return res

    # Residuals function
    def EV_G_ITER(self, X, state):
        # Solver variables
        k_tp1 = X[0 : self.nCountries]
        lambda_t = X[self.nCountries]

        # Previous state variables
        a_t = state[0 : self.nCountries]
        k_t = state[self.nCountries : -2]
        lambda_tm1 = state[-2]
        dim_ind = int(state[-1].item())

        res = torch.zeros(self.nPols)
        # Computation of residuals of the equilibrium system of equations
        if self.typeInt == "MC":
            a_tp1s, weights = a_tp1s_weights_MC(
                self.nCountries,
                self.nShocks,
                self.MC_N,
                a_t,
                self.rhoZ,
                self.sigma,
                self.aMin,
                self.aMax,
            )
        elif self.typeInt == "monomial":
            a_tp1s, weights = a_tp1s_weights_monomial(
                self.nCountries,
                self.nShocks,
                a_t,
                self.rhoZ,
                self.sigma,
                self.aMin,
                self.aMax,
            )
        else:
            raise ValueError(
                f"typeInt must be one of [MC, monomial], received {self.typeInt}"
            )
        for idx, a_tp1 in enumerate(a_tp1s):
            weight = weights[idx]
            state_next = torch.unsqueeze(
                torch.cat((a_tp1, k_tp1, torch.unsqueeze(lambda_t, dim=0)), dim=0),
                dim=0,
            )
            # Interpolate capital and lambda
            # Note that the first model is for the value function, so we need to shift the indices by one
            raw_values = torch.zeros(self.nPols)
            for iint in range(self.nPols):
                raw_values[iint] = self.M[dim_ind][1 + iint](state_next).mean[0]

            k_tp2 = raw_values[: self.nCountries]
            lambda_tp1 = raw_values[self.nCountries]

            for iev in range(self.nCountries):
                temp = lambda_t * (
                    1 + AdjCost_ktom(k_t[iev], k_tp1[iev], self.phi)
                ) - self.beta * lambda_tp1 * (
                    1.0
                    - self.delta
                    + Fk(k_tp1[iev], a_tp1[iev], self.A_tfp, self.alpha)
                    - AdjCost_k(k_tp1[iev], k_tp2[iev], self.phi)
                )
                res[iev] = res[iev] + temp * weight

        # Aggregate resource constraint
        for ires2 in range(self.nCountries):
            temp = (
                F(k_t[ires2], a_t[ires2], self.A_tfp, self.alpha)
                + (1.0 - self.delta) * k_t[ires2]
                - k_tp1[ires2]
                - AdjCost(k_t[ires2], k_tp1[ires2], self.phi)
                - (lambda_t / self.pareto) ** (-1.0 / self.gamma)
            )
            res[self.nCountries] = res[self.nCountries] + temp

        return res

    def sample(self,no_samples=None):
        self.state_sample = (
            torch.rand([self.no_samples, self.nCountries]) * (self.aMax - self.aMin)
            + self.aMin
        )
        self.state_sample = torch.cat(
            (
                self.state_sample,
                torch.rand([self.no_samples, self.nCountries]) * (self.kMax - self.kMin)
                + self.kMin,
            ),
            dim=1,
        )
        self.state_sample = torch.cat(
            (
                self.state_sample,
                torch.rand([self.no_samples, 1]) * (self.lMax - self.lMin) + self.lMin,
            ),
            dim=1,
        )
        self.state_sample = torch.cat(
            (self.state_sample, torch.zeros([self.no_samples, 1])),
            dim=1,
        )
        self.feasible = torch.ones(self.state_sample.shape[0])
        self.combined_sample = torch.zeros([self.state_sample.shape[0], 1+self.policy_dim])

    def eval_g(self, state, params, control):
        if self.epoch == 1:
            return self.EV_G(X=control, state=state)
        else:
            return self.EV_G_ITER(X=control, state=state)

    def lb(self, state, params):
        X_L = torch.zeros(self.policy_dim)
        X_L[0 : self.nCountries] = self.kMin
        X_L[self.nCountries] = self.lMin
        return X_L

    def ub(self, state, params):
        X_U = torch.zeros(self.policy_dim)
        X_U[0 : self.nCountries] = self.kMax
        X_U[self.nCountries] = self.lMax
        return X_U

    def cl(self, state, params):
        # Equality contraints - set to 0 straight
        G_L = torch.zeros(self.control_dim)
        return G_L

    def cu(self, state, params):
        # Equality contraints - set to 0 straight
        G_U = torch.zeros(self.control_dim)
        return G_U

    # no value-function iteration
    def E_V(self, state, params, control):
        return torch.sum(control) * torch.zeros(1)
