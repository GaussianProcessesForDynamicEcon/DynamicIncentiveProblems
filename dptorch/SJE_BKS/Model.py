import torch
import numpy as np
from DPGPIpoptModel import DPGPIpoptModel


def generate_sample(cfg, discrete_state_dim):
    n = cfg["no_samples"]
    p = cfg["model"]["params"]
    # sample K
    K_sample = 0 * p["A"] + 1.5 * p["A"] * torch.rand((n, p["N"]))
    # sample w
    w_sample = torch.nn.functional.normalize(
        torch.rand((n, p["N"] * (p["A"] - 1))), p=1, dim=1
    )
    # sample discrete states
    d_sample = torch.randint(
        low=0,
        high=discrete_state_dim,
        size=torch.Size([K_sample.shape[0], 1]),
        dtype=torch.float64,
    )

    return torch.cat(
        (K_sample, w_sample, d_sample),
        dim=1,
    )


# ======================================================================
class SpecifiedModel(DPGPIpoptModel):
    def __init__(self, cfg={}, **kwargs):
        discrete_state_dim = cfg["model"]["params"]["N"] * cfg["model"]["params"]["Z"]

        policy_names = []
        policy_names += [f"Kpy_{i+1}" for i in range(cfg["model"]["params"]["N"])]
        policy_names += [f"cy_{i+1}_1" for i in range(cfg["model"]["params"]["N"])]
        policy_names += ["cy_sum"]
        policy_names_raw = policy_names.copy()
        self.policy_names_raw = policy_names_raw
        policy_names += [
            f"Vy_{i+1}_{j+1}"
            for i in range(cfg["model"]["params"]["N"])
            for j in range(1, cfg["model"]["params"]["A"])
        ]

        # dimension without "defintions", the one that ipopt uses
        self.policy_raw_dim = len([x for x in policy_names if not x.startswith("Vy_")])

        state_names = [f"Kx_{i+1}" for i in range(cfg["model"]["params"]["N"])]
        state_names += [
            f"wx_{i+1}_{j+1}"
            for i in range(cfg["model"]["params"]["N"])
            for j in range(cfg["model"]["params"]["A"] - 1)
        ]
        state_names += ["Z"]

        if "state_sample" in kwargs:
            init_state = kwargs.pop("state_sample")
            V_sample = kwargs.pop("V_sample")
        else:
            init_state = generate_sample(cfg, discrete_state_dim)
            V_sample = torch.zeros(init_state.shape[0])

        # for faster indexing
        self.c_mask_first = torch.tensor(
            [x.startswith("cy_") and x.endswith("_1") for x in policy_names_raw]
        )
        self.w_mask = torch.tensor([x.startswith("wx_") for x in state_names])
        self._K_mask_policy_raw = torch.tensor(
            [x.startswith("Kpy_") for x in policy_names_raw]
        )
        self.K_mask_policy = torch.tensor([x.startswith("Kpy_") for x in policy_names])
        self.K_mask_state = torch.tensor([x.startswith("Kx_") for x in state_names])

        super().__init__(
            cfg=cfg,
            policy_dim=len(policy_names),
            discrete_state_dim=discrete_state_dim,
            state_sample=init_state,
            V_sample=V_sample,
            policy_names=policy_names,
            state_names=state_names,
            control_dim=2 * (cfg["model"]["params"]["N"]) + 1,
            **kwargs,
        )
        self.constraint_dim = self.control_dim

    # no value-function iteration
    def E_V(self, state, control):
        return torch.sum(control) * torch.zeros(1)

    def sample(self):
        self.state_sample = generate_sample(self.cfg, self.discrete_state_dim)

    def F(self, xi, K, L):
        return (
            xi
            * K ** (self.cfg["model"]["params"]["alpha"])
            * L ** (1 - self.cfg["model"]["params"]["alpha"])
            + (1 - self.cfg["model"]["params"]["delta"]) * K
        )

    def F_K(self, xi, K, L):
        return xi * self.cfg["model"]["params"]["alpha"] * (K / L) ** (
            self.cfg["model"]["params"]["alpha"] - 1
        ) + (1 - self.cfg["model"]["params"]["delta"])

    def F_L(self, xi, K, L):
        return (
            xi
            * (K / L) ** (self.cfg["model"]["params"]["alpha"])
            * (1 - self.cfg["model"]["params"]["alpha"])
        )

    # use eq (6) as definition
    def c_from_sum(self, state, control, i, j):
        return state[self.S[f"wx_{i}_{j-1}"]] * control[self.P["cy_sum"]]

    def state_next(self, state, control, zpy):
        """Return next periods states, given the controls of today and the random discrete realization"""
        s = state.clone()
        # update discrete state
        s[self.S["Z"]] = zpy
        # update weights
        c_sum_old = torch.sum(control[self.c_mask_first]) + control[self.P["cy_sum"]]
        for i in range(self.cfg["model"]["params"]["N"]):
            c_sum_old -= self.c_from_sum(
                state, control, i + 1, self.cfg["model"]["params"]["A"]
            )

        for i in range(self.cfg["model"]["params"]["N"]):
            s[self.S[f"wx_{i+1}_1"]] = control[self.P[f"cy_{i+1}_1"]] / c_sum_old

        for i in range(self.cfg["model"]["params"]["N"]):
            for j in range(1, self.cfg["model"]["params"]["A"] - 1):
                s[self.S[f"wx_{i+1}_{j+1}"]] = (
                    self.c_from_sum(state, control, i + 1, j + 1) / c_sum_old
                )

        # update capital
        s[self.K_mask_state] = control[self._K_mask_policy_raw]
        return s

    def pAS(self, state, control, zpy):
        z = state[self.S["Z"]]
        par = self.cfg["model"]["params"]
        res = par["beta"] * par["pi"][int(z)][zpy]

        n = torch.tensor(0.0)
        dn = torch.tensor(0.0)

        state_next = torch.unsqueeze(self.state_next(state, control, zpy)[:-1], dim=0)

        for i in range(par["N"]):
            n += self.F(
                xi=par["tfp"][zpy][i],
                K=control[self.P[f"Kpy_{i+1}"]],
                L=par["L"][i],
            )
            n -= self.M[int(zpy)][1 + self.P[f"Kpy_{i+1}"]](state_next).mean[0]
            n -= self.M[int(zpy)][1 + self.P[f"cy_{i+1}_1"]](state_next).mean[0]

            dn += control[self.P[f"cy_{i+1}_{1}"]]
            dn -= self.c_from_sum(state, control, i + 1, par["A"])

        dn += control[self.P["cy_sum"]]
        return res * (n / dn) ** (-par["gamma"])

    def eval_g(self, state, control):
        par = self.cfg["model"]["params"]
        res = torch.zeros(self.constraint_dim)

        eq = 0
        # Eq 1
        for i in range(par["N"]):
            res[eq] += self.F(
                xi=par["tfp"][int(state[self.S["Z"]].item())][i],
                K=state[self.S[f"Kx_{i+1}"]],
                L=par["L"][i],
            )
            res[eq] -= control[self.P[f"Kpy_{i+1}"]]
            res[eq] -= control[self.P[f"cy_{i+1}_{1}"]]
        res[eq] -= control[self.P["cy_sum"]]

        eq += 1
        # Eq 2
        for i in range(par["N"]):
            res[eq] = torch.tensor(1)
            for z in range(self.discrete_state_dim):
                res[eq] -= self.pAS(state, control, z) * self.F_K(
                    xi=par["tfp"][int(z)][i],
                    K=control[self.P[f"Kpy_{i+1}"]],
                    L=par["L"][i],
                )
            eq += 1

        # Eq 3
        for i in range(par["N"]):
            res[eq] = -control[self.P[f"cy_{i+1}_1"]]
            res[eq] += par["l"][0] * self.F_L(
                par["tfp"][int(state[self.S["Z"]].item())][i],
                K=state[self.S[f"Kx_{i+1}"]],
                L=par["L"][i],
            )
            for z in range(self.discrete_state_dim):
                res[eq] -= (
                    self.pAS(state, control, z)
                    * self.M[z][1 + self.P[f"Vy_{i+1}_2"]](
                        torch.unsqueeze(self.state_next(state, control, z)[:-1], dim=0)
                    ).mean[0]
                )

            eq += 1

        return res

    def lb(self, state):
        X_L = np.zeros(self.policy_raw_dim)
        return X_L

    def ub(self, state):
        X_U = np.zeros(self.policy_raw_dim)
        X_U[self._K_mask_policy_raw.numpy()] = 100
        X_U[self.c_mask_first] = 10
        X_U[self.P["cy_sum"]] = (
            self.cfg["model"]["params"]["N"] * self.cfg["model"]["params"]["A"] * 10
        )
        return X_U

    def cl(self, state):
        return np.zeros(self.constraint_dim)

    def cu(self, state):
        return self.cl(state)

    def policy_guess(self, state):
        par = self.cfg["model"]["params"]
        x0 = torch.zeros(self.policy_dim)
        x0[self.K_mask_policy] = state[self.K_mask_state]

        c_exc = 0
        for i in range(par["N"]):
            x0[self.P[f"cy_{i+1}_1"]] = par["l"][i] * self.F_L(
                par["tfp"][int(state[self.S["Z"]].item())][i],
                K=state[self.S[f"Kx_{i+1}"]],
                L=par["L"][i],
            )
            c_exc += (
                self.F(
                    par["tfp"][int(state[self.S["Z"]].item())][i],
                    K=state[self.S[f"Kx_{i+1}"]],
                    L=par["L"][i],
                )
                - x0[self.P[f"cy_{i+1}_1"]]
                - x0[self.P[f"Kpy_{i+1}"]]
            )

        x0[self.P["cy_sum"]] = c_exc

        return x0

    def x_init(self, state):
        x0 = self.policy_guess(state)[: self.policy_raw_dim].detach().numpy()
        for p in self.policy_names_raw:
            x0[self.P[p]] = self.M[int(state[-1].item())][1 + self.P[p]](
                torch.unsqueeze(state[:-1], dim=0)
            ).mean.item()
        return x0

    def solve(self, state):
        par = self.cfg["model"]["params"]
        obj_val, x_raw = super().solve(state)
        # add definitions of Vy_...
        x = torch.zeros(self.policy_dim)
        x[: self.policy_raw_dim] = torch.from_numpy(x_raw)

        # Eq 4
        for i in range(par["N"]):
            x[self.P[f"Vy_{i+1}_{par['A']}"]] = self.c_from_sum(
                state, torch.from_numpy(x_raw), i + 1, par["A"]
            ) - par["l"][-1] * self.F_L(
                par["tfp"][int(state[self.S["Z"]].item())][i],
                K=state[self.S[f"Kx_{i+1}"]],
                L=par["L"][i],
            )

        # Eq 5
        for i in range(par["N"]):
            for j in range(1, par["A"] - 1):
                x[self.P[f"Vy_{i+1}_{j+1}"]] = (
                    self.c_from_sum(state, torch.from_numpy(x_raw), i + 1, j + 1)
                ) - par["l"][j] * self.F_L(
                    par["tfp"][int(state[self.S["Z"]].item())][i],
                    K=state[self.S[f"Kx_{i+1}"]],
                    L=par["L"][i],
                )
                for z in range(self.discrete_state_dim):
                    x[self.P[f"Vy_{i+1}_{j+1}"]] += (
                        self.pAS(state, torch.from_numpy(x_raw), z)
                        * self.M[z][1 + self.P[f"Vy_{i+1}_{j+2}"]](
                            torch.unsqueeze(
                                self.state_next(state, torch.from_numpy(x_raw), z)[:-1],
                                dim=0,
                            )
                        ).mean.item()
                    )

        return obj_val, x.detach().numpy()

    def state_iterate_exp(self, state, control):
        weights = torch.tensor(self.cfg["model"]["params"]["pi"][int(state[-1].item())])
        points = torch.cat(
            tuple(
                torch.unsqueeze(self.state_next(state, control, z), dim=0)
                for z in range(self.discrete_state_dim)
            ),
            dim=0,
        )
        return weights, points
