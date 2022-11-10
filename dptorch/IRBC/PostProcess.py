import torch
import gpytorch
import matplotlib
import logging
import pandas as pd

matplotlib.use("Agg")
from matplotlib import pyplot as plt

logger = logging.getLogger(__name__)


def process(m, cfg):
    # get median of states
    med_states = (
        torch.mean(m.state_sample[:,:-1], 0, keepdim=True)
        .expand(100, m.state_sample.shape[1]-1)
        .clone()
    )
    med_states[:, 0] = torch.linspace(
        torch.min(m.state_sample[:, 0]), torch.max(m.state_sample[:, 0]), steps=100
    )

    for d in range(m.discrete_state_dim):
        m.M[d][0].eval()
        m.likelihood[d][0].eval()
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            pred = m.likelihood[d][0](m.M[d][0](med_states.to(m.device)))

            V_pred = pred.mean.numpy()
            lower, upper = pred.confidence_region()

        f, ax = plt.subplots(1, 1, figsize=(12, 9))
        ax.plot(med_states[:, 0].numpy(), V_pred, "b")
        # Shade between the lower and upper confidence bounds
        ax.fill_between(
            med_states[:, 0].numpy(),
            lower.numpy(),
            upper.numpy(),
            alpha=0.5,
        )

        ax.legend(["VF Mean", "VF Confidence"])
        ax.set_xlabel("State 0")

        plt.savefig(f"V_fun_{str(d)}.png")

        conv = pd.DataFrame(
            {
                "epoch": [int(key) for key, value in m.metrics.items()],
                "l2": [value["l2"].item() for key, value in m.metrics.items()],
                "l_inf": [value["l_inf"].item() for key, value in m.metrics.items()],
            }
        )
        conv.plot(x="epoch", y=["l2", "l_inf"], logy=True)
        plt.savefig(f"Conv_metrics_{str(d)}.png")

        # now plot policy
        for p in range(m.policy_dim):
            m.M[d][1 + p].eval()
            m.likelihood[d][1 + p].eval()
            with torch.no_grad(), gpytorch.settings.fast_pred_var():
                pred = m.likelihood[d][1 + p](m.M[0][1 + p](med_states.to(m.device)))
                P_pred = pred.mean.numpy()
                lower, upper = pred.confidence_region()

            f, ax = plt.subplots(1, 1, figsize=(12, 9))
            ax.plot(med_states[:, 0].numpy(), P_pred, "b")
            # Shade between the lower and upper confidence bounds
            ax.fill_between(
                med_states[:, 0].numpy(),
                lower.numpy(),
                upper.numpy(),
                alpha=0.5,
            )

            ax.legend([f"P[{m._policy_names[p]}] Mean", f"P[{m._policy_names[p]}] Confidence"])
            ax.set_xlabel("State 0")

            plt.savefig(f"P_{d}_{m._policy_names[p]}_fun.png")
            plt.close()
    # run trajectories
    state_trajectory, policy_trajectory, eval_g_trajectory = m.generate_trajectory(100, from_sample=False)

    logger.info("Starting trajectory generation...")
    pd.DataFrame(torch.cat([torch.unsqueeze(x,dim=0) for x in state_trajectory]).numpy()).to_csv('state_trajectory.csv')
    pd.DataFrame(torch.cat([torch.unsqueeze(x,dim=0) for x in policy_trajectory]).numpy()).to_csv('policy_trajectory.csv')
    pd.DataFrame(torch.cat([torch.unsqueeze(x,dim=0) for x in eval_g_trajectory]).numpy()).to_csv('eval_g_trajectory.csv')

def compare(m1, m2, cfg):
    """Compare two GP-s"""
    col_idx = cfg.get("convergence_check_index", 0)
    state_sample = m2.state_sample[:,:-1]
    m1.M[0][col_idx].eval()
    m1.likelihood[0][col_idx].eval()
    m2.M[0][col_idx].eval()
    m2.likelihood[0][col_idx].eval()
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        p1 = m1.likelihood[0][col_idx](m1.M[0][col_idx](state_sample))
        V_p1 = p1.mean
        p2 = m2.likelihood[0][col_idx](m2.M[0][col_idx](state_sample))
        V_p2 = p2.mean

    logger.info(
        f"Difference in prediction (norm): L2: {1 / V_p1.shape[0]**0.5 * torch.linalg.norm(V_p1 - V_p2)}; L_inf: {torch.linalg.norm(V_p1 - V_p2,ord=float('inf'))}"
    )
    logger.info(
        f"Difference in T+1 iteration vs interpolated: L2: {1 / V_p1.shape[0]**0.5 * torch.linalg.norm(V_p2 - m2.V_sample)}; L_inf: {torch.linalg.norm(V_p2 - m2.V_sample,ord=float('inf'))}"
    )
