import torch
import math


class ExpectOps:
    """Calculate conditional expectations using standard normal random shocks"""

    def __init__(self, cfg, device):
        self.cfg = cfg
        self.device = device

    def integration_rule(self, state):
        std = torch.tensor(self.cfg["model"]["EXPECT_OP"]["config"]["mc_std"], device=self.device)
        d = state.shape[0] - 1

        return (
            torch.tensor([1 / (2 * d)] * (2 * d), device=self.device),
            torch.cat(
                (
                    torch.unsqueeze(state[:-1].to(self.device), 0).expand(2 * d, -1)
                    + torch.cat(
                        (
                            torch.diag(std) * math.sqrt(d / 2.0),
                            torch.diag(std) * -math.sqrt(d / 2.0),
                        )
                    ),
                    torch.full((2*d, 1), state[-1].item()),
                ),
                dim=1,
            ),
        )
