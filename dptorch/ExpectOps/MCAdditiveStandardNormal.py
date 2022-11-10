import torch


class ExpectOps:
    """Calculate conditional expectations using standard normal random shocks"""

    def __init__(self, cfg, device):
        self.cfg = cfg
        self.device = device

    def integration_rule(self, state):
        gen = torch.Generator(device=self.device)
        # make sure that for each state we will always generate the same randoms
        gen.manual_seed(self.cfg["mc_seed"])
        std = torch.tensor(
            self.cfg["model"]["EXPECT_OP"]["config"]["mc_std"], device=self.device
        ).expand((self.cfg["model"]["EXPECT_OP"]["config"].get("n_mc_samples", 10), -1))

        return (
            torch.tensor([1 / std.shape[0]] * std.shape[0], device=self.device),
            torch.cat(
                (
                    torch.unsqueeze(state[:-1].to(self.device), 0).expand(
                        std.shape[0], -1
                    )
                    + torch.normal(
                        torch.zeros_like(std, device=self.device),
                        std=std,
                        generator=gen,
                    ),
                    torch.full((std.shape[0], 1), state[-1].item()),
                ),
                dim=1,
            ),
        )
