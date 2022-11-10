import torch

class ExpectOps:
    def __init__(self, cfg, device):
        self.cfg = cfg
        self.device = device
        
    def integration_rule(self, state):
        weights = torch.tensor([1], device=self.device)
        points = torch.unsqueeze(state.to(self.device), 0)
        return (weights, points)