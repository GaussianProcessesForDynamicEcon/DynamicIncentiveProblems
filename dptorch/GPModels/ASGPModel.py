import torch
import gpytorch
from .ConfigKernel import ConfigKernel


class SubspaceProjector(torch.nn.Sequential):
    def __init__(self, input_dim, active_dim):
        super(SubspaceProjector, self).__init__()
        self.add_module('linear1', torch.nn.Linear(input_dim, active_dim))

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, torch.nn.Linear):
            # torch.nn.init.xavier_uniform_(module.weight)
            torch.nn.init.eye_(module.weight)
            # lecun_normal_(module.weight)
            if module.bias is not None:
                module.bias.data.zero_()

class GPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, cfg, batch_shape=torch.Size([])):
        super(GPModel, self).__init__(train_x, train_y, likelihood)

        self.mean_module = gpytorch.means.ZeroMean(batch_shape=batch_shape)
        self.covar_module = gpytorch.kernels.ScaleKernel(ConfigKernel(cfg, cfg['model']['GP_MODEL']['config']['active_dim'], batch_shape),batch_shape=batch_shape)

        # Also add the deep net
        self.feature_extractor = SubspaceProjector(input_dim=train_x.size(-1), active_dim=cfg['model']['GP_MODEL']['config']['active_dim'])

        self.batch_shape = batch_shape
        
    def forward(self, x):
        # We're first putting our data through a deep net (feature extractor)
        projected_x = self.feature_extractor(x)

        # The rest of this looks like what we've seen
        mean_x = self.mean_module(projected_x)
        covar_x = self.covar_module(projected_x)
        if not self.batch_shape:
            return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
        else:
            return gpytorch.distributions.MultitaskMultivariateNormal.from_batch_mvn(
                gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
            )
