import torch
import gpytorch
from .ConfigKernel import ConfigKernel


class LargeFeatureExtractor(torch.nn.Sequential):
    def __init__(self, input_dim):
        super(LargeFeatureExtractor, self).__init__()
        self.add_module('linear1', torch.nn.Linear(input_dim, 10))
        self.add_module('relu1', torch.nn.ReLU())
        self.add_module('linear2', torch.nn.Linear(10, 5))
        self.add_module('relu2', torch.nn.ReLU())
        self.add_module('linear3', torch.nn.Linear(5, 5))
        self.add_module('relu3', torch.nn.ReLU())
        self.add_module('linear4', torch.nn.Linear(5, 2))


class GPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, cfg, batch_shape=torch.Size([])):
        super(GPModel, self).__init__(train_x, train_y, likelihood)

        self.mean_module = gpytorch.means.ZeroMean(batch_shape=batch_shape)
        self.covar_module = gpytorch.kernels.GridInterpolationKernel(
            gpytorch.kernels.ScaleKernel(ConfigKernel(cfg, train_x.shape[-1],batch_shape),batch_shape=batch_shape),
            grid_size=100, num_dims=2,
        )

        # Also add the deep net
        self.feature_extractor = LargeFeatureExtractor(input_dim=train_x.size(-1))
        self.batch_shape=batch_shape

    def forward(self, x):
        # We're first putting our data through a deep net (feature extractor)
        # We're also scaling the features so that they're nice values
        projected_x = self.feature_extractor(x)
        projected_x = projected_x - projected_x.min(0)[0]
        projected_x = 2 * (projected_x / projected_x.max(0)[0]) - 1

        # The rest of this looks like what we've seen
        mean_x = self.mean_module(projected_x)
        covar_x = self.covar_module(projected_x)
        if not self.batch_shape:
            return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
        else:
            return gpytorch.distributions.MultitaskMultivariateNormal.from_batch_mvn(
                gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
            )
