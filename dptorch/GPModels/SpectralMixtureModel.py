import torch
import gpytorch
from .ConfigKernel import ConfigKernel


class GPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, cfg, batch_shape=torch.Size([])):
        super(GPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ZeroMean(batch_shape=batch_shape)

        self.covar_module = gpytorch.kernels.SpectralMixtureKernel(num_mixtures=4,ard_num_dims=train_x.shape[-1])
        self.covar_module.initialize_from_data(train_x, train_y)
        self.batch_shape = batch_shape

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        if not self.batch_shape:
            return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
        else:
            return gpytorch.distributions.MultitaskMultivariateNormal.from_batch_mvn(
                gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
            )
