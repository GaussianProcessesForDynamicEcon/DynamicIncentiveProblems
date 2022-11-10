import gpytorch
import torch


def ConfigKernel(cfg, x_dim, batch_shape=torch.Size([])):
    kernel_config = cfg["kernel"].get("config")
    kernel_config.update({"batch_shape": batch_shape})
    kernel_config.update({"ard_num_dims": x_dim})
    # kernel_config.update({"lengthscale_constraint": gpytorch.constraints.GreaterThan(0.10)})
    return getattr(gpytorch.kernels, cfg["kernel"].get("name"))(
        **kernel_config
    )
