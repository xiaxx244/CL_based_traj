import torch
import torch.distributions as td
import numpy as np
from model.model_utils import to_one_hot

class GMM2D1(td.Distribution):
    """
    Gaussian Mixture Model in 2D, parameterized by means (mus) and covariance matrices (cov_mats).

    :param mus: Mixture component means [..., N, 2]
    :param cov_mats: Covariance matrices [..., N, 2, 2]
    :param log_pis: (Optional) Log mixing proportions [..., N]. Defaults to uniform mixing if not provided.
    """
    def __init__(self, mus, cov_mats, log_pis=None):
        # Derive parameters from covariance matrices
        sigmas = torch.sqrt(torch.stack([cov_mats[..., 0, 0], cov_mats[..., 1, 1]], dim=-1))  # [..., N, 2]
        corrs = cov_mats[..., 0, 1] / (sigmas[..., 0] * sigmas[..., 1])  # [..., N]
        log_sigmas = torch.log(sigmas)  # [..., N, 2]

        # Default log_pis to uniform mixing if not provided
        if log_pis is None:
            log_pis = torch.full(mus.shape[:-2] + (mus.shape[-2],), fill_value=-np.log(mus.shape[-2]), device=mus.device)

        super(GMM2D1, self).__init__(batch_shape=mus.shape[:-2], event_shape=(2,))
        self.components = mus.shape[-2]
        self.dimensions = 2
        self.device = mus.device

        log_pis = torch.clamp(log_pis, min=-1e5)
        self.log_pis = log_pis - torch.logsumexp(log_pis, dim=-1, keepdim=True)  # Normalize log mixing proportions
        self.mus = mus  # [..., N, 2]
        self.log_sigmas = log_sigmas  # [..., N, 2]
        self.sigmas = sigmas  # [..., N, 2]
        self.corrs = torch.clamp(corrs, min=-0.999, max=0.999)  # Prevent extreme values
        self.one_minus_rho2 = 1 - self.corrs**2
        self.one_minus_rho2 = torch.clamp(self.one_minus_rho2, min=1e-5, max=1)

        self.L = torch.stack([
            torch.stack([self.sigmas[..., 0], torch.zeros_like(self.sigmas[..., 0])], dim=-1),
            torch.stack([self.sigmas[..., 1] * self.corrs,
                         self.sigmas[..., 1] * torch.sqrt(self.one_minus_rho2)], dim=-1)
        ], dim=-2)

        self.pis_cat_dist = td.Categorical(logits=log_pis)

    def rsample(self, sample_shape=torch.Size()):
        mvn_samples = (self.mus +
                       torch.squeeze(
                           torch.matmul(self.L,
                                        torch.unsqueeze(
                                            torch.randn(size=sample_shape + self.mus.shape, device=self.device),
                                            dim=-1)
                                        ),
                           dim=-1))
        component_cat_samples = self.pis_cat_dist.sample(sample_shape)
        selector = torch.unsqueeze(to_one_hot(component_cat_samples, self.components), dim=-1)
        return torch.sum(mvn_samples * selector, dim=-2)

    def log_prob(self, value):
        value = torch.unsqueeze(value, dim=-2)  # [..., 1, 2]
        dx = value - self.mus  # [..., N, 2]

        exp_nominator = ((torch.sum((dx / self.sigmas) ** 2, dim=-1)  # First two terms
                          - 2 * self.corrs * torch.prod(dx, dim=-1) / torch.prod(self.sigmas, dim=-1)))  # [..., N]

        component_log_p = -(2 * np.log(2 * np.pi)
                            + torch.log(self.one_minus_rho2)
                            + 2 * torch.sum(self.log_sigmas, dim=-1)
                            + exp_nominator / self.one_minus_rho2) / 2

        return torch.logsumexp(self.log_pis + component_log_p, dim=-1)

    def get_for_node_at_time(self, n, t):
        return self.__class__(self.mus[:, n:n+1, t:t+1],
                              self.get_covariance_matrix()[:, n:n+1, t:t+1],
                              self.log_pis[:, n:n+1, t:t+1])

    def mode(self):
        if self.mus.shape[-2] > 1:
            samp, bs, time, comp, _ = self.mus.shape
            assert samp == 1, "For taking the mode only one sample makes sense."
            mode_node_list = []
            for n in range(bs):
                mode_t_list = []
                for t in range(time):
                    nt_gmm = self.get_for_node_at_time(n, t)
                    x_min = self.mus[:, n, t, :, 0].min()
                    x_max = self.mus[:, n, t, :, 0].max()
                    y_min = self.mus[:, n, t, :, 1].min()
                    y_max = self.mus[:, n, t, :, 1].max()
                    search_grid = torch.stack(torch.meshgrid([torch.arange(x_min, x_max, 0.01),
                                                              torch.arange(y_min, y_max, 0.01)]), dim=2
                                              ).view(-1, 2).float().to(self.device)

                    ll_score = nt_gmm.log_prob(search_grid)
                    argmax = torch.argmax(ll_score.squeeze(), dim=0)
                    mode_t_list.append(search_grid[argmax])
                mode_node_list.append(torch.stack(mode_t_list, dim=0))
            return torch.stack(mode_node_list, dim=0).unsqueeze(dim=0)
        return torch.squeeze(self.mus, dim=-2)

    def reshape_to_components(self, tensor):
        if len(tensor.shape) == 5:
            return tensor
        return torch.reshape(tensor, list(tensor.shape[:-1]) + [self.components, self.dimensions])

    def get_covariance_matrix(self):
        cov = self.corrs * torch.prod(self.sigmas, dim=-1)
        E = torch.stack([torch.stack([self.sigmas[..., 0] ** 2, cov], dim=-1),
                         torch.stack([cov, self.sigmas[..., 1] ** 2], dim=-1)], dim=-2)
        return E
