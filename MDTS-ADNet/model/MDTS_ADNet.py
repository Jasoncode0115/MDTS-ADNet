import torch
import torch.nn as nn
import torch.nn.functional as F
from .RGB import RGB_convAE
from .Flow import OF_convAE

class TwoStreamModel(nn.Module):
    def __init__(self, rgb_channels=3, of_channels=3, t_length=2, latent_dim=128):
        super(TwoStreamModel, self).__init__()
        self.rgb_model = RGB_convAE(n_channel=rgb_channels, t_length=t_length, latent_dim=latent_dim)
        self.of_model = OF_convAE(n_channel=of_channels, t_length=t_length, latent_dim=latent_dim)

        self.fusion_weights = nn.Parameter(torch.ones(2))
        self.weight_rgb = nn.Parameter(torch.tensor(0.5))
        self.softmax = nn.Softmax(dim=0)

    def forward(self, rgb_frames, of_frames):
        rgb_output, rgb_mu, rgb_logvar = self.rgb_model(rgb_frames)
        of_output, of_mu, of_logvar = self.of_model(of_frames)

        rgb_reconstruction_error = F.mse_loss(rgb_output, rgb_frames, reduction='none')
        of_reconstruction_error = F.mse_loss(of_output, of_frames, reduction='none')

        weights = self.softmax(self.fusion_weights)
        rgb_weight, of_weight = weights[0], weights[1]

        combined_error = self.weight_rgb * rgb_reconstruction_error + (1 - self.weight_rgb) * of_reconstruction_error

        return combined_error, rgb_output, of_output, rgb_mu, rgb_logvar, of_mu, of_logvar, weights, self.weight_rgb

