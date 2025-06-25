import torch
from torch import nn
import math

class Diffusion(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(Diffusion, self).__init__()

        self.layers = nn.Sequential(
            nn.Linear(input_dim * 2, input_dim),
            nn.LeakyReLU(inplace=True),
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(inplace=True),
            nn.Linear(hidden_dim, input_dim),
            nn.Tanh(),
        )

    def forward(self, x, prototype):
        # Concatenate the input with the prototype
        x = torch.cat((x, prototype), dim=-1)
        return self.layers(x)

    # Distort the input data by adding Gaussian noise
    def distort(self, x, epoch_percentage):
        # Clamp the epoch percentage so that we arent training on full x or full noise
        epoch_percentage = min(max(epoch_percentage, 0.01), 0.5)
        # Apply a logarithmic schedule to the noise
        log_epoch_percentage = math.log10(1 - 0.9 * epoch_percentage) / math.log10(0.1)        # Generate random noise
        noise = torch.randn_like(x)
        # Compute norms
        x_norm = x.norm(dim=1, keepdim=True)
        noise_norm = noise.norm(dim=1, keepdim=True)
        # Scale noise to have the same norm as x
        scaled_noise = noise / (noise_norm + 1e-8) * x_norm
        # Interpolate between x and scaled_noise using the log-scaled percentage
        return (1 - log_epoch_percentage) * x + log_epoch_percentage * scaled_noise