"""Conditional variational autoencoder (cVAE) for DNAm generation.

Follows the methCancer-gen-style baseline described in the proposal: the
phenotype/conditioning vector is concatenated to *both* the encoder and decoder
inputs, so the latent code captures residual epigenetic structure while
generation is steered by demographics and cell-type composition.

Kept deliberately small (a couple of dense layers) so it trains on CPU in
minutes on the prototype's ~400-CpG stand-in. The interface generalizes to the
genome-wide, block-structured encoders in the grant without changing callers.
"""

from __future__ import annotations

from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConditionalVAE(nn.Module):
    def __init__(
        self,
        n_cpgs: int,
        cond_dim: int,
        latent_dim: int = 32,
        hidden_dims: Tuple[int, ...] = (256, 128),
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.n_cpgs = n_cpgs
        self.cond_dim = cond_dim
        self.latent_dim = latent_dim

        # --- encoder: [DNAm ; phenotype] -> latent params ---
        enc_layers: List[nn.Module] = []
        in_dim = n_cpgs + cond_dim
        for h in hidden_dims:
            enc_layers += [nn.Linear(in_dim, h), nn.LayerNorm(h), nn.ReLU(), nn.Dropout(dropout)]
            in_dim = h
        self.encoder = nn.Sequential(*enc_layers)
        self.fc_mu = nn.Linear(in_dim, latent_dim)
        self.fc_logvar = nn.Linear(in_dim, latent_dim)

        # --- decoder: [latent ; phenotype] -> DNAm reconstruction ---
        dec_layers: List[nn.Module] = []
        in_dim = latent_dim + cond_dim
        for h in reversed(hidden_dims):
            dec_layers += [nn.Linear(in_dim, h), nn.LayerNorm(h), nn.ReLU(), nn.Dropout(dropout)]
            in_dim = h
        self.decoder = nn.Sequential(*dec_layers)
        self.fc_out = nn.Linear(in_dim, n_cpgs)

    def encode(self, x: torch.Tensor, c: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.encoder(torch.cat([x, c], dim=1))
        return self.fc_mu(h), self.fc_logvar(h)

    @staticmethod
    def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        h = self.decoder(torch.cat([z, c], dim=1))
        return self.fc_out(h)

    def forward(
        self, x: torch.Tensor, c: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(x, c)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z, c)
        return recon, mu, logvar

    @torch.no_grad()
    def generate(self, c: torch.Tensor, z: torch.Tensor | None = None) -> torch.Tensor:
        """Generate DNAm profiles conditioned on ``c``.

        Samples ``z`` from the standard-normal prior when not provided.
        """
        if z is None:
            z = torch.randn(c.shape[0], self.latent_dim, device=c.device)
        return self.decode(z, c)


def vae_loss(
    recon: torch.Tensor,
    target: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    kl_weight: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Reconstruction (MSE, summed over CpGs) + KL, with a KL anneal weight."""
    recon_loss = F.mse_loss(recon, target, reduction="none").sum(dim=1).mean()
    kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1).mean()
    total = recon_loss + kl_weight * kld
    return total, recon_loss, kld
