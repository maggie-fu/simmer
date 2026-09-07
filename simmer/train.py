"""Training loop for the SimMeR cVAE.

Implements the stabilization practices named in the proposal: KL annealing to
avoid posterior collapse, gradient clipping, and early stopping on a held-out
set. Standardizes CpG M-values before fitting (stored on the returned bundle so
generation can invert it).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from .data import DNAmDataset
from .model import ConditionalVAE, vae_loss
from .phenotypes import condition_dim, encode_phenotypes


@dataclass
class TrainConfig:
    latent_dim: int = 32
    hidden_dims: Tuple[int, ...] = (256, 128)
    dropout: float = 0.1
    batch_size: int = 128
    epochs: int = 60
    lr: float = 1e-3
    weight_decay: float = 1e-5
    kl_max: float = 1.0
    kl_anneal_epochs: int = 25  # linearly ramp KL weight 0 -> kl_max over this many epochs
    grad_clip: float = 5.0
    patience: int = 12
    seed: int = 0
    device: str = "cpu"


@dataclass
class Scaler:
    mean: np.ndarray
    std: np.ndarray

    def transform(self, x: np.ndarray) -> np.ndarray:
        return (x - self.mean) / self.std

    def inverse(self, x: np.ndarray) -> np.ndarray:
        return x * self.std + self.mean


@dataclass
class TrainedModel:
    model: ConditionalVAE
    scaler: Scaler
    age_mean: float
    age_sd: float
    history: Dict[str, List[float]]
    config: TrainConfig


def _kl_weight(epoch: int, cfg: TrainConfig) -> float:
    if cfg.kl_anneal_epochs <= 0:
        return cfg.kl_max
    return cfg.kl_max * min(1.0, (epoch + 1) / cfg.kl_anneal_epochs)


def _make_loader(
    ds: DNAmDataset, scaler: Scaler, age_mean: float, age_sd: float, cfg: TrainConfig, shuffle: bool
) -> DataLoader:
    x = torch.from_numpy(scaler.transform(ds.m_values).astype(np.float32))
    c = torch.from_numpy(encode_phenotypes(ds.phenotypes, age_mean, age_sd))
    g = torch.Generator().manual_seed(cfg.seed)
    return DataLoader(TensorDataset(x, c), batch_size=cfg.batch_size, shuffle=shuffle, generator=g)


def train_cvae(
    train_ds: DNAmDataset,
    val_ds: Optional[DNAmDataset] = None,
    cfg: Optional[TrainConfig] = None,
    age_mean: float = 44.6,
    age_sd: float = 18.8,
    verbose: bool = True,
) -> TrainedModel:
    cfg = cfg or TrainConfig()
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    device = torch.device(cfg.device)

    scaler = Scaler(
        mean=train_ds.m_values.mean(axis=0),
        std=train_ds.m_values.std(axis=0) + 1e-6,
    )

    train_loader = _make_loader(train_ds, scaler, age_mean, age_sd, cfg, shuffle=True)
    val_loader = (
        _make_loader(val_ds, scaler, age_mean, age_sd, cfg, shuffle=False)
        if val_ds is not None
        else None
    )

    model = ConditionalVAE(
        n_cpgs=train_ds.n_cpgs,
        cond_dim=condition_dim(),
        latent_dim=cfg.latent_dim,
        hidden_dims=cfg.hidden_dims,
        dropout=cfg.dropout,
    ).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    history: Dict[str, List[float]] = {"train_total": [], "train_recon": [], "train_kl": [], "val_recon": []}
    best_val = float("inf")
    best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    bad_epochs = 0

    for epoch in range(cfg.epochs):
        model.train()
        klw = _kl_weight(epoch, cfg)
        tot = recon_sum = kl_sum = 0.0
        n_batches = 0
        for xb, cb in train_loader:
            xb, cb = xb.to(device), cb.to(device)
            opt.zero_grad()
            recon, mu, logvar = model(xb, cb)
            loss, recon_loss, kld = vae_loss(recon, xb, mu, logvar, klw)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            opt.step()
            tot += loss.item(); recon_sum += recon_loss.item(); kl_sum += kld.item(); n_batches += 1

        history["train_total"].append(tot / n_batches)
        history["train_recon"].append(recon_sum / n_batches)
        history["train_kl"].append(kl_sum / n_batches)

        val_recon = float("nan")
        if val_loader is not None:
            model.eval()
            with torch.no_grad():
                vr = 0.0; vb = 0
                for xb, cb in val_loader:
                    xb, cb = xb.to(device), cb.to(device)
                    recon, mu, logvar = model(xb, cb)
                    _, recon_loss, _ = vae_loss(recon, xb, mu, logvar, klw)
                    vr += recon_loss.item(); vb += 1
                val_recon = vr / vb
            history["val_recon"].append(val_recon)

            if val_recon < best_val - 1e-4:
                best_val = val_recon
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
                bad_epochs = 0
            else:
                bad_epochs += 1

        if verbose and (epoch % 5 == 0 or epoch == cfg.epochs - 1):
            print(
                f"epoch {epoch:3d} | klw {klw:.2f} | train_recon {history['train_recon'][-1]:8.3f}"
                f" | train_kl {history['train_kl'][-1]:7.3f} | val_recon {val_recon:8.3f}"
            )

        if val_loader is not None and bad_epochs >= cfg.patience:
            if verbose:
                print(f"early stopping at epoch {epoch} (best val_recon {best_val:.3f})")
            break

    model.load_state_dict(best_state)
    return TrainedModel(
        model=model,
        scaler=scaler,
        age_mean=age_mean,
        age_sd=age_sd,
        history=history,
        config=cfg,
    )
