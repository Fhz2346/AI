import json
import sys
from pathlib import Path

import torch
import torchvision
from omegaconf import OmegaConf
from torch import nn
from d2l import torch as d2l

from network import Discriminator, Generator
from save_load import save_checkpoint, save_history



def build_models(cfg):
    net_G = Generator(n_G=cfg.model.n_G, latent_dim=cfg.model.latent_dim)
    net_D = Discriminator(n_D=cfg.model.n_D)
    return net_G, net_D


def get_loader(cfg):
    dataset = torchvision.datasets.ImageFolder(cfg.data.root)
    dataset.transform = torchvision.transforms.Compose(
        [
            torchvision.transforms.Resize((cfg.data.image_size, cfg.data.image_size)),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(0.5, 0.5),
        ]
    )
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=cfg.data.batch_size,
        shuffle=True,
        num_workers=cfg.data.num_workers,
    )




def train(net_D, net_G, data_iter, cfg, device=None):
    device = device or d2l.try_gpu()
    loss = nn.BCEWithLogitsLoss(reduction="sum")
    for w in net_D.parameters():
        nn.init.normal_(w, 0, 0.02)
    for w in net_G.parameters():
        nn.init.normal_(w, 0, 0.02)
    net_D, net_G = net_D.to(device), net_G.to(device)
    trainer_hp = {"lr": cfg.train.lr, "betas": list(cfg.train.betas)}
    trainer_D = torch.optim.Adam(net_D.parameters(), **trainer_hp)
    trainer_G = torch.optim.Adam(net_G.parameters(), **trainer_hp)
    history = {"epoch": [], "loss_D": [], "loss_G": []}
    num_epochs = cfg.train.num_epochs
    ckpt_every = cfg.train.ckpt_every
    latent_dim = cfg.model.latent_dim

    for epoch in range(1, num_epochs + 1):
        timer = d2l.Timer()
        metric = d2l.Accumulator(3)
        net_D.train()
        net_G.train()
        for X, _ in data_iter:
            batch_size = X.shape[0]
            Z = torch.normal(0, 1, size=(batch_size, latent_dim, 1, 1))
            X, Z = X.to(device), Z.to(device)
            metric.add(
                d2l.update_D(X, Z, net_D, net_G, loss, trainer_D),
                d2l.update_G(Z, net_D, net_G, loss, trainer_G),
                batch_size,
            )
        loss_D = metric[0] / metric[2]
        loss_G = metric[1] / metric[2]
        speed = metric[2] / timer.stop()
        history["epoch"].append(epoch)
        history["loss_D"].append(float(loss_D))
        history["loss_G"].append(float(loss_G))
        save_history(history)
        print(
            f"epoch {epoch}/{num_epochs}  "
            f"loss_D {loss_D:.3f}, loss_G {loss_G:.3f}, "
            f"{speed:.1f} examples/sec on {device}"
        )
        if epoch % ckpt_every == 0 or epoch == num_epochs:
            save_checkpoint(epoch, net_G, net_D, cfg, loss_D, loss_G)

