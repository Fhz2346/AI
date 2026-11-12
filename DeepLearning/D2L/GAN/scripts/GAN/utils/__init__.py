import torch
from torch import nn
from d2l import torch as d2l

from Pathes import P
from .save_load import make_models, save_checkpoint, save_history


def build_models(cfg):
    return make_models(int(cfg.train.latent_dim), int(cfg.data.dim_feature))


def get_loader(cfg):
    data = torch.load(P.data_dir / "data_gen.pkl")
    return d2l.load_array((data,), cfg.data_loader.batch_size)


def update_D(X, Z, net_D, net_G, loss, trainer_D):
    """Update discriminator."""
    batch_size = X.shape[0]
    ones = torch.ones((batch_size,), device=X.device)
    zeros = torch.zeros((batch_size,), device=X.device)
    trainer_D.zero_grad()
    real_Y = net_D(X)
    fake_X = net_G(Z)
    fake_Y = net_D(fake_X.detach())
    loss_D = (
        loss(real_Y, ones.reshape(real_Y.shape))
        + loss(fake_Y, zeros.reshape(fake_Y.shape))
    ) / 2
    loss_D.backward()
    trainer_D.step()
    return loss_D


def update_G(Z, net_D, net_G, loss, trainer_G):
    """Update generator."""
    batch_size = Z.shape[0]
    ones = torch.ones((batch_size,), device=Z.device)
    trainer_G.zero_grad()
    fake_X = net_G(Z)
    fake_Y = net_D(fake_X)
    loss_G = loss(fake_Y, ones.reshape(fake_Y.shape))
    loss_G.backward()
    trainer_G.step()
    return loss_G


def train(net_D, net_G, data_iter, cfg):
    loss = nn.BCEWithLogitsLoss(reduction="sum")
    for w in net_D.parameters():
        nn.init.normal_(w, 0, 0.02)
    for w in net_G.parameters():
        nn.init.normal_(w, 0, 0.02)
    trainer_D = torch.optim.Adam(net_D.parameters(), lr=cfg.train.lr_D)
    trainer_G = torch.optim.Adam(net_G.parameters(), lr=cfg.train.lr_G)
    history = {"epoch": [], "loss_D": [], "loss_G": []}
    num_epochs = cfg.train.num_epochs
    ckpt_every = cfg.train.ckpt_every
    latent_dim = cfg.train.latent_dim

    for epoch in range(1, num_epochs + 1):
        timer = d2l.Timer()
        metric = d2l.Accumulator(3)
        for (X,) in data_iter:
            batch_size = X.shape[0]
            Z = torch.normal(0, 1, size=(batch_size, latent_dim))
            metric.add(
                update_D(X, Z, net_D, net_G, loss, trainer_D),
                update_G(Z, net_D, net_G, loss, trainer_G),
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
            f"{speed:.1f} examples/sec"
        )
        if epoch % ckpt_every == 0 or epoch == num_epochs:
            save_checkpoint(epoch, net_G, net_D, cfg, loss_D, loss_G)
