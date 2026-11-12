import torch
from d2l import torch as d2l
from omegaconf import OmegaConf

from Pathes import P
from .pathes import CKPT_DIR, HISTORY_PATH, OUT_DIR
from .save_load import ckpt_path, list_checkpoints, load_checkpoint, load_history


def sample_fake(net_G, latent_dim, n_samples, seed=0):
    net_G.eval()
    generator = torch.Generator().manual_seed(seed)
    Z = torch.randn(n_samples, latent_dim, generator=generator)
    with torch.no_grad():
        return net_G(Z).detach().numpy()


def _setup_script_plot():
    import matplotlib

    d2l.use_svg_display = lambda: None
    backend = matplotlib.get_backend().lower()
    if backend == "agg" or "inline" in backend:
        matplotlib.use("TkAgg", force=True)
    d2l.plt.ion()
    d2l.display.display = lambda *args, **kwargs: d2l.plt.pause(0.001)
    d2l.display.clear_output = lambda *args, **kwargs: None


def _save_and_show(fig, name):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig_path = OUT_DIR / name
    fig.tight_layout()
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    print(f"figure saved to {fig_path}")
    d2l.plt.ioff()
    d2l.plt.show()


def visualize_loss():
    history = load_history()
    if not history["epoch"]:
        raise FileNotFoundError(f"no loss history at {HISTORY_PATH}")
    _setup_script_plot()
    fig, ax = d2l.plt.subplots(figsize=(6, 4))
    ax.plot(history["epoch"], history["loss_D"], label="discriminator")
    ax.plot(history["epoch"], history["loss_G"], label="generator")
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss")
    ax.legend()
    ax.grid(True)
    _save_and_show(fig, "loss.png")


def visualize_generate(epoch=None, n_samples=None):
    cfg = OmegaConf.load(P.cfg_dir / "GAN.yaml")
    n_samples = n_samples or int(cfg.visualize.n_samples)
    data = torch.load(P.data_dir / "data_gen.pkl")
    real = data[:n_samples].detach().numpy()

    paths = list_checkpoints()
    if not paths:
        raise FileNotFoundError(f"no checkpoints in {CKPT_DIR}")
    if epoch is not None:
        path = ckpt_path(epoch)
        if path not in paths:
            raise FileNotFoundError(f"missing checkpoint {path}")
        paths = [path]

    _setup_script_plot()
    n = len(paths)
    n_fig_col = 2 if n > 1 else 1
    n_fig_row = (n + n_fig_col - 1) // n_fig_col
    fig, axes = d2l.plt.subplots(
        n_fig_row, n_fig_col, figsize=(5 * n_fig_col, 4 * n_fig_row)
    )
    if n == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for i, path in enumerate(paths):
        ckpt, net_G, _ = load_checkpoint(path)
        fake = sample_fake(net_G, ckpt["latent_dim"], n_samples)
        axes[i].scatter(real[:, 0], real[:, 1], label="real")
        axes[i].scatter(fake[:, 0], fake[:, 1], label="generated")
        axes[i].set_title(f"epoch {ckpt['epoch']}")
        axes[i].legend()

        fig_i, ax_i = d2l.plt.subplots(figsize=(5, 4))
        ax_i.scatter(real[:, 0], real[:, 1], label="real")
        ax_i.scatter(fake[:, 0], fake[:, 1], label="generated")
        ax_i.set_title(f"epoch {ckpt['epoch']}")
        ax_i.legend()
        img_path = OUT_DIR / f"epoch_{ckpt['epoch']:03d}.png"
        fig_i.tight_layout()
        fig_i.savefig(img_path, dpi=150, bbox_inches="tight")
        d2l.plt.close(fig_i)
        print(f"loaded {path}")
    for ax in axes[n:]:
        ax.axis("off")
    _save_and_show(fig, "generate.png")
