import torch
from d2l import torch as d2l
from omegaconf import OmegaConf
from pathes import CKPT_DIR, HISTORY_PATH, OUT_DIR
from save_load import load_history, list_checkpoints, load_checkpoint, ckpt_path


def sample_grid(net_G, latent_dim, device, n_row=2, n_col=2, seed=0):
    net_G.eval()
    n = n_row * n_col
    generator = torch.Generator()
    generator.manual_seed(seed)
    Z = torch.randn(n, latent_dim, 1, 1, generator=generator, device="cpu").to(device)
    with torch.no_grad():
        fake_x = (net_G(Z).permute(0, 2, 3, 1) / 2 + 0.5).clamp(0, 1).cpu()
    rows = [
        torch.cat([fake_x[i * n_col + j] for j in range(n_col)], dim=1)
        for i in range(n_row)
    ]
    return torch.cat(rows, dim=0)


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


def visualize_generate(epoch=None, device=None, n_row=None, n_col=None):
    cfg = OmegaConf.load(P.cfg_dir / "DCGAN.yaml")
    n_row = n_row or int(cfg.visualize.n_row)
    n_col = n_col or int(cfg.visualize.n_col)
    device = device or d2l.try_gpu()
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
    cell_w, cell_h = 0.9 * n_col, 0.9 * n_row
    fig, axes = d2l.plt.subplots(
        n_fig_row,
        n_fig_col,
        figsize=(cell_w * n_fig_col, cell_h * n_fig_row),
    )
    if n == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for i, path in enumerate(paths):
        ckpt, net_G, _ = load_checkpoint(path, device)
        imgs = sample_grid(
            net_G, ckpt["latent_dim"], device, n_row=n_row, n_col=n_col
        )
        img_path = OUT_DIR / f"epoch_{ckpt['epoch']:03d}.png"
        d2l.plt.imsave(img_path, imgs.numpy())
        axes[i].imshow(imgs.numpy())
        axes[i].set_title(f"epoch {ckpt['epoch']}")
        axes[i].axis("off")
        print(f"loaded {path}")
    for ax in axes[n:]:
        ax.axis("off")
    _save_and_show(fig, "generate.png")
