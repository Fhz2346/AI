import torch
import numpy as np
import matplotlib.pyplot as plt
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from Network.CVAE import CVAE
from Pathes import P


def plot_latent_scatter(cvae, data_loader, figsize=6):
    cvae.eval()
    device = next(cvae.parameters()).device
    zs, ys = [], []
    with torch.no_grad():
        for x, y in data_loader:
            z_mean, _ = cvae.encoder(x.to(device))
            zs.append(z_mean.cpu())
            ys.append(y)
    z_mean = torch.cat(zs, dim=0).numpy()
    y = torch.cat(ys, dim=0).numpy()

    plt.figure(figsize=(figsize, figsize))
    scatter = plt.scatter(z_mean[:, 0], z_mean[:, 1], c=y, cmap="tab10", s=8)
    plt.colorbar(scatter)
    plt.xlabel("z[0]")
    plt.ylabel("z[1]")
    plt.title("CVAE latent space")
    P.out_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(P.out_dir / "cvae_latent.png", dpi=150)
    plt.show()


def plot_class_conditional(cvae, output_digit=9, n=15, figsize=10):
    """Decode a Gaussian-quantile grid centered at the class prior mean."""
    cvae.eval()
    device = next(cvae.parameters()).device
    digit_size = 28
    num_classes = cvae.class_mean.in_features

    with torch.no_grad():
        eye = torch.eye(num_classes, device=device)
        mu = cvae.class_mean(eye).cpu()

    quantiles = torch.special.ndtri(torch.linspace(0.05, 0.95, n))
    grid_z0 = quantiles + mu[output_digit, 0]
    grid_z1 = quantiles + mu[output_digit, 1]
    figure = np.zeros((digit_size * n, digit_size * n))

    with torch.no_grad():
        for i, z1 in enumerate(grid_z1):
            for j, z0 in enumerate(grid_z0):
                z = torch.tensor([[z0, z1]], dtype=torch.float32, device=device)
                decoded = cvae.decoder(z).cpu().numpy().reshape(digit_size, digit_size)
                figure[
                    i * digit_size : (i + 1) * digit_size,
                    j * digit_size : (j + 1) * digit_size,
                ] = decoded

    plt.figure(figsize=(figsize, figsize))
    plt.imshow(figure, cmap="Greys_r")
    plt.axis("off")
    plt.title(f"CVAE samples around class {output_digit}")
    P.out_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(P.out_dir / f"cvae_digit_{output_digit}.png", dpi=150)
    plt.show()


if __name__ == "__main__":
    cfg = OmegaConf.load("config/CVAE.yaml")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cvae = CVAE(cfg.network).to(device)
    cvae.load_state_dict(torch.load(P.ckpt_dir / "cvae.pth", map_location=device))

    test_set = datasets.MNIST(
        root=cfg.dataset.root, train=False, download=False, transform=transforms.ToTensor()
    )
    test_loader = DataLoader(test_set, batch_size=cfg.dataloader.batch_size, shuffle=False)
    plot_latent_scatter(cvae, test_loader)
    plot_class_conditional(cvae, output_digit=8)
