import torch
import numpy as np
import matplotlib.pyplot as plt
from omegaconf import OmegaConf
from Network.VAE import VAE
from Pathes import P

def plot_latent_space(vae, n=30, figsize=15):
    vae.eval()  # set to eval mode
    device = next(vae.parameters()).device  # detect CPU/GPU

    digit_size = 28
    scale = 1.0
    figure = np.zeros((digit_size * n, digit_size * n))

    # Latent space grid
    grid_x = np.linspace(-scale, scale, n)
    grid_y = np.linspace(-scale, scale, n)[::-1]

    for i, yi in enumerate(grid_y):
        for j, xi in enumerate(grid_x):

            z_sample = torch.tensor([[xi, yi]], dtype=torch.float32).to(device)

            with torch.no_grad():
                decoded = vae.decoder(z_sample).cpu().numpy()

            digit = decoded[0, 0]  # (1, 1, 28, 28) → (28, 28)

            figure[
                i * digit_size:(i + 1) * digit_size,
                j * digit_size:(j + 1) * digit_size
            ] = digit

    # Plot
    plt.figure(figsize=(figsize, figsize))
    start_range = digit_size // 2
    end_range = n * digit_size + start_range
    pixel_range = np.arange(start_range, end_range, digit_size)

    plt.xticks(pixel_range, np.round(grid_x, 1))
    plt.yticks(pixel_range, np.round(grid_y, 1))
    plt.xlabel("z[0]")
    plt.ylabel("z[1]")
    plt.imshow(figure, cmap="Greys_r")
    plt.show()

if __name__ == "__main__":
    cfg = OmegaConf.load("config/MNIST.yaml")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    vae = VAE(cfg.network).to(device)
    vae.load_state_dict(torch.load(P.ckpt_dir / "vae_pytorch.pth"))
    plot_latent_space(vae)
    