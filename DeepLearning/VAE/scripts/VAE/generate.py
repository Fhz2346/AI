import torch
import numpy as np
import matplotlib.pyplot as plt
from omegaconf import OmegaConf
from Network.VAE import VAE
from Pathes import P

if __name__ == "__main__":
    cfg = OmegaConf.load("config/MNIST.yaml")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    vae = VAE(cfg.network).to(device)
    vae.load_state_dict(torch.load(P.ckpt_dir / "vae_pytorch.pth"))
    z = torch.randn(1, cfg.network.latent_dim).to(device)
    reconstruction = vae.decoder(z)
    plt.imshow(reconstruction.cpu().detach().numpy()[0, 0], cmap="Greys_r")
    plt.show()
