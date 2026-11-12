from omegaconf import OmegaConf
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch import optim
from Network.VAE import VAE
from loss import vae_loss
from Pathes import P


def train(vae: VAE, data_loader, optimizer, cfg):
    pbar = tqdm(range(cfg.epochs), desc="Training")
    for epoch in pbar:
        total_loss = 0
        total_recon = 0
        total_kl = 0
        for batch, (x, _) in enumerate(data_loader):
            x = x.to(device)
            reconstruction, z_mean, z_log_var = vae(x)
            loss, recon_loss, kl_loss = vae_loss(x, reconstruction, z_mean, z_log_var)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            total_recon += recon_loss.item()
            total_kl += kl_loss.item()
        pbar.set_postfix(
            Loss=f"{total_loss / len(data_loader):.2f}",
            Recon=f"{total_recon / len(data_loader):.2f}",
            KL=f"{total_kl / len(data_loader):.2f}",
        )
    torch.save(vae.state_dict(), P.ckpt_dir / "vae1.pth")


if __name__ == "__main__":
    cfg = OmegaConf.load("config/MNIST.yaml")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dataset = datasets.MNIST(root=cfg.dataset.root, train=True, download=False, transform=transforms.ToTensor())
    data_loader = DataLoader(dataset, batch_size=cfg.dataloader.batch_size, shuffle=True)
    vae = VAE(cfg.network).to(device)
    optimizer = optim.Adam(vae.parameters(), lr=cfg.optimizer.lr)
    train(vae, data_loader, optimizer, cfg.train)
