from omegaconf import OmegaConf
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch import optim
from torch.nn import functional as F
from Network.CVAE import CVAE, loss_fn
from Pathes import P


def train(cvae: CVAE, data_loader, optimizer, cfg, num_classes):
    pbar = tqdm(range(cfg.epochs), desc="Training CVAE")
    for epoch in pbar:
        total_loss = 0
        total_recon = 0
        total_kl = 0
        for x, y in data_loader:
            x = x.to(device)
            y = F.one_hot(y, num_classes=num_classes).float().to(device)
            reconstruction, z_mean, z_log_var, yh = cvae(x, y)
            loss, recon_loss, kl_loss = loss_fn(x, reconstruction, z_mean, z_log_var, yh)
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
    P.ckpt_dir.mkdir(parents=True, exist_ok=True)
    torch.save(cvae.state_dict(), P.ckpt_dir / "cvae.pth")


if __name__ == "__main__":
    cfg = OmegaConf.load("config/CVAE.yaml")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dataset = datasets.MNIST(
        root=cfg.dataset.root, train=True, download=False, transform=transforms.ToTensor()
    )
    data_loader = DataLoader(dataset, batch_size=cfg.dataloader.batch_size, shuffle=True)
    cvae = CVAE(cfg.network).to(device)
    optimizer = optim.RMSprop(cvae.parameters(), lr=cfg.optimizer.lr)
    train(cvae, data_loader, optimizer, cfg.train, cfg.network.num_classes)
