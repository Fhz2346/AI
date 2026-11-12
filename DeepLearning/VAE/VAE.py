import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

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


class Sampling(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, z_mean, z_log_var):
        eps = torch.randn_like(z_mean)
        return z_mean + torch.exp(0.5 * z_log_var) * eps


class Encoder(nn.Module):
    def __init__(self, latent_dim=2):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)

        self.flatten = nn.Flatten()
        self.fc = nn.Linear(7 * 7 * 64, 16)

        self.z_mean = nn.Linear(16, latent_dim)
        self.z_log_var = nn.Linear(16, latent_dim)

        self.sampling = Sampling()

    def forward(self, x):
        x = F.relu(self.conv1(x))      # (batch, 32, 14, 14)
        x = F.relu(self.conv2(x))      # (batch, 64, 7, 7)

        x = self.flatten(x)
        x = F.relu(self.fc(x))

        z_mean = self.z_mean(x)
        z_log_var = self.z_log_var(x)
        z = self.sampling(z_mean, z_log_var)
        return z_mean, z_log_var, z


class Decoder(nn.Module):
    def __init__(self, latent_dim=2):
        super().__init__()
        self.fc = nn.Linear(latent_dim, 7 * 7 * 64)
        self.deconv1 = nn.ConvTranspose2d(64, 64, 3, stride=2, padding=1, output_padding=1)
        self.deconv2 = nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1)
        self.deconv3 = nn.ConvTranspose2d(32, 1, 3, padding=1)

    def forward(self, z):
        x = F.relu(self.fc(z))
        x = x.view(-1, 64, 7, 7)
        x = F.relu(self.deconv1(x))
        x = F.relu(self.deconv2(x))
        x = torch.sigmoid(self.deconv3(x))
        return x


class VAE(nn.Module):
    def __init__(self, latent_dim=2):
        super().__init__()
        self.encoder = Encoder(latent_dim)
        self.decoder = Decoder(latent_dim)

    def forward(self, x):
        z_mean, z_log_var, z = self.encoder(x)
        reconstruction = self.decoder(z)
        return reconstruction, z_mean, z_log_var


def vae_loss(x, reconstruction, z_mean, z_log_var):
    # Reconstruction loss = BCE
    recon_loss = F.binary_cross_entropy(reconstruction, x, reduction="sum")

    # KL divergence
    kl_loss = -0.5 * torch.sum(1 + z_log_var - z_mean.pow(2) - z_log_var.exp())

    return recon_loss + kl_loss, recon_loss, kl_loss


# =========================================================
# Training Script
# =========================================================
def train(latent_dim=2, epochs=30, batch_size=128, lr=1e-3):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    dataset = datasets.MNIST(root="E:\Dataset", train=True, download=False, transform=transforms.ToTensor())
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Model
    vae = VAE(latent_dim).to(device)
    optimizer = optim.Adam(vae.parameters(), lr=lr)

    # Training
    for epoch in range(epochs):
        total_loss = 0
        total_recon = 0
        total_kl = 0

        for batch, (x, _) in enumerate(loader):
            x = x.to(device)

            reconstruction, z_mean, z_log_var = vae(x)

            loss, recon_loss, kl_loss = vae_loss(x, reconstruction, z_mean, z_log_var)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_recon += recon_loss.item()
            total_kl += kl_loss.item()

        print(
            f"Epoch {epoch+1}/{epochs} | "
            f"Loss: {total_loss/len(loader):.2f} | "
            f"Recon: {total_recon/len(loader):.2f} | "
            f"KL: {total_kl/len(loader):.2f}"
        )

    # Save model
    torch.save(vae.state_dict(), "vae_pytorch.pth")
    plot_latent_space(vae)
    print("Model saved → vae_pytorch.pth")


if __name__ == "__main__":
    train()
