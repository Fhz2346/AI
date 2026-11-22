from pathlib import Path
import json
import torch
from network import Generator, Discriminator
from pathes import CKPT_DIR, HISTORY_PATH

def ckpt_path(epoch):
    return CKPT_DIR / f"epoch_{epoch:03d}.pt"


def list_checkpoints():
    return sorted(CKPT_DIR.glob("epoch_*.pt"))


def save_checkpoint(epoch, net_G, net_D, cfg, loss_D, loss_G):
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    path = ckpt_path(epoch)
    torch.save(
        {
            "epoch": epoch,
            "net_G": net_G.state_dict(),
            "net_D": net_D.state_dict(),
            "loss_D": float(loss_D),
            "loss_G": float(loss_G),
            "n_G": int(cfg.model.n_G),
            "n_D": int(cfg.model.n_D),
            "latent_dim": int(cfg.model.latent_dim),
        },
        path,
    )
    print(f"saved {path}")
    return path


def load_checkpoint(path, device):
    ckpt = torch.load(path, map_location=device)
    net_G = Generator(n_G=ckpt["n_G"], latent_dim=ckpt["latent_dim"])
    net_D = Discriminator(n_D=ckpt["n_D"])
    net_G.load_state_dict(ckpt["net_G"])
    net_D.load_state_dict(ckpt["net_D"])
    return ckpt, net_G.to(device), net_D.to(device)


def save_history(history):
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    HISTORY_PATH.write_text(json.dumps(history, indent=2), encoding="utf-8")


def load_history():
    if not HISTORY_PATH.exists():
        return {"epoch": [], "loss_D": [], "loss_G": []}
    return json.loads(HISTORY_PATH.read_text(encoding="utf-8"))
