import sys
from pathlib import Path
from omegaconf import OmegaConf

from Pathes import P
from utils import build_models, get_loader, train

if __name__ == "__main__":
    cfg = OmegaConf.load(P.cfg_dir / "DCGAN.yaml")
    data_iter = get_loader(cfg)
    net_G, net_D = build_models(cfg)
    train(net_D, net_G, data_iter, cfg)
