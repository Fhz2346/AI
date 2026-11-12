import torch
from omegaconf import OmegaConf
from d2l import torch as d2l
from Pathes import P


def gen_data(cfg):
    X = torch.normal(0.0, 1, (cfg.num_samples, cfg.dim_feature))
    A = torch.tensor(cfg.A)
    b = torch.tensor(cfg.b)
    data = X @ A + b
    return data


if __name__ == "__main__":
    cfg = OmegaConf.load('config/GAN.yaml')
    data = gen_data(cfg.data)
    torch.save(data, P.data_dir / 'data_gen.pkl')

    d2l.set_figsize()
    d2l.plt.scatter(data[:100, (0)].detach().numpy(), data[:100, (1)].detach().numpy())
    d2l.plt.show()
