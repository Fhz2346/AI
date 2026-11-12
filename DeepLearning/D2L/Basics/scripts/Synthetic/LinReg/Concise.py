import torch
from torch.utils import data
from torch import nn
import numpy as np
import hydra
from omegaconf import DictConfig

from Config import cfgG


@hydra.main(config_path=f"{cfgG.wsf}/Config/Linear", config_name="LinReg")
def main(cfg: DictConfig):
    dataset = np.load(f"{cfgG.out_dir}/synthetic.npz")
    features = dataset["features"]
    labels = dataset["labels"]
    features = torch.tensor(features, dtype=torch.float32)
    labels = torch.tensor(labels, dtype=torch.float32)

    dataset = data.TensorDataset(*(features, labels))
    data_iter = data.DataLoader(dataset, cfg.training.batch_size, shuffle=True)

    net = nn.Sequential(nn.Linear(2, 1))
    net[0].weight.data.normal_(0, 0.01)
    net[0].bias.data.fill_(0)

    loss = nn.MSELoss()
    trainer = torch.optim.SGD(net.parameters(), lr=cfg.training.lr)
    # Training
    for epoch in range(cfg.training.num_epochs):
        for X, y_hat in data_iter:
            l = loss(net(X), y_hat)
            trainer.zero_grad()
            l.backward()
            trainer.step()
        l = loss(net(features), labels)
        print(f"epoch {epoch + 1}, loss {l:f}")
    # Evaluate
    w, b = net[0].weight.data, net[0].bias.data
    w_true = torch.tensor(cfg.data.w)
    print(f"w_true={w_true}")
    print(f"w_estimated={w}")
    print(f"b_true={cfg.data.b}")
    print(f"b_estimated={b}")

if __name__ == "__main__":
    main()