import torch
import numpy as np
import hydra
from omegaconf import DictConfig

from Train.Utils.Loss import squared_loss
from Train.Utils.Optimizer import SGD
from Data.Dataset.Synthetic import data_iter_self
from Config import cfgG


@hydra.main(config_path=f"{cfgG.wsf}/Config/Linear", config_name="LinReg")
def main(cfg: DictConfig):
    data = np.load(f"{cfgG.out_dir}/{cfg.data.save_name}.npz")
    features = data["features"]
    labels = data["labels"]
    features = torch.tensor(features, dtype=torch.float32)
    labels = torch.tensor(labels, dtype=torch.float32)

    w = torch.normal(0, 0.01, size=(2, 1), requires_grad=True)
    b = torch.zeros(1, requires_grad=True)
    sgd = SGD([w, b], lr=cfg.training.lr)

    def net(X):
        return X @ w + b

    # Training
    for epoch in range(cfg.training.num_epochs):
        for X, y_hat in data_iter_self(cfg.training.batch_size, features, labels):
            l = squared_loss(net(X), y_hat)
            l.mean().backward()
            sgd.step()
        with torch.no_grad():
            train_l = squared_loss(net(features), labels)
            print(f"epoch {epoch + 1}, loss {float(train_l.mean()):f}")

    # Evaluate
    w_true = torch.tensor(cfg.data.w)
    print(f"w_true={w_true}")
    print(f"w_estimated={w.reshape(w_true.shape)}")
    print(f"b_true={cfg.data.b}")
    print(f"b_estimated={b}")

if __name__ == "__main__":
    main()