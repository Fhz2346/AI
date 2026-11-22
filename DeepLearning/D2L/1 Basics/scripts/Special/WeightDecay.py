import torch
from tqdm import tqdm
import hydra
from omegaconf import DictConfig

from Train.Utils.Loss import squared_loss
from Train.Utils.Optimizer import SGD
from Data.Dataset.Synthetic import synthetic_data
from Data.Utils import load_array
from Evaluate import evaluate_loss
from Config import cfgG


def train(cfg: DictConfig, train_iter, test_iter, lambd):
    w = torch.normal(0, 1, size=(cfg.data.num_inputs, 1), requires_grad=True)
    b = torch.zeros(1, requires_grad=True)
    sgd = SGD([w, b], lr=cfg.training.lr)

    def net(X):
        return X @ w + b

    pbar = tqdm(range(cfg.training.num_epochs))
    pbar.set_description(f"lambda={lambd}")
    for epoch in pbar:
        for X, y in train_iter:
            l = squared_loss(net(X), y) + lambd * (w**2).sum() / 2
            l.sum().backward()
            sgd.step()
        train_loss = evaluate_loss(net, train_iter, squared_loss)
        test_loss = evaluate_loss(net, test_iter, squared_loss)
        pbar.set_postfix(
            {
                "train_loss": f"{train_loss:.6f}",
                "test_loss": f"{test_loss:.6f}",
            }
        )


@hydra.main(config_path=f"{cfgG.wsf}/Config/Special", config_name="WeightDecay")
def main(cfg: DictConfig):
    true_w = torch.ones((cfg.data.num_inputs, 1)) * cfg.data.w_scale
    true_b = cfg.data.b
    train_iter = load_array(
        synthetic_data(true_w, true_b, cfg.data.n_train), cfg.training.batch_size
    )
    test_iter = load_array(
        synthetic_data(true_w, true_b, cfg.data.n_test),
        cfg.training.batch_size,
        is_train=False,
    )
    # Compare training with and without weight decay
    train(cfg, train_iter, test_iter, lambd=0)
    train(cfg, train_iter, test_iter, lambd=3)


if __name__ == "__main__":
    main()
