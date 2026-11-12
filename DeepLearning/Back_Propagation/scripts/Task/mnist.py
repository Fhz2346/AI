import hydra
from omegaconf import DictConfig

from Network.MLP import MLP
from Train.TrainPipeline import TrainPipeline
from Load import load_mnist
from Config import cfgG


@hydra.main(config_path=f"{cfgG.wsf}/Config", config_name="MLP", version_base=None)
def main(cfg: DictConfig):
    X_train, y_train, X_test, y_test = load_mnist()
    model = MLP(cfg.network)
    train_pip = TrainPipeline(model, cfg.train)
    train_pip.train(X_train, y_train, X_test, y_test, cfg.train.num_epochs)

if __name__ == "__main__":
    main()
