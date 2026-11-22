import hydra
from omegaconf import DictConfig

from Data.Load import load_data_time_machine
from Model.RNN_dict import RNN_dict
from Utils.Device import try_gpu
from Train.TrainPipeline import TrainPipeline
from Config import cfgG


@hydra.main(config_path=f"{cfgG.wsf}/Config", config_name="RNN")
def main(cfg: DictConfig):
    train_iter, vocab = load_data_time_machine(cfg.train.data)
    train_pip = TrainPipeline(cfg, vocab, try_gpu())
    train_pip.train(train_iter, cfg.train.num_epochs, cfg.train.save_freq)


if __name__ == "__main__":
    main()
