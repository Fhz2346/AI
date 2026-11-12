import hydra
from omegaconf import DictConfig
import torch

from Data.Load import load_data_time_machine
from Model.RNN_dict import RNN_dict
from Utils.Device import try_gpu
from Evaluate import predict_ch8
from Config import cfgG


@hydra.main(config_path=f"{cfgG.wsf}/Config", config_name="RNN")
def main(cfg: DictConfig):
    train_iter, vocab = load_data_time_machine(cfg.train.data)
    cfg.train.data.vocab_size = len(vocab)
    device = try_gpu()
    net = RNN_dict[cfg.network.type](cfg, device)
    net.load_state_dict(torch.load(f"{cfgG.out_dir}/{net.__class__.__name__}_{cfg.evaluate.load_epoch}.pth"))
    print(predict_ch8(cfg.evaluate.prefix, cfg.evaluate.num_preds, net, vocab, device))

if __name__ == "__main__":
    main()
