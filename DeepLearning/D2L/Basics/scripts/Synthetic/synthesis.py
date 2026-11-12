import torch
import numpy as np
import hydra
from omegaconf import DictConfig

from Data.Dataset.Synthetic import synthetic_data, visualize_data
from Config import cfgG

@hydra.main(config_path=f"{cfgG.wsf}/Config/Linear", config_name="WeightDecay")
def main(cfg: DictConfig):
    cfgD = cfg.data
    w_true = torch.tensor(cfgD.w)
    features, labels = synthetic_data(w_true, cfgD.b, cfgD.num_examples)
    np.savez(f"{cfgG.out_dir}/{cfgD.save_name}.npz", features=features, labels=labels)
    visualize_data(features, labels)

if __name__ == "__main__":
    main()
    