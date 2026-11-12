import hydra
from omegaconf import DictConfig

from config import cfgG


@hydra.main(config_path=f"{cfgG.wsf}/Config", config_name="", version_base=None)
def main(cfg: DictConfig):
    pass


if __name__ == "__main__":
    main()
