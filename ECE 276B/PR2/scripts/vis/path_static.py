import hydra
from omegaconf import DictConfig
import numpy as np
import matplotlib.pyplot as plt

from config import cfgG
from utils import load_map, draw_map



# E1_flappy_bird, E2_maze, E3_monza, E4_single_cube, E5_tower
@hydra.main(
    config_path=f"{cfgG.wsf}/config/maps/static",
    config_name="E1_flappy_bird",
    version_base=None,
)
def main(cfg: DictConfig):
    boundary, blocks = load_map(cfgG.maps_dir / f"{cfg.mapfile}.txt")
    path = np.load(cfgG.out_dir / "static" / f"{cfg.mapfile}.npy")
    fig, ax, hb, hs, hg = draw_map(boundary, blocks, cfg.start, cfg.goal)
    ax.plot(path[:, 0], path[:, 1], path[:, 2], "r-")
    plt.show()

if __name__ == "__main__":
    main()
