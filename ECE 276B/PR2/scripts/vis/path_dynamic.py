import hydra
from omegaconf import DictConfig
import numpy as np
import matplotlib.pyplot as plt

from config import cfgG
from utils import load_map, draw_map, animate_dynamic_target, is_time_met


@hydra.main(
    config_path=f"{cfgG.wsf}/config/maps/dynamic",
    config_name="E7_room", # E6_window, E7_room
    version_base=None,
)
def main(cfg: DictConfig):
    boundary, blocks = load_map(cfgG.maps_dir / f"{cfg.mapfile}.txt")
    data = np.load(cfgG.out_dir / "dynamic" / f"{cfg.mapfile}.npy", allow_pickle=True).item()
    paths = data["paths"]
    times = data["times"]
    goals = np.asarray(cfg.goal, dtype=np.float64)
    fig, ax, hb, hs, hg = draw_map(boundary, blocks, cfg.start, cfg.goal)
    path_success = is_time_met(times, cfg.stay_time)
    # animate_dynamic_target(
    #     ax, goals, paths, times, cfg.stay_time.goal_1, cfg.stay_time.goal_other
    # )
    for i, path in enumerate(paths):
        if path_success[i]:
            ax.plot(path[:, 0], path[:, 1], path[:, 2], "r-")
        else:
            ax.plot(
                [goals[i, 0]],
                [goals[i, 1]],
                [goals[i, 2]],
                "mo",
                markersize=9,
                markeredgecolor="k",
            )
    plt.show()

if __name__ == "__main__":
    main()
