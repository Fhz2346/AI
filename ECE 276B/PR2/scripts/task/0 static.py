import hydra
from omegaconf import DictConfig

from utils import load_map, tic, toc
from planner import StaticPlanner
from planner.utils import path_has_collision
from planner.GridMap import GridMap
import numpy as np
from config import cfgG

@hydra.main(
    config_path=f"{cfgG.wsf}/config/maps/static",
    config_name="E5_tower",  # E1_flappy_bird, E2_maze, E3_monza, E4_single_cube, E5_tower
    version_base=None,
)
def main(cfg: DictConfig):
    boundary, blocks = load_map(cfgG.maps_dir / f"{cfg.mapfile}.txt")
    map = GridMap(boundary, blocks, resolution=0.25)
    MP = StaticPlanner(map)
    t0 = tic()
    path = MP.plan(cfg.start, cfg.goal)
    np.save(cfgG.out_dir / "static" / f"{cfg.mapfile}.npy", path)
    toc(t0, "Planning")
    collision = path_has_collision(path, boundary, blocks)
    goal_reached = np.sum((path[-1] - cfg.goal) ** 2) <= 0.1
    pathlength = np.sum(np.sqrt(np.sum(np.diff(path, axis=0) ** 2, axis=1)))
    success = (not collision) and goal_reached
    print("Success: %r" % success)
    print("Path length: %s" % pathlength)
    print("\n")


if __name__ == "__main__":
    main()
