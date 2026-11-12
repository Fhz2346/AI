import hydra
from omegaconf import DictConfig
import numpy as np
import time

from config import cfgG
from utils import load_map, is_time_met
from planner import DynamicPlanner
from planner.GridMap import GridMap
from planner.utils import path_has_collision


@hydra.main(
    config_path=f"{cfgG.wsf}/config/maps/dynamic",
    config_name="E7_room",  # E6_window, E7_room
    version_base=None,
)
def main(cfg: DictConfig): 
    boundary, blocks = load_map(cfgG.maps_dir / f"{cfg.mapfile}.txt")
    goals = np.asarray(cfg.goal, dtype=np.float64)
    map = GridMap(boundary, blocks)
    MP = DynamicPlanner(map, epsilon=2.5)
    paths = []
    times = []
    for i in range(goals.shape[0]):
        t0 = time.perf_counter()
        path = MP.plan(cfg.start, goals[i])
        times.append(time.perf_counter() - t0)
        paths.append(path)
    data = {
        "paths": paths,
        "times": times,
    }
    np.save(cfgG.out_dir / "dynamic" / f"{cfg.mapfile}.npy", data)
    path_success = is_time_met(times, cfg.stay_time)
    collision = False
    goal_reached_list = []
    pathlength = []
    for i, path in enumerate(paths):
        collision = collision or path_has_collision(path, boundary, blocks)
        reached = np.sum((path[-1] - goals[i]) ** 2) <= 0.1
        goal_reached_list.append(reached)
        pathlength.append(np.sum(np.sqrt(np.sum(np.diff(path, axis=0) ** 2, axis=1))))

    goal_reached = np.all(goal_reached_list)
    success = (not collision) and goal_reached and np.all(path_success)
    print(f"Success: {success}")

if __name__ == "__main__":
    main()
