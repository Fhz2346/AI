from math import inf
import pickle
from pathlib import Path
import hydra
from omegaconf import DictConfig

from Config import cfgG
from Env.utils import load_env
from Problem.DoorKeyProblem import DoorKeyProblem
from Algorithm.DP import explore_reachable, dynamic_programming


@hydra.main(
    config_path=str(cfgG.wsf / "Config"),
    config_name="known",
    version_base=None,
)
def main(cfg: DictConfig) :
    env_name = f"{cfg.env.size}-{cfg.env.type}"
    env_path = f"./envs/known_envs/doorkey-{env_name}.env"
    env, _ = load_env(env_path)

    problem = DoorKeyProblem()
    problem.get_info(env)

    states, state2env = explore_reachable(env, problem)
    print(f"Reachable states: {len(states)}")
    V = dynamic_programming(problem, states, state2env)
    out_dir = Path("output/DP/Known")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{env_name}.pkl"
    with out_path.open("wb") as f:
        pickle.dump(V, f)
    print(f"Saved V to {out_path}")


if __name__ == "__main__":
    main()
