"""
Load one random DoorKey map from envs/random_envs and run Label Correcting (UCS).
"""
from __future__ import annotations

import hydra
from omegaconf import DictConfig

from Config import cfgG
from Env.utils import load_env, load_random_env, native_actions_to_act, draw_gif_from_seq
from LC.LC import LabelCorrecting
from Problem.DoorKeyProblem import DoorKeyProblem


@hydra.main(
    config_path=str(cfgG.wsf / "Config"),
    config_name="random",
    version_base=None,
)
def main(cfg: DictConfig):
    env_dir = cfgG.wsf / "envs" / "random_envs"
    env, _info, env_path = load_random_env(str(env_dir))

    problem = DoorKeyProblem()
    problem.get_info(env)

    lc = LabelCorrecting(problem)
    states, actions_native = lc.solve(env, cfg)
    if not actions_native:
        raise RuntimeError(f"No feasible path for {env_path}.")

    path_cost = sum(problem.cost_fn(a) for a in actions_native)
    seq = native_actions_to_act(env, actions_native)
    draw_gif_from_seq(seq, load_env(env_path)[0], file_name="random")
    print(f"Loaded: {env_path}")
    print(f"States on path: {len(states)}, actions: {len(actions_native)}")
    print(f"Total cost (UCS / LC): {path_cost:.4g}")
    print(f"Action sequence: {[a.name for a in seq]}")


if __name__ == "__main__":
    main()
