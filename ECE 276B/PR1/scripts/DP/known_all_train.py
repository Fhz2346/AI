import pickle
from pathlib import Path

from Config import cfgG
from Env.create_env import known_map_configs
from Env.utils import load_env
from Problem.DoorKeyProblem import DoorKeyProblem
from Algorithm.DP import explore_reachable, dynamic_programming


def main() -> None:
    known_dir = cfgG.wsf / "envs" / "known_envs"
    out_dir = cfgG.out_dir / "DP" / "Known"
    out_dir.mkdir(parents=True, exist_ok=True)

    for map_name in sorted(known_map_configs.keys()):
        env_path = known_dir / f"{map_name}.env"
        env_name = map_name.removeprefix("doorkey-")
        print(f"\n--- {map_name} ---")
        if not env_path.is_file():
            print(f"[skip] missing: {env_path}")
            continue

        env, _ = load_env(str(env_path))
        problem = DoorKeyProblem()
        problem.get_info(env)

        states, state2env = explore_reachable(env, problem)
        print(f"Reachable states: {len(states)}")
        V = dynamic_programming(problem, states, state2env)

        out_path = out_dir / f"{env_name}.pkl"
        with out_path.open("wb") as f:
            pickle.dump(V, f)
        print(f"Saved V to {out_path}")


if __name__ == "__main__":
    main()
