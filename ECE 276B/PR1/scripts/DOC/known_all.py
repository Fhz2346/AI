from omegaconf import OmegaConf

from Config import cfgG
from Env.create_env import known_map_configs
from Env.utils import draw_gif_from_seq, load_env, native_actions_to_act
from Algorithm.LC import LabelCorrecting
from Problem.DoorKeyProblem import DoorKeyProblem


def main() -> None:
    cfg = OmegaConf.load(cfgG.wsf / "Config" / "known.yaml")
    known_dir = cfgG.wsf / "envs" / "known_envs"

    for map_name in sorted(known_map_configs.keys()):
        env_path = known_dir / f"{map_name}.env"
        print(f"\n--- {map_name} ---")
        if not env_path.is_file():
            print(f"[skip] missing: {env_path}")
            continue

        env, _ = load_env(str(env_path))
        problem = DoorKeyProblem()
        problem.get_info(env)

        lc = LabelCorrecting(problem)
        _states, actions_native = lc.solve(env, cfg)
        if not actions_native:
            raise RuntimeError(f"No feasible path for {map_name}.")

        seq = native_actions_to_act(env, actions_native)
        print("Optimal action sequence:")
        print([act.name for act in seq])
        draw_gif_from_seq(
            seq, load_env(str(env_path))[0], file_name=f"known/{map_name[8:]}"
        )


if __name__ == "__main__":
    main()
