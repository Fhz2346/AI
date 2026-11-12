from omegaconf import OmegaConf

from Config import cfgG
from Env.utils import draw_gif_from_seq, load_env, native_actions_to_act
from Algorithm.LC import LabelCorrecting
from Problem.DoorKeyProblem import DoorKeyProblem


def main() -> None:
    cfg = OmegaConf.load(cfgG.wsf / "Config" / "random.yaml")
    random_dir = cfgG.wsf / "envs" / "random_envs"
    n_maps = 36

    for idx in range(1, n_maps + 1):
        env_name = f"DoorKey-10x10-{idx}"
        env_path = random_dir / f"{env_name}.env"
        print(f"\n--- {env_name} ---")
        if not env_path.is_file():
            print(f"[skip] missing: {env_path}")
            continue

        env, _ = load_env(str(env_path))
        problem = DoorKeyProblem()
        problem.get_info(env)

        lc = LabelCorrecting(problem)
        _states, actions_native = lc.solve(env, cfg)
        if not actions_native:
            raise RuntimeError(f"No feasible path for {env_name}.")

        seq = native_actions_to_act(env, actions_native)
        print("Optimal action sequence:")
        print([act.name for act in seq])
        draw_gif_from_seq(
            seq, load_env(str(env_path))[0], file_name=f"random/10x10-random-{idx}"
        )


if __name__ == "__main__":
    main()
