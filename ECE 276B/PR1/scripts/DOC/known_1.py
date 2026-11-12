import hydra
from omegaconf import DictConfig

from Config import cfgG
from Env.utils import load_env, draw_gif_from_seq, native_actions_to_act
from Problem.DoorKeyProblem import DoorKeyProblem
from Algorithm.LC import LabelCorrecting


@hydra.main(
    config_path=str(cfgG.wsf / "Config"),
    config_name="known",
    version_base=None,
)
def main(cfg: DictConfig):
    env_name = f"{cfg.env.size}-{cfg.env.type}"
    env_path = f"./envs/known_envs/doorkey-{env_name}.env"
    env, _ = load_env(env_path)

    problem = DoorKeyProblem()
    problem.get_info(env)

    lc = LabelCorrecting(problem)
    _states, actions_native = lc.solve(env, cfg.LC)

    if not actions_native:
        raise RuntimeError("No feasible path to the goal was found.")

    seq = native_actions_to_act(env, actions_native)
    print("Optimal action sequence:")
    print([act.name for act in seq])
    draw_gif_from_seq(seq, load_env(env_path)[0], file_name=env_name)


if __name__ == "__main__":
    main()
