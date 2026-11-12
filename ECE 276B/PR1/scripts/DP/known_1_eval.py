import pickle
import hydra
from omegaconf import DictConfig

from Config import cfgG
from Env.utils import load_env, native_actions_to_act, draw_gif_from_seq
from Problem.DoorKeyProblem import DoorKeyProblem
from Algorithm.DP import extract_optimal_actions

@hydra.main(
    config_path=str(cfgG.wsf / "Config"),
    config_name="known",
    version_base=None,
)
def main(cfg: DictConfig):
    env_name = f"{cfg.env.size}-{cfg.env.type}"
    env_path = f"./envs/known_envs/doorkey-{env_name}.env"
    v_path = cfgG.out_dir / "DP" / "Known" / f"{env_name}.pkl"

    env, _ = load_env(env_path)
    with v_path.open("rb") as f:
        V = pickle.load(f)
    print(f"Loaded V from {v_path}")

    problem = DoorKeyProblem()
    problem.get_info(env)

    actions_native, total_cost = extract_optimal_actions(env, problem, V)
    seq = native_actions_to_act(env, actions_native)

    print("Optimal action sequence from saved V:")
    print([act.name for act in seq])
    print(f"Path length: {len(seq)}")
    print(f"Total stage cost: {total_cost}")
    draw_gif_from_seq(seq, load_env(env_path)[0], file_name=f"known/{env_name}")


if __name__ == "__main__":
    main()
