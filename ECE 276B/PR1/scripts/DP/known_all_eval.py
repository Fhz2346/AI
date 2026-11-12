import pickle

from Config import cfgG
from Env.create_env import known_map_configs
from Env.utils import load_env, native_actions_to_act, draw_gif_from_seq
from Problem.DoorKeyProblem import DoorKeyProblem
from Algorithm.DP import extract_optimal_actions


def main():
    known_dir = cfgG.wsf / "envs" / "known_envs"
    v_dir = cfgG.out_dir / "DP" / "Known"

    for map_name in sorted(known_map_configs.keys()):
        env_path = known_dir / f"{map_name}.env"
        env_name = map_name.removeprefix("doorkey-")
        v_path = v_dir / f"{env_name}.pkl"
        print(f"\n--- {map_name} ---")

        if not env_path.is_file():
            print(f"[skip] missing env: {env_path}")
            continue
        if not v_path.is_file():
            print(f"[skip] missing V: {v_path}")
            continue

        env, _ = load_env(str(env_path))
        with v_path.open("rb") as f:
            V = pickle.load(f)
        print(f"Loaded V from {v_path}")
        print("total states: ", len(V))
        problem = DoorKeyProblem()
        problem.get_info(env)

        actions_native, total_cost = extract_optimal_actions(env, problem, V)
        seq = native_actions_to_act(env, actions_native)

        print("Optimal action sequence from saved V:")
        print([act.name for act in seq])
        print(f"Path length: {len(seq)}")
        print(f"Total stage cost: {total_cost}")
        draw_gif_from_seq(seq, load_env(str(env_path))[0], file_name=f"known/{env_name}")


if __name__ == "__main__":
    main()
