"""
Task helper: load trained random policy and generate rollout GIFs
for all 36 random 10x10 maps.
"""
from __future__ import annotations

import pickle
from pathlib import Path

from Config import cfgG
from Env.utils import draw_gif_from_seq, load_env, native_actions_to_act
from Problem.DoorKeyProblemRandom import DoorKeyProblemRandom


POLICY_FILENAME = "random_policy.pkl"


def _rollout_policy(env, env_path: Path, problem: DoorKeyProblemRandom, table: dict):
    """Execute policy until goal and return native action list."""
    cur = env
    actions_native = []
    steps = 0
    while not problem.is_goal(problem.get_state(cur)):
        s = problem.get_state(cur)
        if s not in table:
            raise RuntimeError(
                f"Policy has no entry for state {s} on {env_path.name}. "
                "Training trajectory may be incomplete."
            )
        action = table[s]
        actions_native.append(action)
        cur, _s2, _step_cost, _done = problem.step(cur, action)
        steps += 1
        if steps > 10**6:
            raise RuntimeError(f"Rollout exceeded step limit on {env_path.name}.")
    return actions_native


def main() -> None:
    policy_path = cfgG.out_dir / POLICY_FILENAME
    if not policy_path.is_file():
        raise FileNotFoundError(
            f"Cannot find policy file: {policy_path}. Please run scripts/Task/random.py first."
        )

    with open(policy_path, "rb") as f:
        policy_table = pickle.load(f)
    print(f"Loaded policy from {policy_path}, table size={len(policy_table)}")

    env_dir = cfgG.wsf / "envs" / "random_envs"
    (cfgG.out_dir / "gif" / "policy").mkdir(parents=True, exist_ok=True)

    problem = DoorKeyProblemRandom()
    for idx in range(1, 37):
        env_path = env_dir / f"DoorKey-10x10-{idx}.env"
        print(f"\n--- {env_path.name} ---")
        if not env_path.is_file():
            print(f"[skip] missing: {env_path}")
            continue

        env, _ = load_env(str(env_path))
        problem.get_info(env)
        actions_native = _rollout_policy(env, env_path, problem, policy_table)

        env_ref, _ = load_env(str(env_path))
        seq = native_actions_to_act(env_ref, actions_native)
        print("Policy action sequence:")
        print([act.name for act in seq])
        draw_gif_from_seq(
            seq,
            env_ref,
            file_name=f"policy/10x10-policy-{idx}",
        )


if __name__ == "__main__":
    main()
