import pickle
from pathlib import Path

import hydra
from omegaconf import DictConfig

from Config import cfgG
from Env.utils import load_env, native_actions_to_act
from Problem.DoorKeyProblemRandom import DoorKeyProblemRandom
from Algorithm.LC import LabelCorrecting


POLICY_FILENAME = "random_policy.pkl"

def _train_policy_table(
    env_dir: Path,
    cfg: DictConfig,
) -> dict[object, object]:
    """Fill π(s) → minigrid action from optimal UCS trajectories."""
    table: dict[object, object] = {}
    problem = DoorKeyProblemRandom()
    lc = LabelCorrecting(problem)

    for idx in range(1, 37):
        env_path = env_dir / f"DoorKey-10x10-{idx}.env"
        env, _ = load_env(str(env_path))
        problem.get_info(env)

        states, actions_native = lc.solve(env, cfg)
        if not actions_native:
            raise RuntimeError(f"No feasible path for {env_path}.")

        for i, action in enumerate(actions_native):
            s = states[i]
            if s in table and table[s] != action:
                raise RuntimeError(
                    "Inconsistent expert action for the same MDP state across envs. "
                    f"state={s}, existing={table[s]}, new={action}, env={env_path.name}"
                )
            table[s] = action

    return table


def _rollout_policy(
    env,
    env_path: Path,
    problem: DoorKeyProblemRandom,
    table: dict[object, object],
) -> tuple[float, int, list]:
    """Execute the trained policy from the env's initial state."""
    cur = env
    total_cost = 0.0
    steps = 0
    actions_native: list = []
    while not problem.is_goal(problem.get_state(cur)):
        s = problem.get_state(cur)
        if s not in table:
            raise RuntimeError(
                f"Policy has no entry for state {s} on {env_path.name}. "
                "Training trajectory may be incomplete."
            )
        action = table[s]
        actions_native.append(action)
        cur, _s2, step_cost, _done = problem.step(cur, action)
        total_cost += step_cost
        steps += 1
        if steps > 10**6:
            raise RuntimeError(f"Rollout exceeded step limit on {env_path.name}.")
    return total_cost, steps, actions_native


@hydra.main(
    config_path=str(cfgG.wsf / "Config"),
    config_name="random",
    version_base=None,
)
def main(cfg: DictConfig):
    env_dir = cfgG.wsf / "envs" / "random_envs"

    print("Part B: training one policy from 36 environments (expert = UCS/LC)...")
    policy_table = _train_policy_table(env_dir, cfg)
    print(f"Policy table size: {len(policy_table)} (state,action) pairs.")

    out_path = cfgG.out_dir / POLICY_FILENAME
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "wb") as f:
        pickle.dump(policy_table, f)
    print(f"Saved trained policy to {out_path}")

    print("Evaluating the single policy (rollout, no planner)...")
    problem = DoorKeyProblemRandom()
    total_rollout_cost = 0.0
    for idx in range(36):
        env_path = env_dir / f"DoorKey-10x10-{idx+1}.env"
        env, _ = load_env(str(env_path))
        problem.get_info(env)

        path_cost, n_steps, acts = _rollout_policy(env, env_path, problem, policy_table)
        total_rollout_cost += path_cost

        env_ref, _ = load_env(str(env_path))
        seq = native_actions_to_act(env_ref, acts)
        print(
            f"{env_path.name}: rollout_cost={path_cost:.4g}, steps={n_steps}, "
            f"actions={[a.name for a in seq]}"
        )

    print(f"Sum of rollout costs (trained policy): {total_rollout_cost:.4g}")


if __name__ == "__main__":
    main()
