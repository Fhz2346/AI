from collections import deque
from math import inf
import numpy as np
from Problem.Problem import Problem


def explore_reachable(env, problem: Problem):
    s0 = problem.get_state(env)
    states = {s0}
    state2env = {s0: problem.clone_env(env)}
    q = deque([s0])
    while q:
        cur_state = q.popleft()
        cur_env = state2env[cur_state]
        for action in problem.actions:
            try:
                nxt_env, nxt_state, _cost, _done = problem.step(cur_env, action)
            except Exception:
                continue
            if nxt_state is None:
                continue
            if nxt_state not in states:
                states.add(nxt_state)
                state2env[nxt_state] = nxt_env
                q.append(nxt_state)
    return states, state2env


def build_cost_matrix(problem: Problem, state_list, state2env):
    n = len(state_list)
    C = [[inf] * n for _ in range(n)]
    state2idx = {s: i for i, s in enumerate(state_list)}
    for i, s in enumerate(state_list):
        env = state2env[s]
        for action in problem.actions:
            try:
                _nxt_env, nxt_state, cost, _done = problem.step(env, action)
            except Exception:
                continue
            if nxt_state is None or nxt_state not in state2idx:
                continue
            j = state2idx[nxt_state]
            if cost < C[i][j]:
                C[i][j] = cost
    C = np.array(C)
    return C


def extract_optimal_actions(env, problem: Problem, V:dict):
    actions_native = []
    total_cost = 0.0
    cur_env = problem.clone_env(env)
    cur_state = problem.get_state(cur_env)
    for _ in range(len(V) + 5):  # max_steps = len(V) + 5
        if problem.is_goal(cur_state):
            return actions_native, total_cost
        best_action = None
        best_env = None
        best_state = None
        best_q = inf
        best_step_cost = None

        for action in problem.actions:
            nxt_env, nxt_state, step_cost, _done = problem.step(cur_env, action)
            v_nxt = V.get(nxt_state, inf)
            q_val = step_cost + v_nxt
            if q_val < best_q:
                best_q = q_val
                best_action = action
                best_env = nxt_env
                best_state = nxt_state
                best_step_cost = step_cost

        if best_action is None or best_q >= inf:
            raise RuntimeError("No feasible action found from current state using loaded V.")

        actions_native.append(best_action)
        total_cost += best_step_cost
        cur_env = best_env
        cur_state = best_state
    raise RuntimeError("Exceeded step limit while extracting policy. Check V consistency.")
