from math import inf
from Problem.Problem import Problem
from .utils import build_cost_matrix


def dynamic_programming(problem: Problem, states, state2env):
    state_list = list(states)
    n = len(state_list)
    C = build_cost_matrix(problem, state_list, state2env)
    T = n - 1
    V = [inf] * n
    for i in range(n):
        if problem.is_goal(state_list[i]):
            V[i] = 0.0
    for _ in range(T - 1, -1, -1):
        V_new = [0.0] * n
        for i in range(n):
            if problem.is_goal(state_list[i]):
                V_new[i] = 0.0
            else:
                V_new[i] = min(C[i][j] + V[j] for j in range(n))
        V = V_new
    return {state_list[i]: V[i] for i in range(n)}
