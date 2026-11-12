from math import inf
from Problem.Problem import Problem
from .OpenList import OpenList


class LabelCorrecting:

    def __init__(self, problem: Problem):
        self.problem = problem

    def init_open(self, cfg):
        policy2open = {
            "DFS": OpenList("lifo"),
            "BFS": OpenList("fifo"),
            "UCS": OpenList("min", lambda s: self.g[s]),
            "A*" : OpenList("min", lambda s: self.g[s] + self.problem.heuristic(s)),
        }
        self.OPEN = policy2open[cfg.policy]

    def init(self, env):
        s = self.problem.get_state(env)
        self.g = {s: 0}  # lowest cost discovered so far from s
        self.parent = {s: None}
        self.parent_action = {}
        self.state2env = {s: self.problem.clone_env(env)}
        self.OPEN.push(s)

    def solve(self, env, cfg):
        self.init_open(cfg)
        self.init(env)
        tau = None  # there might be multiple goal state
        while self.OPEN:
            cur_state = self.OPEN.pop()
            cur_env = self.state2env[cur_state]
            cur_cost = self.g[cur_state]
            for action in self.problem.actions:
                try:
                    nxt_env, nxt_state, step_cost, done = self.problem.step(cur_env, action)
                except:
                    continue
                if nxt_state is None:  # action is invalid
                    continue
                new_cost = cur_cost + step_cost
                if new_cost < self.g.get(nxt_state, inf) and \
                   new_cost < self.g.get(tau, inf):
                    self.g[nxt_state] = new_cost
                    self.parent[nxt_state] = cur_state
                    self.parent_action[nxt_state] = action
                    self.state2env[nxt_state] = nxt_env
                    if self.problem.is_goal(nxt_state):
                        tau = nxt_state  # among all states that satisfy goal, tau is the one with lowest cost
                    else:
                        self.OPEN.push(nxt_state)
        return self.reconstruct_path(tau)

    def reconstruct_path(self, tau):
        states = []
        actions = []
        cur = tau
        while cur is not None:
            states.append(cur)
            if self.parent[cur] is not None:
                actions.append(self.parent_action[cur])
            cur = self.parent[cur]
        states.reverse()
        actions.reverse()
        return states, actions
