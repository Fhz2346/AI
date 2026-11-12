import copy


class Problem:

    def __init__(self):
        self.actions = None
        pass

    def cost_fn(self, action):
        raise NotImplementedError

    def heuristic(self, state):
        raise NotImplementedError

    def is_goal(self, state):
        raise NotImplementedError

    # each distinct state is a node in graph
    def get_state(self, env, obs=None):
        raise NotImplementedError

    def step(self, env, action):
        nxt_env = self.clone_env(env)
        obs, reward, terminated, truncated, _ = nxt_env.step(action)
        nxt_state = self.get_state(nxt_env, obs)
        done = terminated or truncated
        return nxt_env, nxt_state, self.cost_fn(action), done

    def get_info(self, env):
        pass  # No static information in env

    def clone_env(self, env):
        return copy.deepcopy(env)
