from collections import deque


class OpenList:

    def __init__(self, policy, get_cost=None):
        self._open = deque()
        self._in_open = set()

        self.policy = policy.lower()
        self.cost_fn = get_cost

    def push(self, state):
        if state in self._in_open:
            return
        self._open.append(state)
        self._in_open.add(state)

    def pop(self):
        if self.policy == "fifo":
            state = self._open.popleft()
        elif self.policy == "lifo":
            state = self._open.pop()
        else:
            state = min(self._open, key=self.cost_fn)
            self._open.remove(state)
        self._in_open.remove(state)
        return state

    def __contains__(self, state):
        return state in self._in_open

    def __bool__(self):
        return bool(self._open)


