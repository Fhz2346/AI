from dataclasses import dataclass
from typing import List, Tuple

from minigrid.core.actions import Actions
import math

from Env.utils import step_cost
from Pack import Act

from .Problem import Problem

_ACTION_TO_ACT = {
    Actions.forward: Act.MF,
    Actions.left: Act.TL,
    Actions.right: Act.TR,
    Actions.pickup: Act.PK,
    Actions.toggle: Act.UD,
}


@dataclass(frozen=True)
class DoorKeyState:
    agent_pos: tuple
    agent_dir: int
    is_carrying_key: bool
    # One flag per door, in row-major order of door cells (same as door_positions).
    door_open: Tuple[bool, ...]


class DoorKeyProblem(Problem):

    def __init__(self):
        self.actions = [
            Actions.forward,  # Move Forward
            Actions.left,  # Turn Left
            Actions.right,  # Turn Right
            Actions.pickup,  # Pick Key
            Actions.toggle,  # Unlock Door
        ]
        self.goal_pos = None
        self.door_positions = ()

    def cost_fn(self, action):
        return step_cost(_ACTION_TO_ACT[action])

    def heuristic(self, state):
        dx = self.goal_pos[0] - state.agent_pos[0]
        dy = self.goal_pos[1] - state.agent_pos[1]
        return math.hypot(dx, dy)

    def get_state(self, env, obs=None):
        ux = env.unwrapped
        opens: List[bool] = []
        for pos in self.door_positions:
            cell = ux.grid.get(*pos)
            opens.append(bool(cell.is_open))
        return DoorKeyState(
            agent_pos=tuple(ux.agent_pos),
            agent_dir=int(ux.agent_dir),
            is_carrying_key=ux.carrying is not None,
            door_open=tuple(opens),
        )

    def is_goal(self, state):
        return state.agent_pos == self.goal_pos

    def get_info(self, env):
        grid = env.unwrapped.grid
        doors: list[Tuple[int, int]] = []
        for x in range(grid.width):
            for y in range(grid.height):
                obj = grid.get(x, y)
                if obj is None:
                    continue
                if obj.type == "goal":
                    self.goal_pos = (x, y)
                elif obj.type == "door":
                    doors.append((x, y))
        doors.sort(key=lambda p: (p[1], p[0]))
        self.door_positions = tuple(doors)
