from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import math
from minigrid.core.actions import Actions

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
class DoorKeyRandomState:
    agent_pos: Tuple[int, int]
    agent_dir: int
    is_carrying_key: bool
    # One flag per door, in row-major order of door cells (same as door_positions).
    door_open: Tuple[bool, ...]
    # Map configuration (constant within an episode, varies across random envs).
    key_pos: Tuple[int, int]
    goal_pos: Tuple[int, int]


class DoorKeyProblemRandom(Problem):
    """
    DoorKey problem variant for Part B "Random Map".

    Unlike DoorKeyProblem (Part A), the MDP state includes the random map's
    key/goal locations so a single policy can generalize across the 36 envs
    without a per-map signature.
    """

    def __init__(self):
        self.actions = [
            Actions.forward,  # Move Forward
            Actions.left,  # Turn Left
            Actions.right,  # Turn Right
            Actions.pickup,  # Pick Key
            Actions.toggle,  # Toggle / Unlock Door
        ]
        self.goal_pos: Tuple[int, int] | None = None
        self.key_pos: Tuple[int, int] | None = None
        self.door_positions: Tuple[Tuple[int, int], ...] = ()

    def cost_fn(self, action):
        return step_cost(_ACTION_TO_ACT[action])

    def heuristic(self, state: DoorKeyRandomState):
        dx = state.goal_pos[0] - state.agent_pos[0]
        dy = state.goal_pos[1] - state.agent_pos[1]
        return math.hypot(dx, dy)

    def get_state(self, env, obs=None):
        if self.goal_pos is None or self.key_pos is None:
            raise RuntimeError(
                "DoorKeyProblemRandom.get_info(env) must be called before get_state()."
            )

        ux = env.unwrapped
        opens: List[bool] = []
        for pos in self.door_positions:
            cell = ux.grid.get(*pos)
            opens.append(bool(cell.is_open))

        return DoorKeyRandomState(
            agent_pos=tuple(ux.agent_pos),
            agent_dir=int(ux.agent_dir),
            is_carrying_key=ux.carrying is not None,
            door_open=tuple(opens),
            key_pos=self.key_pos,
            goal_pos=self.goal_pos,
        )

    def is_goal(self, state: DoorKeyRandomState):
        return state.agent_pos == state.goal_pos

    def get_info(self, env):
        grid = env.unwrapped.grid
        doors: list[Tuple[int, int]] = []
        goal_pos: Tuple[int, int] | None = None
        key_pos: Tuple[int, int] | None = None

        for x in range(grid.width):
            for y in range(grid.height):
                obj = grid.get(x, y)
                if obj is None:
                    continue
                if obj.type == "goal":
                    goal_pos = (x, y)
                elif obj.type == "key":
                    key_pos = (x, y)
                elif obj.type == "door":
                    doors.append((x, y))

        if goal_pos is None:
            raise RuntimeError("Goal not found in environment grid.")
        if key_pos is None:
            raise RuntimeError("Key not found in environment grid.")

        doors.sort(key=lambda p: (p[1], p[0]))
        self.door_positions = tuple(doors)
        self.goal_pos = goal_pos
        self.key_pos = key_pos
