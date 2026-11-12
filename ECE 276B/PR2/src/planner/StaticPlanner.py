import heapq
import numpy as np
from .utils import segment_hits_any_block, point_in_any_block
from .GridMap import GridMap
from .constant import _NEI26


class StaticPlanner:
    __slots__ = ["map", "epsilon"]

    def __init__(self, map: GridMap):
        self.map = map

    def _heuristic(self, p, goal):
        # return float(np.linalg.norm(p - goal))
        return float(np.linalg.norm(p - goal, ord=1))
        

    def _weighted_astar(self, start_w, goal_w, epsilon=1.5):

        def goal_met(ijk):
            p = self.map.ijk2world(ijk)
            return np.sum((p - goal_w) ** 2) <= 0.1

        start_ijk = self.map.world2ijk(start_w)
        g = {start_ijk: 0.0}
        parent = {}
        h0 = self._heuristic(start_w, goal_w)
        OPEN = [(epsilon * h0, start_ijk)] # set of candidates for expansion (frontier)
        CLOSED = set() # set of states that have already been expanded
        while OPEN:
            _, p_ijk = heapq.heappop(OPEN)
            if p_ijk in CLOSED:
                continue
            if goal_met(p_ijk):
                print("num of nodes expanded: ", len(CLOSED))
                path_ijk = [p_ijk]
                cur = p_ijk
                while cur in parent:
                    cur = parent[cur]
                    path_ijk.append(cur)
                path_ijk.reverse()
                path_w = [self.map.ijk2world(ijk) for ijk in path_ijk]
                return np.array(path_w, dtype=np.float64)
            CLOSED.add(p_ijk)
            for di, dj, dk in _NEI26:
                n_ijk = (p_ijk[0] + di, p_ijk[1] + dj, p_ijk[2] + dk)
                if not self.map.in_bounds(n_ijk):
                    continue
                if n_ijk in CLOSED:
                    continue
                n_w = self.map.ijk2world(n_ijk)
                p_w = self.map.ijk2world(p_ijk)
                if point_in_any_block(n_w, self.map.blocks):
                    # if segment_hits_any_block(p_w, n_w, self.map.blocks):
                    continue
                step_size_w = self.map.resolution * np.sqrt(di * di + dj * dj + dk * dk)
                tentative = g[p_ijk] + step_size_w
                if tentative < g.get(n_ijk, np.inf):
                    g[n_ijk] = tentative
                    parent[n_ijk] = p_ijk
                    h = self._heuristic(n_w, goal_w)
                    f = tentative + epsilon * h
                    heapq.heappush(OPEN, (f, n_ijk))
        return None

    def plan(self, start, goal):
        start = np.asarray(start, dtype=np.float64).reshape(3)
        goal = np.asarray(goal, dtype=np.float64).reshape(3)

        internal = self._weighted_astar(start, goal)
        if internal is None or len(internal) == 0:
            return np.array([start])

        out = []
        if np.sum((start - internal[0]) ** 2) > 1e-12:
            if not segment_hits_any_block(start, internal[0], self.map.blocks):
                out.append(start.copy())
        out.extend([r.copy() for r in internal])

        last = out[-1]
        if np.sum((last - goal) ** 2) > 1e-12:
            if not segment_hits_any_block(last, goal, self.map.blocks):
                out.append(goal.copy())

        return np.array(out, dtype=np.float64)
