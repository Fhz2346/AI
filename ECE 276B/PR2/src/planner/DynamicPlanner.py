import heapq
import math
import numpy as np
from .utils import segment_hits_any_block, point_in_any_block
from .GridMap import GridMap
from .constant import _NEI26


class DynamicPlanner:

    def __init__(self, map: GridMap, epsilon = 2.5):
        self.map = map
        self.g = {}
        self.parent = {}
        self.closed = set()
        self.open_heap = []
        self._start_w = None
        self._epsilon = epsilon
        self._cache_ready = False

    def _heuristic(self, p_w, goal_w):
        return float(np.linalg.norm(p_w - goal_w))

    def _goal_met(self, p_ijk, goal_w):
        p_w = self.map.ijk2world(p_ijk)
        return np.sum((p_w - goal_w) ** 2) <= 0.1

    def _prepare_cold(self, start_w, goal_w):
        start_ijk = self.map.world2ijk(start_w)
        self.g = {start_ijk: 0.0}
        self.parent = {}
        # New search: must not reuse closed from a previous goal (different expansion order).
        self.closed = set()
        h0 = self._heuristic(start_w, goal_w)
        self.open_heap = [(self._epsilon * h0, start_ijk)]

    def _prepare_warm(self, start_w, goal_w):
        start_ijk = self.map.world2ijk(start_w)
        self.g[start_ijk] = 0.0
        self.closed = set()
        self.open_heap = [
            (
                gv + self._epsilon * self._heuristic(self.map.ijk2world(ijk), goal_w),
                ijk,
            )
            for ijk, gv in self.g.items()
        ]
        heapq.heapify(self.open_heap)

    def _search(self, goal_w):
        while self.open_heap:
            _, p_ijk = heapq.heappop(self.open_heap)
            if p_ijk in self.closed:
                continue
            if self._goal_met(p_ijk, goal_w):
                path_ijk = [p_ijk]
                cur = p_ijk
                while cur in self.parent:
                    cur = self.parent[cur]
                    path_ijk.append(cur)
                path_ijk.reverse()
                path_w = [self.map.ijk2world(ijk) for ijk in path_ijk]
                return np.array(path_w, dtype=np.float64)
            self.closed.add(p_ijk)
            for di, dj, dk in _NEI26:
                n_ijk = (p_ijk[0] + di, p_ijk[1] + dj, p_ijk[2] + dk)
                if not self.map.in_bounds(n_ijk):
                    continue
                if n_ijk in self.closed:
                    continue
                n_w = self.map.ijk2world(n_ijk)
                step_size_w = self.map.resolution * math.sqrt(di * di + dj * dj + dk * dk)
                if point_in_any_block(n_w, self.map.blocks):
                    continue
                tentative = self.g[p_ijk] + step_size_w
                if tentative < self.g.get(n_ijk, np.inf):
                    self.g[n_ijk] = tentative
                    self.parent[n_ijk] = p_ijk
                    h = self._heuristic(n_w, goal_w)
                    f = tentative + self._epsilon * h
                    heapq.heappush(self.open_heap, (f, n_ijk))
        return None

    def plan(self, start, goal, use_incremental=True, epsilon=None):
        start = np.asarray(start, dtype=np.float64).reshape(3)
        goal = np.asarray(goal, dtype=np.float64).reshape(3)
        blocks = self.map.blocks

        can_warm = self._cache_ready
        if can_warm:
            self._prepare_warm(start, goal)
        else:
            self._prepare_cold(start, goal)

        internal = self._search(goal)
        self._start_w = start.copy()
        self._cache_ready = True

        if internal is None or len(internal) == 0:
            return np.array([start])

        out = []
        if np.sum((start - internal[0]) ** 2) > 1e-12:
            if not segment_hits_any_block(start, internal[0], blocks):
                out.append(start.copy())
        out.extend([r.copy() for r in internal])

        last = out[-1]
        if np.sum((last - goal) ** 2) > 1e-12:
            if not segment_hits_any_block(last, goal, blocks):
                out.append(goal.copy())

        return np.array(out, dtype=np.float64)
