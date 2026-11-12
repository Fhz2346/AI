from collections import deque
import numpy as np
from .utils import point_in_boundary, point_in_any_block
from .constant import _NEI6


class GridMap:

    def __init__(self, boundary, blocks, resolution=0.25):
        self.boundary = boundary
        self.blocks = blocks
        self.resolution = resolution

        b = boundary[0]
        self.xmin, self.ymin, self.zmin = b[0], b[1], b[2]
        xmax, ymax, zmax = b[3], b[4], b[5]
        res = resolution
        self.nx = int(np.floor((xmax - self.xmin) / res + 1e-9)) + 1
        self.ny = int(np.floor((ymax - self.ymin) / res + 1e-9)) + 1
        self.nz = int(np.floor((zmax - self.zmin) / res + 1e-9)) + 1

    def in_bounds(self, ijk):
        i, j, k = ijk
        return 0 <= i < self.nx and 0 <= j < self.ny and 0 <= k < self.nz

    def ijk2world(self, ijk):
        i, j, k = ijk
        res = self.resolution
        return np.array(
            [self.xmin + i * res, self.ymin + j * res, self.zmin + k * res],
            dtype=np.float64,
        )

    def world2ijk(self, p):
        res = self.resolution
        i = int(np.round((p[0] - self.xmin) / res))
        j = int(np.round((p[1] - self.ymin) / res))
        k = int(np.round((p[2] - self.zmin) / res))
        i = int(np.clip(i, 0, self.nx - 1))
        j = int(np.clip(j, 0, self.ny - 1))
        k = int(np.clip(k, 0, self.nz - 1))
        return i, j, k

    def is_free(self, ijk):
        p = self.ijk2world(ijk)
        return point_in_boundary(p, self.boundary[0]) and not point_in_any_block(
            p, self.blocks
        )

    def nearest_free(self, ijk0):
        """BFS from (i0,j0,k0) over 6-neighbors until a free cell is found."""
        if self.is_free(ijk0):
            return ijk0
        seen = {ijk0}
        q = deque([ijk0])
        while q:
            ijk = q.popleft()
            for di, dj, dk in _NEI6:
                ijk_neighbor = (ijk[0] + di, ijk[1] + dj, ijk[2] + dk)
                if not self.in_bounds(ijk_neighbor):
                    continue
                if ijk_neighbor in seen:
                    continue
                seen.add(ijk_neighbor)
                if self.is_free(ijk_neighbor):
                    return ijk_neighbor
                q.append(ijk_neighbor)
        return None
