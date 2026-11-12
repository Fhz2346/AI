def segment_intersects_aabb(p0, p1, bmin, bmax):
    """Return True if segment p0-p1 intersects the closed AABB [bmin, bmax]."""
    d = p1 - p0
    t0, t1 = 0.0, 1.0
    eps = 1e-9
    for i in range(3):
        if abs(d[i]) < eps:
            if p0[i] < bmin[i] - eps or p0[i] > bmax[i] + eps:
                return False
            continue
        inv = 1.0 / d[i]
        t_near = (bmin[i] - p0[i]) * inv
        t_far = (bmax[i] - p0[i]) * inv
        if t_near > t_far:
            t_near, t_far = t_far, t_near
        t0 = max(t0, t_near)
        t1 = min(t1, t_far)
        if t0 > t1 + 1e-12:
            return False
    return True


def point_in_aabb(p, bmin, bmax):
    return (
        bmin[0] <= p[0] <= bmax[0]
        and bmin[1] <= p[1] <= bmax[1]
        and bmin[2] <= p[2] <= bmax[2]
    )


def point_in_any_block(p, blocks):
    for k in range(blocks.shape[0]):
        if point_in_aabb(p, blocks[k, :3], blocks[k, 3:6]):
            return True
    return False


def point_in_boundary(p, boundary_row):
    return point_in_aabb(p, boundary_row[:3], boundary_row[3:6])


def segment_hits_any_block(p0, p1, blocks):
    for k in range(blocks.shape[0]):
        if segment_intersects_aabb(p0, p1, blocks[k, :3], blocks[k, 3:6]):
            return True
    return False


def path_has_collision(path, boundary, blocks):
    """True if any segment hits an obstacle or leaves the workspace boundary."""
    if path is None or len(path) < 2:
        return False
    b = boundary[0]
    for i in range(len(path) - 1):
        a, c = path[i], path[i + 1]
        if not point_in_boundary(a, b) or not point_in_boundary(c, b):
            return True
        if segment_hits_any_block(a, c, blocks):
            return True
    return False
