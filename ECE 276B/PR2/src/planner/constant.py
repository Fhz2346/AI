_NEI6 = (
    (-1, 0, 0),
    (1, 0, 0),
    (0, -1, 0),
    (0, 1, 0),
    (0, 0, -1),
    (0, 0, 1),
)

_NEI26 = []
for di in (-1, 0, 1):
    for dj in (-1, 0, 1):
        for dk in (-1, 0, 1):
            if di == 0 and dj == 0 and dk == 0:
                continue
            _NEI26.append((di, dj, dk))
