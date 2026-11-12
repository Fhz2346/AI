import numpy as np
from scipy.optimize import linprog

gamma = 0.9

P1 = np.array([
    [0,   1,   0,   0],
    [0,   0,   1,   0],
    [1/4, 3/4, 0,   0],
    [1/2, 0,   0,   1/2]
])

P2 = np.array([
    [1/2, 0,   1/2, 0],
    [0,   1,   0,   0],
    [0,   0,   1/2, 1/2],
    [0,   1,   0,   0]
])

L = np.array([
    [-10, -8],
    [1,    8],
    [1,    2],
    [6,    3]
])

P = [P1, P2]

n_states = 4
n_actions = 2

# Objective:
# maximize sum(V)  <=>  minimize -sum(V)
c = -np.ones(n_states)

A_ub = []
b_ub = []

for x in range(n_states):
    for u in range(n_actions):
        e_x = np.zeros(n_states)
        e_x[x] = 1

        # V(x) - gamma * P^u[x, :] @ V <= L[x, u]
        row = e_x - gamma * P[u][x, :]
        A_ub.append(row)
        b_ub.append(L[x, u])

A_ub = np.array(A_ub)
b_ub = np.array(b_ub)

result = linprog(
    c,
    A_ub=A_ub,
    b_ub=b_ub,
    bounds=[(None, None)] * n_states,
    method="highs"
)

print(result.success)
print(result.message)
print("V* =", result.x)
print("objective =", -result.fun)
