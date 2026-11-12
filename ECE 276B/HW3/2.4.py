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

# Variable vector z:
# z = [V1, V2, V3, V4,
#      Q11, Q12, Q21, Q22, Q31, Q32, Q41, Q42]
n_V = n_states
n_Q = n_states * n_actions
n_vars = n_V + n_Q

def q_index(x, u):
    """
    x = 0,1,2,3
    u = 0,1
    """
    return n_V + x * n_actions + u

# Objective:
# maximize sum(V) <=> minimize -sum(V)
c = np.zeros(n_vars)
c[:n_states] = -1

A_ub = []
b_ub = []

A_eq = []
b_eq = []

# Inequality constraints:
# V(x) <= Q(x,u)
# equivalent to V(x) - Q(x,u) <= 0
for x in range(n_states):
    for u in range(n_actions):
        row = np.zeros(n_vars)

        row[x] = 1
        row[q_index(x, u)] = -1

        A_ub.append(row)
        b_ub.append(0)

# Equality constraints:
# Q(x,u) = L(x,u) + gamma * P^u[x,:] @ V
# equivalent to Q(x,u) - gamma * P^u[x,:] @ V = L(x,u)
for x in range(n_states):
    for u in range(n_actions):
        row = np.zeros(n_vars)

        # coefficient for Q(x,u)
        row[q_index(x, u)] = 1

        # coefficients for V
        row[:n_states] -= gamma * P[u][x, :]

        A_eq.append(row)
        b_eq.append(L[x, u])

A_ub = np.array(A_ub)
b_ub = np.array(b_ub)

A_eq = np.array(A_eq)
b_eq = np.array(b_eq)

result = linprog(
    c,
    A_ub=A_ub,
    b_ub=b_ub,
    A_eq=A_eq,
    b_eq=b_eq,
    bounds=[(None, None)] * n_vars,
    method="highs"
)

print(result.success)
print(result.message)

z_star = result.x

V_star = z_star[:n_states]
Q_star = z_star[n_states:].reshape(n_states, n_actions)

print("V* =")
print(V_star)

print("Q* =")
print(Q_star)

print("pi* =")
print(np.argmin(Q_star, axis=1) + 1)