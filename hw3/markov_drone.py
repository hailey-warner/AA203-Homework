import numpy as np
import matplotlib.pyplot as plt
np.random.seed(0)

n = 20
sigma = 10
gamma = 0.95
x_goal = (19, 9)
x_eye = (15, 15)
A = ['up', 'down', 'left', 'right']
X = [(i, j) for i in range(n) for j in range(n)]

def w(x):
    return np.exp(-np.linalg.norm(np.array(x)-x_eye)**2 / (2*sigma**2))

def R(x):
    return 1.0 if x == x_goal else 0.0

def step(x, a):
    d = {'up': (0, 1), 'down': (0, -1), 'left': (-1, 0), 'right': (1, 0)}[a]
    nx = min(max(x[0] + d[0], 0), n - 1)
    ny = min(max(x[1] + d[1], 0), n - 1)
    return (nx, ny)

def neighbors(x):
    return list({step(x,a) for a in A})

def p(x_next, x, a):
    k = len(neighbors(x)) # 2, 3, or 4
    if x_next == step(x, a): # intended
        return (1 - w(x)) + w(x)/k
    elif x_next in neighbors(x): # random
        return w(x)/k
    else: # impossible
        return 0.0

def value_iteration(tol=1e-6):
    V = np.zeros((n, n))
    while True:
        print("iterating...")
        V_next = np.zeros_like(V)
        for i, j in X:
            V_next[i, j] = max(
                sum(
                    p(y, (i, j), a) * (R(y) + gamma * V[y])
                    for y in neighbors((i, j))
                )
                for a in A
            )
        if np.max(np.abs(V_next - V)) < tol:
            break
        V = V_next
    return V

V = value_iteration()

'''
plt.imshow(V.T, origin='lower', extent=(0, n-1, 0, n-1), aspect='equal')
plt.colorbar()
plt.xlabel('x₁')
plt.ylabel('x₂')
plt.title('V(x)')
plt.xticks([])
plt.yticks([])
plt.scatter([x_eye[0]], [x_eye[1]], marker='*', s=200, c='white', label='x_eye')
plt.scatter([x_goal[0]], [x_goal[1]], marker='*', s=200, c='red', label='x_goal')
plt.legend()
plt.show()
'''

policy = np.zeros((n, n), int)
for i, j in X:
    Q = [sum(p(y, (i, j), a) * (R(y) + gamma * V[y]) for y in neighbors((i, j))) for a in A]
    policy[i, j] = np.argmax(Q)

trajectory = []
state = (0, 19)
trajectory.append(state)
while state != x_goal:
    a = A[policy[state]]
    next_states = [step(state, act) for act in A]
    probs = np.array([p(ns, state, a) for ns in next_states])
    probs /= probs.sum()
    choice = np.random.choice(len(next_states), p=probs)
    state = next_states[choice]
    trajectory.append(state)

tx, ty = zip(*trajectory)
plt.figure(figsize=(6,6))
plt.imshow(policy.T, origin='lower', extent=(0, n-1, 0, n-1), aspect='equal')
plt.colorbar(label='0=up,1=down,2=left,3=right')
plt.plot(tx, ty, marker='o', markersize=2, linewidth=1, color='white')
plt.scatter([x_eye[0]], [x_eye[1]], marker='*', s=100, color='white')
plt.scatter([x_goal[0]], [x_goal[1]], marker='*', s=100, color='red')
plt.xlabel('x₁')
plt.ylabel('x₂')
plt.title('Pi(x)')
plt.show()
