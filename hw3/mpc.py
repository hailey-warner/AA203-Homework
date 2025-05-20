import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt


A = np.array([[0.9, 0.6],
              [0.0, 0.8]])
B = np.array([[0.],
              [1.]])
r_x = 5.0
r_u = 1.0
N = 4
T = 15

Q = np.eye(2)
R = np.eye(1)
P = np.eye(2)

X = cp.Variable((2, N+1))
U = cp.Variable((1, N))
x0 = cp.Parameter(2)

cost_terms = []
constraints = []
planned_trajectories = []

# IC
constraints += [X[:,0] == x0]

for t in range(N):
    constraints += [X[:,t+1] == A @ X[:,t] + B @ U[:,t]]
    constraints += [cp.norm(X[:,t], 2) <= r_x]
    constraints += [cp.norm(U[:,t], 2) <= r_u]
    cost_terms  += [cp.quad_form(X[:,t], Q), cp.quad_form(U[:,t], R)]

# FC
cost_terms += [cp.quad_form(X[:,N], P)]
constraints += [cp.norm(X[:,N], 2) <= r_x]

# objective
objective = cp.Minimize(sum(cost_terms))
mpc_prob = cp.Problem(objective, constraints)

x_actual = np.zeros((2, T+1))
u_actual = np.zeros((1, T))
x_actual[:,0] = np.array([0.0, -4.5]) # IC

for k in range(T):
    x0.value = x_actual[:,k] # update state
    mpc_prob.solve(solver=cp.ECOS, warm_start=True)
    
    u_k = U[:,0].value # optimal first control
    x_pred = X.value.copy() # predicted states
    u_actual[:,k] = u_k
    planned_trajectories.append(x_pred)
    x_actual[:,k+1] = A @ x_actual[:,k] + B.flatten() * u_k # dynamics rollout

# states & planned trajectories
plt.figure(figsize=(6,6))
plt.plot(x_actual[0,:], x_actual[1,:], 'b-o', linewidth=2, label='actual trajectory')
for x_pred in planned_trajectories:
    plt.plot(x_pred[0,:], x_pred[1,:], 'r--', alpha=0.3)
plt.xlabel('$x_1$'); plt.ylabel('$x_2$')
plt.title('MPC actual vs. planned trajectories')
plt.legend(); plt.grid(True); plt.axis('equal')

# control inputs
plt.figure()
plt.step(np.arange(T), u_actual.flatten(), where='post')
plt.xlabel('$t$'); plt.ylabel('$u_t$')
plt.title('MPC control trajectory')
plt.ylim([-r_u*1.2, r_u*1.2])
plt.grid(True)
plt.show()
