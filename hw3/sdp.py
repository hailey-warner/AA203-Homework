import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt

A = np.array([[0.9, 0.6],
              [0.0, 0.8]])
r_x = 5.0

M = cp.Variable((2,2), PSD=True)

block = cp.bmat([[M, A @ M],
                [M @ A.T, M]])

constraints = [block >> 0,
               M <= r_x**2 *np.eye(2)]

J = cp.Maximize(cp.log_det(M))

prob = cp.Problem(J, constraints)
prob.solve(solver=cp.SCS, verbose=True)

W = np.linalg.inv(M.value)
print("W*:", np.round(W, 3))

def generate_ellipsoid_points(Mmat, num=200):
    L = np.linalg.cholesky(Mmat)
    θ = np.linspace(0, 2*np.pi, num)
    u = np.stack([np.cos(θ), np.sin(θ)], axis=1)
    return u @ L.T


ell_XT   = generate_ellipsoid_points(M.value)
ell_AXT = (A @ ell_XT.T).T
ell_X    = generate_ellipsoid_points((r_x**2)*np.eye(2))

plt.figure(figsize=(6,6))
plt.plot(ell_XT[:,0],  ell_XT[:,1],  label='XT')
plt.plot(ell_AXT[:,0], ell_AXT[:,1], label='A XT')
plt.plot(ell_X[:,0],   ell_X[:,1],   label=r'|x| <= r_x')
plt.axis('equal')
plt.legend()
plt.xlabel('x₁'); plt.ylabel('x₂')
plt.title('Ellipsoids')
plt.grid(True)
plt.show()
