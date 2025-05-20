# PART (a): YOUR CODE BELOW ###############################################
# INSTRUCTIONS: Construct and solve the MPC problem using CVXPY.

cost = 0.0
constraints = []
constraints += [x_cvx[0] == x0] # IC

for k in range(N):
    constraints += [x_cvx[k + 1] == A @ x_cvx[k] + B @ u_cvx[k]] # dynamics
    constraints += [cvx.norm_inf(x_cvx[k]) <= rx] # safety
    constraints += [cvx.norm_inf(u_cvx[k]) <= ru] # safety
    cost += cvx.quad_form(x_cvx[k], Q) + cvx.quad_form(u_cvx[k], R)

if rf == 0:
    constraints += [x_cvx[N] == 0] # FC
else:
    constraints += [cvx.norm_inf(x_cvx[N]) <= rx] # FC

cost += cvx.quad_form(x_cvx[N], P)
# END PART (a) ############################################################

# PART (b): YOUR CODE BELOW #######################################
# INSTRUCTIONS: Simulate the closed-loop system for `max_steps`,
#               stopping early only if the problem becomes
#               infeasible or the state has converged close enough
#               to the origin. If the state converges, flag the
#               corresponding entry of `roa` with a value of `1`.
for k in range(max_steps):
    x, u, status = do_mpc(x, A, B, P, Q, R, N, rx, ru, rf)
    if status == "infeasible":
        break
    if np.linalg.norm(x, ord=np.inf) <= tol:
        roa[i, j] = 1.0
        break
    x = A@x[0, :] + B@u[0, :] # apply first MPC control
# END PART (b) ####################################################

# PART (a): WRITE YOUR CODE BELOW ###############################################
# You may find `jnp.where` to be useful; see corresponding numpy docstring:
# https://numpy.org/doc/stable/reference/generated/numpy.where.html
y, vy, phi, omega = state
p_y, p_vy, p_phi, p_omega = grad_value # co-state?

a = (p_vy*jnp.cos(phi)/self.m) - (p_omega*self.l/self.Iyy)
b = (p_vy*jnp.cos(phi)/self.m) + (p_omega*self.l/self.Iyy)

T1 = jnp.where(a < 0.0, self.max_thrust_per_prop, self.min_thrust_per_prop)
T2 = jnp.where(b < 0.0, self.max_thrust_per_prop, self.min_thrust_per_prop)

return jnp.stack([T1, T2])
#################################################################################

# PART (b): WRITE YOUR CODE BELOW ###############################################
y, vy, phi, omega = state
return jnp.max(jnp.stack([y-7, 3-y,
                    vy-1, -1-vy,
                    phi-(np.pi/12), -(np.pi/12)-phi,
                    omega-1, -1-omega]))
#################################################################################