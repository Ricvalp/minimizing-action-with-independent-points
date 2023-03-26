from diffrax import diffeqsolve, ODETerm, Tsit5, SaveAt
import jax.numpy as jnp
import matplotlib.pyplot as plt
import jax
import argparse


def pendulum_ivp(theta_zero, theta_dot_zero, T, N):
    
    def H_pend(q, p):

        m=1.
        l=1.
        g=10.

        return p*p/(2*m*l) + m*g*l*(1-jnp.cos(q))

    H_q = jax.grad(H_pend, argnums=0)
    H_p = jax.grad(H_pend, argnums=1)

    def f(t, y, args):
        q, p = y
        dy = H_p(q, p), -H_q(q, p)
        return dy

    t0 = 0.
    t1 = T
    dt0 = 0.1
    y0 = (theta_zero, theta_dot_zero)
    saveat = SaveAt(ts=jnp.linspace(t0, t1, N))

    term = ODETerm(f)
    solver = Tsit5()
    sol = diffeqsolve(term, solver, t0, t1, dt0, y0, saveat=saveat)

    return jnp.linspace(t0, t1, N), sol.ys[0], sol.ys[1]


if __name__=="__main__":
    t, theta = pendulum_ivp(0., 1., 5., 100)
    plt.plot(t, theta)
    plt.show()


"""
# q_dot = [(sol.ys[0][i+1]-sol.ys[0][i])/dt0 for i in range(99)]
# plt.scatter(sol.ys[0][1:100], q_dot)
# # plt.xlim((-7, 7))
# # plt.ylim((-2, 2))
# plt.xlabel("$q$")
# plt.ylabel("$q_{dot}$")
# # plt.savefig("pendulum")
# # plt.show()
# plt.scatter(sol.ys[0], sol.ys[1])
# plt.xlim((-4, 4))
# plt.ylim((-7, 7))
# plt.xlabel("$q$")
# plt.ylabel("$p$")
# plt.savefig("pendulum.png")
# plt.show()
"""