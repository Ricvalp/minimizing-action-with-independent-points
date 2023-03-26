import jax
import jax.numpy as jnp
import flax
import flax.linen as nn
import action.numerical_solver_bvp  as ns
import matplotlib.pyplot as plt


class curve(nn.Module):

    N: int
    dt: float
    y_0: jnp.array # starting point
    y_1: jnp.array # ending point


    def setup(self):

        self.curve = self.param('curve',
                                solver_init(y_0=self.y_0.item(), y_1=self.y_1.item(), T=self.N*self.dt, N=self.N, sol=1, sigma=0.01), (self.N,)
                                )

    def __call__(self):

        return jnp.concatenate([self.y_0, self.curve, self.y_1])


def solver_init(y_0, y_1, T, N, sol, sigma):

    (x, ya), (x, yb) = ns.pendulum_bvp(y_0, y_1, T, N)

    if sol=='zeros':
        return nn.initializers.zeros

    if sol:
        def init(key, shape):
            return ya + jax.random.normal(key, shape=ya.shape)*sigma
        return init
    else:
        def init(key, shape):
            return yb + jax.random.normal(key, shape=yb.shape)*sigma
        return init




if __name__=="__main__":

    y_0 = jnp.array([0.])
    y_1 = jnp.array([2.])
    c = curve(N=100, dt=0.05, y_0=y_0, y_1=y_1)

    rng = jax.random.PRNGKey(42)
    params = c.init(rng)
    trajectory = c.apply(params)
    plt.plot(trajectory)
    plt.show()
    assert True
