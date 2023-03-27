import jax
from jax import jacfwd, jacrev, jit, vmap
import jax.numpy as jnp
import flax


def pull_back_g(f):

    """ Returns the pull-back metric of f """

    phi = lambda x: f(jnp.array([x]))[0]
    Jphi = jax.jacrev(phi)
    g = lambda x: Jphi(x).T @ Jphi(x)

    return vmap(g, (0), 0)

def sqrt_det_pull_back_g(f):

    phi = lambda x: f(jnp.array([x]))[0]
    Jphi = jax.jacrev(phi)
    sqrt_det_g = lambda x: jnp.sqrt(jnp.linalg.det(Jphi(x).T @ Jphi(x)))

    return vmap(sqrt_det_g, (0), 0)


