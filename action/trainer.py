import jax
import jax.numpy as jnp
import flax.linen as nn
from flax.training import train_state
import matplotlib.pyplot as plt
import optax
import wandb
from action.numerical_solver_bvp import pendulum_bvp


class TrainerModule:

    def __init__(self, 
                 x: nn.Module,
                 g,
                 cfg,
                 lr=1e-4,
                 seed=43,
                 ):

        self.x = x
        self.g = g
        self.dt = cfg.dt
        self.lr = lr
        self.seed = seed
        self.wandb_log = cfg.wandb_log
        self.log_plot = cfg.log_plot

        self.init_model()
        self.create_functions()

        # if cfg.log_plot:
        #     self.init_ground_truth(y_0=cfg.y_0, y_1=cfg.y_1, T=cfg.N*cfg.dt, N=cfg.N)

    def create_functions(self):

        def train_step(state):

            loss_fn = lambda params: action(self.x, params, self.g, self.dt)
            loss, grads = jax.value_and_grad(loss_fn)(state.params)
            state = state.apply_gradients(grads=grads)
            return state, loss

        self.train_step = jax.jit(train_step) # train_step #

    def init_model(self):

        rng = jax.random.PRNGKey(self.seed)
        rng, init_rng = jax.random.split(rng, 2)
        params = self.x.init(init_rng)['params']

        optimizer = optax.adam(learning_rate=self.lr)

        # init_learning_rate = self.lr
        # decay_rate = 0.1
        # self.exponential_decay_scheduler = optax.exponential_decay(init_value=init_learning_rate, transition_steps=5000,
        #                                                     decay_rate=decay_rate, transition_begin=200, end_value=1e-04,
        #                                                     staircase=False)
        # optimizer = optax.adam(learning_rate=self.exponential_decay_scheduler)

        self.state = train_state.TrainState.create(apply_fn=self.x.apply, params=params, tx=optimizer)

    def train_model(self, num_epochs=100):

        for epoch_idx in range(num_epochs):

            self.state, loss = self.train_step(self.state)
            print(f"epoch: {epoch_idx}, loss: {loss}")

            # if self.wandb_log and self.log_plot:
                
            #     tr = self.state.apply_fn({'params': self.state.params}, self.t)
            #     plt.plot(self.t_gt, self.y_plot_a, color='r', label='svb solution 1')
            #     plt.plot(self.t_gt, self.y_plot_b, color='g', label='svb solution 2')
            #     plt.plot(self.t[:, 0], tr[:, 0], color='b', label='nef action minimization')
            #     plt.legend()
            #     wandb.log({"loss": loss, "plot": plt})

            # elif self.wandb_log:
            #     wandb.log({"loss": loss})

        return self.state
    
    # def init_ground_truth(self, y_0, y_1, T, N):
    #     (self.t_gt, self.y_plot_a), (self.t_gt, self.y_plot_b) = pendulum_bvp(y_0=y_0, y_1=y_1, T=T, N=N)




# def action(x, params, dt):

#     q = x.apply({'params': params})
#     q_dot = (q[1:]-q[:-1])/dt

#     actn = lagrangian(q[1:], q_dot)

#     assert True

#     return actn.mean()

# def lagrangian_single_point(q, q_dot):

#     K = 0.5 * q_dot @ jnp.identity(2) @ q_dot.T
#     V = -1/(q @ q.T)**2

#     return K - V
# lagrangian = jax.vmap(lagrangian_single_point, (0,0), 0)



def action(x, params, g, dt):

    q = x.apply({'params': params})
    # q = x.apply({'params': params}, t)

    q_dot = (q[1:]-q[:-1])/dt

    # action = jnp.sqrt((q_dot[:, None, :]@g(q)[:-1]@q_dot[:, :, None]).squeeze(-1))

    # I don't use the square root because its extremals are also extremals of the folloing functional
    action = (q_dot[:, None, :]@g(q)[:-1]@q_dot[:, :, None]).squeeze(-1)

    # # continuity regularizer
    # lamb = 6.
    # id = jnp.array([jnp.identity(2) for _ in range(q_dot.shape[0])])
    # regularizer = (q_dot[:, None, :]@ id @q_dot[:, :, None]).squeeze(-1)

    # assert True

    return action.mean() # + lamb*regularizer.mean()



if __name__=="__main__":

    from numerical_solver_bvp import pendulum_bvp
    

    key = jax.random.PRNGKey(42)
    x = jax.random.normal(key, (10, 2))
    l = lagrangian(x, x)

    assert True
