import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import hydra
from omegaconf import OmegaConf
import action
import wandb


@hydra.main(config_path='config', config_name='config')
def main(cfg):

    OmegaConf.set_struct(cfg, False)

    y_0 = jnp.array([cfg.y_0])
    y_1 = jnp.array([cfg.y_1])
    x = action.curve_2d(N=cfg.N, dt=cfg.dt, y_0=y_0, y_1=y_1)

    rng = jax.random.PRNGKey(43)
    rng, init_rng, data_rng = jax.random.split(rng, 3)
    init_points = jax.random.normal(data_rng, shape=(100, 2))

    num_hidden = [5, 8, 10]

    decoder = action.decoder(num_hidden=num_hidden)
    f_params = decoder.init(init_rng, init_points)['params']


    f = lambda x: decoder.apply({'params': f_params}, x)
    g = action.pull_back_g(f=f)


    sqrt_det_g = action.sqrt_det_pull_back_g(f=f)

    xlim = [-5., 5.]
    ylim = [-5., 5.]
    N = 50
    x_ = jnp.linspace(xlim[0], xlim[1], N)
    y_ = jnp.linspace(ylim[0], ylim[1], N)
    xx, yy = jnp.meshgrid(x_, y_)
    points = jnp.concatenate(jnp.stack((xx, yy), axis=-1))
    # img = sqrt_det_g(points).reshape(N, N)
    # plt.imshow(img)
    # plt.show()
    detgs = sqrt_det_g(points)


    if cfg.wandb_log:
        wandb.init(project="experiments-independent-points-action", entity='ricvalp')

    trainer = action.TrainerModule(x=x, cfg=cfg, g=g, lr=1e-3)
    state = trainer.train_model(num_epochs=cfg.num_epochs)

    tr = state.apply_fn({'params': state.params})


    plt.scatter(points[:, 0], points[:, 1], s=55, marker='s', c=detgs)
    plt.colorbar()
    plt.scatter(tr[:, 0], tr[:, 1], color='r', s=3, label='action minimization')
    plt.legend()
    plt.show()

    assert True


if __name__=="__main__":
    main()




"""

# # gt_t, gt_trajectory = action.pendulum_ivp(theta_zero=x_0[0][0], theta_dot_zero=((tr[1]-tr[0])/(t[1]-t[0]))[0], T=T, N=100)
# (x, y_plot_a), (x, y_plot_b) = action.pendulum_bvp(y_0=y_0.item(), y_1=y_1.item(), T=(cfg.N+2)*cfg.dt, N=cfg.N+2)
# t, y_plot_c, y_dot_c = action.pendulum_ivp(theta_zero=y_0.item(), theta_dot_zero=(tr[1]-tr[0])/cfg.dt, T=(cfg.N+2)*cfg.dt, N=cfg.N+2)

# plt.plot(x, y_plot_a, color='r', label='svb solution 1')
# plt.plot(x, y_plot_b, color='g', label='svb solution 2')
# plt.plot(x, y_plot_c, color='y', label= 'ivp solver')


### ### ###


# y_dot_a = (y_plot_b[1:]-y_plot_b[:-1])/cfg.dt
# E_gt_a = 0.5*y_dot_a*y_dot_a + 10*(1-jnp.cos(y_plot_a[:-1]))

# y_dot_b = (y_plot_b[1:]-y_plot_b[:-1])/cfg.dt
# E_gt_b = 0.5*y_dot_b*y_dot_b + 10*(1-jnp.cos(y_plot_b[:-1]))

# E_gt_c = 0.5*y_dot_c*y_dot_c + 10*(1-jnp.cos(y_plot_c))

# tr_dot = (tr[1:]-tr[:-1])/cfg.dt
# E = 0.5*tr_dot*tr_dot + 10*(1-jnp.cos(tr[:-1]))

# plt.plot(x[1:], E, color='b', label='action minimization')
# # plt.plot(x[1:], E_gt_a, color='r', label='svb solution 1')
# plt.plot(x[1:], E_gt_b, color='g', label='svb solution 2')
# plt.plot(x, E_gt_c, color='y', label='ivb solution')
# plt.ylabel("Energy")
# plt.xlabel("$t$")
# plt.legend()
# plt.show()

"""