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

    if cfg.wandb_log:
        wandb.init(project="experiments-independent-points-action", entity='ricvalp')

    trainer = action.TrainerModule(x=x, cfg=cfg, lr=1e-4)
    state = trainer.train_model(num_epochs=cfg.num_epochs)

    tr = state.apply_fn({'params': state.params})
    # # gt_t, gt_trajectory = action.pendulum_ivp(theta_zero=x_0[0][0], theta_dot_zero=((tr[1]-tr[0])/(t[1]-t[0]))[0], T=T, N=100)
    # (x, y_plot_a), (x, y_plot_b) = action.pendulum_bvp(y_0=y_0.item(), y_1=y_1.item(), T=(cfg.N+2)*cfg.dt, N=cfg.N+2)
    # t, y_plot_c, y_dot_c = action.pendulum_ivp(theta_zero=y_0.item(), theta_dot_zero=(tr[1]-tr[0])/cfg.dt, T=(cfg.N+2)*cfg.dt, N=cfg.N+2)

    # plt.plot(x, y_plot_a, color='r', label='svb solution 1')
    # plt.plot(x, y_plot_b, color='g', label='svb solution 2')
    # plt.plot(x, y_plot_c, color='y', label= 'ivp solver')
    plt.scatter(tr[:, 0], tr[:, 1], color='b', label='action minimization')
    plt.legend()
    plt.show()

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

    assert True


if __name__=="__main__":
    main()
