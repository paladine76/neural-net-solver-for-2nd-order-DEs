import numpy as np
import tensorflow as tf
from .base import DEProblem


class HeatEquationProblem(DEProblem):
    """
    1D Heat Equation (2nd-order PDE, parabolic):
        u_t = u_xx,   x in [x_lb, x_ub],   t in [0, t_ub]
    
    Initial conditions:
        u(x, 0) = cos(pi * x) + sin(pi * x)

    Point boundary constraints (from the true solution at t = 1):
        u(0, 1) = exp(-1)
        u(-1, 1) = -exp(-1)
        du/dt(0, 1) = -exp(-1)

    Analytic:
        u(x, t) = (cos(pi * x) + sin(pi * x)) * exp(-pi^2 * t)
    
    Note: the notebook uses exp(-t), which requires pi^2 = 1, i.e. the
    equation is u_t = (1/pi^2) * u_xx. We use the standard form here
    with the corrected true solution. Set `diffusivity=1/pi**2` to
    match the notebook's convention exactly.
    """

    def __init__(
            self,
            x_lb: float = -5.0,
            x_ub: float = 5.0,
            t_ub: float = 5.0,
            diffusivity: float = 1.0
    ):
        self.x_lb = x_lb
        self.x_ub = x_ub
        self.t_ub = t_ub
        self.D = diffusivity
        self._xt_01 = tf.Variable([[0.0, 1.0]], dtype=tf.float32)

    def residual(self, net: callable, xt: tf.Tensor):
        """PDE residual: u_t - D*u_xx evaluated at interior collocation points."""
        x = xt[:, 0:1]
        t = xt[:, 1:2]

        with tf.GradientTape() as t2:
            t2.watch(x)
            with tf.GradientTape(persistent=True) as t1:
                t1.watch(x)
                t1.watch(t)
                u = net(tf.concat([x, t], axis=1))
            u_x = t1.gradient(u, x)
            u_t = t1.gradient(u, t)
            del t1
        u_xx = t2.gradient(u_x, x)

        return u_t - self.D * u_xx

    def constraints(self, net: callable):
        """
        Constraints:
            1. IC:       u(x, 0)     = cos(pi * x) + sin(pi * x) sampled at N_ic points
            2. Point BC: u(0, 1)     = exp(-1)
            3. Point BC: u(-1, 1)    = -exp(-1)
            4. Point BC: du/dt(0, 1) = -exp(-1)
        """
        # --- IC over a fixed grid ---
        N_ic = 100
        x_ic = tf.cast(
            tf.linspace(self.x_lb, self.x_ub, N_ic)[:, tf.newaxis], tf.float32
        )
        t_ic = tf.zeros_like(x_ic)
        xt_ic = tf.concat([x_ic, t_ic], axis=1)
        u_ic_pred = net(xt_ic)
        u_ic_true = tf.cos(np.pi * x_ic) + tf.sin(np.pi * x_ic)

        # --- Spatial BCs sampled over t: u(x_lb, t) and u(x_ub, t) ---
        N_bc = 50
        t_bc = tf.cast(tf.linspace(0.0, self.t_ub, N_bc)[:, tf.newaxis], tf.float32)

        x_left = tf.fill([N_bc, 1], self.x_lb)
        u_left_pred = net(tf.concat([x_left, t_bc], axis=1))
        u_left_true = (np.cos(np.pi * self.x_lb) + np.sin(np.pi * self.x_lb)) * tf.exp(-self.D * np.pi**2 * t_bc)

        x_right = tf.fill([N_bc, 1], self.x_ub)
        u_right_pred = net(tf.concat([x_right, t_bc], axis=1))
        u_right_true = (np.cos(np.pi * self.x_ub) + np.sin(np.pi * self.x_ub)) * tf.exp(-self.D * np.pi**2 * t_bc)

        # --- Point BCs at t=1 ---
        xt_01 = tf.constant([[0.0, 1.0]], dtype=tf.float32)
        u_01_pred = net(xt_01)
        u_01_true = tf.constant([[np.exp(-1.0)]], dtype=tf.float32)

        xt_m11 = tf.constant([[-1.0, 1.0]], dtype=tf.float32)
        u_m11_pred = net(xt_m11)
        u_m11_true = tf.constant([[-np.exp(-1.0)]], dtype=tf.float32)

        with tf.GradientTape() as tape:
            tape.watch(self._xt_01)
            u_01_v = net(self._xt_01)
        grads = tape.gradient(u_01_v, self._xt_01)
        ut_01_pred = grads[:, 1:2]
        ut_01_true = tf.constant([[-np.exp(-1.0)]], dtype=tf.float32)

        # Assemble all residuals (everything expressed as pred - true = 0)
        all_res = tf.concat([
            u_ic_pred - u_ic_true,
            u_left_pred - u_left_true,
            u_right_pred - u_right_true,
            u_01_pred - u_01_true,
            u_m11_pred - u_m11_true,
            ut_01_pred - ut_01_true,
        ], axis=0)

        return all_res, tf.zeros_like(all_res)
    
    def domain(self):
        return {
            'x_lb': self.x_lb, 'x_ub': self.x_ub,
            't_lb': 0.0, 't_ub': self.t_ub
        }
    
    def true_solution(self, x, t):
        return (np.cos(np.pi * x) + np.sin(np.pi * x)) * np.exp(-self.D * np.pi ** 2 * t)