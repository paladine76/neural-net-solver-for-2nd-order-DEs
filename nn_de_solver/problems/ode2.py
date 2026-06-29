import numpy as np
import tensorflow as tf
from .base import DEProblem


class ODE2Problem(DEProblem):
    """
    2nd-order linear ODE:
        y" + a*y' + b*y = f(x)

    with initial conditions y(x0) = y0, y'(x0) = dy0.

    Default demo: y" + 6y' + 9y = 0, y(0) = 2, y'(0) = 0
    Analytic:     y(x) = (2 + 6x) * exp(-3x) 
    """

    def __init__(
            self, 
            a: float = 6.0,
            b: float = 9.0,
            f = None,
            x0: float = 0.0,
            y0: float = 2.0,
            dy0: float = 0.0,
            lb: float = -0.5,
            ub: float = 2.5
    ):
        self.a, self.b = a, b
        self.f = f if f is not None else lambda x: tf.zeros_like(x)
        self.x0, self.y0, self.dy0 = x0, y0, dy0
        self._lb, self._ub = lb, ub
    
    def residual(self, net: callable, x: tf.Tensor):
        with tf.GradientTape() as t2:
            t2.watch(x)
            with tf.GradientTape() as t1:
                t1.watch(x)
                u = net(x)
            u_x = t1.gradient(u, x)
        u_xx = t2.gradient(u_x, x)

        return u_xx + self.a * u_x + self.b * u - self.f(x)
    
    def constraints(self, net: callable):
        x0 = tf.constant([[self.x0]], dtype=tf.float32)
        with tf.GradientTape() as tape:
            tape.watch(x0)
            u0 = net(x0)
        u_x0 = tape.gradient(u0, x0)
        pred = tf.concat([u0, u_x0], axis=0)
        target = tf.constant([[self.y0], [self.dy0]], dtype=tf.float32)

        return pred, target
    
    def domain(self):
        return {'lb': self._lb, 'ub': self._ub}
    
    def true_solution(self, x):
        # Valid only for the default critically-damped case: a=6, b=9
        # Override this method for other coefficients.
        assert self.a == 6.0 and self.b == 9.0, (
            "true solution is only implemented for the default a = 6 and b = 9 case."
        )

        return (self.y0 + (self.dy0 - self.y0 * (-self.a / 2)) * x) * np.exp((-self.a / 2) * x)