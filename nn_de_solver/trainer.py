import time
import tensorflow as tf


class Trainer:
    """
    Trains an MLP to satisfy a DEProblem via collocation loss minimization.

    Loss = sqrt(mean([MSE_residual, MSE_constraints]))
    """

    def __init__(self, net, problem, lr: float = 1e-3, use_lr_decay: bool = False):
        self.net = net
        self.problem = problem
        if use_lr_decay:
            lr = tf.keras.optimizers.schedules.ExponentialDecay(
                initial_learning_rate=lr, decay_steps=5_000, decay_rate=0.5
            )
        self.optimizer = tf.optimizers.Adam(lr)

    def _sample_collocation(self, batch_size: int):
        """Sample collocation points from the problem domain."""
        bounds = self.problem.domain()
        if 'x_lb' in bounds:
            xs = tf.random.uniform((batch_size, 1), bounds['x_lb'], bounds['x_ub'])
            ts = tf.random.uniform((batch_size, 1), bounds['t_lb'], bounds['t_ub'])

            return tf.concat([xs, ts], axis=1)
        else:
            return tf.random.uniform((batch_size, 1), bounds['lb'], bounds['ub'])

    def loss(self, batch_size: int):
        xt = self._sample_collocation(batch_size)        
        residual = self.problem.residual(self.net, xt)
        pred, target = self.problem.constraints(self.net)

        mse_res = tf.reduce_mean(residual ** 2)
        mse_cst = tf.reduce_sum((pred - target) ** 2)

        return tf.sqrt(tf.reduce_mean([mse_res, mse_cst]))
    
    @tf.function
    def _step(self, batch_size: int):
        with tf.GradientTape() as tape:
            loss_val = self.loss(batch_size)
        
        grads = tape.gradient(loss_val, self.net.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.net.trainable_variables))

        return loss_val
    
    def train(
            self,
            steps: int = 10_000,
            batch_size: int = 50,
            tol: float = 0.05,
            log_every: int = 1_000
    ):
        """
        Run training loop.

        Returns list of mean losses recorded at each log interval.
        """

        history, cum_loss = [], 0.0
        t0 = time.time()

        for i in range(steps):
            loss_val = self._step(batch_size).numpy()
            cum_loss += loss_val

            if (i + 1) % log_every == 0:
                mean_loss = cum_loss / log_every
                history.append(mean_loss)
                elapsed = time.time() - t0
                print(f"step {i+1:>6d} | mean loss: {mean_loss:.4f} | {elapsed:.1f}s")
                cum_loss = 0.0
            
            if loss_val <= tol:
                print(f"converged at step {i+1} with loss {loss_val:.4f}")
                history.append(loss_val)
                break
        
        return history