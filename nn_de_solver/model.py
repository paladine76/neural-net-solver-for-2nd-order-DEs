import tensorflow as tf


class MLP(tf.Module):
    """
    Configurable MLP: input_dim -> [hidden layers] -> output_dim.

    hidden: tuple of layer widths, e.g. (128, 64) or (256, 256, 256, 256).
    activation: 'sigmoid' (default, good for ODEs) or 'tanh' (better for PDEs).
    """

    _activations = {
        'sigmoid': tf.nn.sigmoid,
        'tanh': tf.nn.tanh,
        'relu': tf.nn.relu
    }

    def __init__(
            self, 
            input_dim: int, 
            hidden: tuple = (128, 64), 
            output_dim: int = 1,
            activation: str = 'sigmoid'
    ):
        super().__init__()
        self.act = self._activations[activation]
        layer_sizes = [input_dim] + list(hidden) + [output_dim]
        self.Ws = [
            tf.Variable(tf.initializers.GlorotUniform()(
                shape=[layer_sizes[i], layer_sizes[i+1]])) for i in range(len(layer_sizes)-1)
        ]
        self.bs = [
            tf.Variable(tf.random.normal([layer_sizes[i+1]])) for i in range(len(layer_sizes)-1)
        ]

    @tf.function
    def __call__(self, x: tf.Tensor):
        h = x
        for W, b in zip(self.Ws[:-1], self.bs[:-1]):
            h = self.act(h @ W + b)

        return h @ self.Ws[-1] + self.bs[-1]
    
    @property
    def trainable_variables(self):
        return self.Ws + self.bs