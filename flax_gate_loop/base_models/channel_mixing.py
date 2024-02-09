from flax import linen as nn
from jax.nn import gelu

class ChannelMixing(nn.Module):
    d_models: int
    dropout: float
    eps: float
    def setup(self):
        self.n_layers = len(self.d_models) - 1
        self.dense_layers = [nn.Dense(self.d_models[i+1]) for i in range(self.n_layers)]
        self.layer_norm = nn.LayerNorm(epsilon=self.eps)
        self.dropout_function = nn.Dropout(rate=self.dropout)

    def __call__(self, x, training: bool):
        residual = x
        x = self.layer_norm(x)
        for i in range(self.n_layers):
            x = self.dense_layers[i](x)
            if i != self.n_layers - 1:  # No activation after the last layer
                x = gelu(x)
        x = x + residual
        x = self.dropout_function(x, deterministic=not training)
        return x
