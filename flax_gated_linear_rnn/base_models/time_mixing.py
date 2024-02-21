from flax import linen as nn
from typing import Any

class CausalTimeMixing(nn.Module):
    eps: float
    dropout: float
    model: Any

    def setup(self):
        self.layer_norm_function = nn.LayerNorm(epsilon=self.eps)
        self.dropout_function = nn.Dropout(rate=self.dropout)

    def __call__(self, x, training: bool, carry=None, mask=None):
        """
        :param      x           float     (batch_size, seq_len, d_model)         required
        :param      training    bool                                            optional
        :return:    y           float     (batch_size, seq_len, d_model)
        """
        residual = x
        x = self.layer_norm_function(x)
        h, x = self.model(x, carry=carry, mask=mask)
        x = x + residual
        x = self.dropout_function(x, deterministic=not training)
        return h, x
