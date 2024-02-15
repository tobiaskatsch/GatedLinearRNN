from flax import linen as nn
import jax.numpy as jnp
from flax_gate_loop.base_models.channel_mixing import ChannelMixing

class SequenceModel(nn.Module):
    n_layer: int
    d_model: int
    d_channel_mixing: int
    eps: float
    channel_mixing_dropout: float
    time_mixing_dropout: float

    def setup(self):
        self.channel_mixing_layers = [ChannelMixing(
            d_models=[self.d_model, self.d_channel_mixing, self.d_model],
            dropout=self.channel_mixing_dropout,
            eps=self.eps
        ) for _ in range(self.n_layer)]

    def __call__(self, x, training: bool, carry=None, mask=None):
        """
        :param      x           float     (batch_size, seq_len, d_model)         required
        :param      training    bool                                            optional
        :return:    y           float     (batch_size, seq_len, d_model)
        """
        h = []
        for l, (time_mixing, channel_mixing) in enumerate(zip(self.time_mixing_layers, self.channel_mixing_layers)):
            h_l, x = time_mixing(x, training, carry=(carry[:, l, :] if carry is not None else None), mask=mask)
            x = channel_mixing(x, training)
            h.append(h_l)
        h = jnp.stack(h, axis=1)
        return h, x
