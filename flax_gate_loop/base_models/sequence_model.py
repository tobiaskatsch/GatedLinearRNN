from flax import linen as nn
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

    def __call__(self, x, training: bool):
        """
        :param      x           float     (batch_size, seq_len, d_model)         required
        :param      training    bool                                            optional
        :return:    y           float     (batch_size, seq_len, d_model)
        """
        for time_mixing, channel_mixing in zip(self.time_mixing_layers, self.channel_mixing_layers):
            x = time_mixing(x, training)
            x = channel_mixing(x, training)
        return x


