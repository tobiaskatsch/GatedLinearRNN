from ..encoder.gate_loop_encoder import GateLoopEncoderLM
from flax import linen as nn

class GateLoopEncoderDecoder(nn.Module):
    max_phonetics: int
    n_layer_encoder: int
    n_layer_decoder: int
    d_model: int
    d_channel_mixing: int
    eps: int
    channel_mixing_dropout: float
    time_mixing_dropout: float
    embedding_dropout: float
    d_h: int
    def setup(self):
        self.encoder = GateLoopEncoderLM(
            n_layer=self.n_layer_encoder,
            d_model=self.d_model,
            d_channel_mixing=self.d_channel_mixing,
            eps=self.eps,
            channel_mixing_dropout=self.channel_mixing_dropout,
            time_mixing_dropout=self.time_mixing_dropout,
            input_vocab_size=71,
            output_vocab_size=-1,
            max_seq_length=self.max_phonetics,
            embedding_dropout=self.embedding_dropout,
            use_word_embedding=True,
            positional_encoding_mode='none',
            use_head=False,
            d_h=self.dh,
        )






class GateLoopEncoder(nn.Module):
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