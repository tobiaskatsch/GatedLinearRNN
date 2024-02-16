from flax import linen as nn

from flax_gate_loop.attention import MultiHeadCrossAttention
from flax_gate_loop.base_models import SequenceModel
from flax_gate_loop.base_models.channel_mixing import ChannelMixing
import jax.numpy as jnp

class CrossAttentionSequenceModel(SequenceModel):
    n_head: int
    n_layer: int
    d_model: int
    d_channel_mixing: int
    eps: float
    cross_attention_dropout: float
    channel_mixing_dropout: float
    time_mixing_dropout: float
    input_vocab_size: int
    output_vocab_size: int
    max_seq_length: int
    embedding_dropout: float
    use_word_embedding: bool
    positional_encoding_mode: str
    use_head: bool

    def setup(self):
        super().setup()
        self.layer_norm = nn.LayerNorm(epsilon=self.eps)
        self.dropout_function = nn.Dropout(rate=self.cross_attention_dropout)
        self.cross_attention_layers = [MultiHeadCrossAttention(
            d_model=self.d_model,
            d_h=self.d_h,
            n_head=self.n_head,
        ) for _ in range(self.n_layer)]

    def __call__(self, x, training: bool, encoding, carry=None, mask=None):
        seq_length = x.shape[1]
        x = self.input_function(x)
        if self.positional_encoding_mode == 'sinusoidal' or self.positional_encoding_mode == 'learned':
            x = x + self.wpe(seq_length)
        x = self.embedding_dropout_function(x, deterministic=not training)
        h = []
        for l, (cross_attention, time_mixing, channel_mixing) in enumerate(zip(self.cross_attention_layers, self.time_mixing_layers, self.channel_mixing_layers)):
            residual = x
            x = self.layer_norm(x)
            x = self.cross_attention(x, encoding)
            x = x + residual
            x = self.dropout_function(x, deterministic=not training)
            h_l, x = time_mixing(x, training, carry=(carry[:, l, :] if carry is not None else None), mask=mask)
            x = channel_mixing(x, training)
            h.append(h_l)
        h = jnp.stack(h, axis=1)
        if self.use_head is True:
            x = self.head(x)
        return h, x