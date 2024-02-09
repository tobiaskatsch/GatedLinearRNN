from flax import linen as nn
import jax.numpy as jnp
from flax_gate_loop.base_models.sequence_model import SequenceModel

class LanguageModel(SequenceModel):
    n_layer: int
    d_model: int
    d_channel_mixing: int
    eps: float
    channel_mixing_dropout: float
    time_mixing_dropout: float
    input_vocab_size: int
    output_vocab_size: int
    max_seq_length: int
    embedding_dropout: float
    use_word_embedding: bool
    positional_encoding_mode: str

    def setup(self):
        if self.positional_encoding_mode == 'learned':
            self.wpe = nn.Embed(self.max_seq_length, self.d_model)
        elif self.positional_encoding_mode == 'sinusoidal':
            self.wpe = SinusoidalPositionalEncoding(max_seq_length=self.max_seq_length, emb_dim=self.d_model)
        elif self.positional_encoding_mode == "none":
            pass
        else:
            raise NotImplementedError
        if self.input_vocab_size is None and self.use_word_embedding:
            raise AttributeError("self.input_vocab_size is None and self.use_word_embedding")
        self.head = nn.Dense(self.output_vocab_size)
        super().setup()
        self.embedding_dropout_function = nn.Dropout(rate=self.embedding_dropout)
        if self.use_word_embedding:
            self.input_function = nn.Embed(self.input_vocab_size, self.d_model)
        else:
            self.input_function = nn.Dense(self.d_model)


    def __call__(self, x, training: bool):
        """
        :param      x           int       (batch_size, seq_len) or (batch_size, seq_len, input_dim)     required
        :param      training    bool                                                                    optional
        :return:    y           float     (batch_size, seq_len, d_model)
        """
        seq_length = x.shape[1]

        x = self.input_function(x)

        if self.positional_encoding_mode == 'sinusoidal' or self.positional_encoding_mode == 'learned':
            x = x + self.wpe(seq_length)

        x = self.embedding_dropout_function(x, deterministic=not training)
        x = super().__call__(x, training)
        x = self.head(x)
        return x

class SinusoidalPositionalEncoding(nn.Module):
    max_seq_length: int
    emb_dim: int

    def setup(self):
        position = jnp.arange(self.max_seq_length)[:, None]
        div_term = jnp.exp(jnp.arange(0, self.emb_dim, 2) *
                           -(jnp.log(10000.0) / self.emb_dim))
        positional_encoding = jnp.zeros((1, self.max_seq_length, self.emb_dim))
        positional_encoding = positional_encoding.at[:, :, 0::2].set(jnp.sin(position * div_term))
        positional_encoding = positional_encoding.at[:, :, 1::2].set(jnp.cos(position * div_term))
        self.positional_encoding = positional_encoding

    def __call__(self, seq_len):
        return self.positional_encoding[:, :seq_len, :]