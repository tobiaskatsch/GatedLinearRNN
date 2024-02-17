from flax import linen as nn
import jax.numpy as jnp
from flax_gate_loop.base_models.channel_mixing import ChannelMixing
from flax_gate_loop.base_models.sequence_model import SinusoidalPositionalEncoding

class Text2SpeechModel(nn.Module):
    n_layer: int
    d_model: int
    d_channel_mixing: int
    eps: float
    channel_mixing_dropout: float
    time_mixing_dropout: float
    speech_embedding_size: int
    text_embedding_size: int
    speech_vocab_size: int
    text_vocab_size: int
    max_seq_length: int
    embedding_dropout: float
    positional_encoding_mode: str

    def setup(self):
        self.channel_mixing_layers = [ChannelMixing(
            d_models=[self.d_model, self.d_channel_mixing, self.d_model],
            dropout=self.channel_mixing_dropout,
            eps=self.eps
        ) for _ in range(self.n_layer)]
        if self.positional_encoding_mode == 'learned':
            self.wpe = nn.Embed(self.max_seq_length, self.d_model)
        elif self.positional_encoding_mode == 'sinusoidal':
            self.wpe = SinusoidalPositionalEncoding(max_seq_length=self.max_seq_length, emb_dim=self.d_model)
        elif self.positional_encoding_mode == "none":
            pass
        else:
            raise NotImplementedError

        self.text_embedding = nn.Embed(self.text_vocab_size, self.text_embedding_size)
        self.speech_embedding = nn.Embed(self.speech_vocab_size, self.speech_embedding_size)
        self.input_proj = nn.Dense(self.d_model)

        self.text_head = nn.Dense(self.text_vocab_size)
        self.speech_head = nn.Dense(self.speech_vocab_size)

    def __call__(self, x, training: bool, carry=None):
        b, _, seq_length = x.shape  # (b, 2, seq_length)
        x_text = self.text_embedding(x[:, 0, :])
        x_speech = self.speech_embedding(x[:, 1, :])
        x = jnp.concatenate((x_text, x_speech), axis=2)
        x = self.input_proj(x)

        if self.positional_encoding_mode == 'sinusoidal' or self.positional_encoding_mode == 'learned':
            x = x + self.wpe(seq_length)
        x = self.embedding_dropout_function(x, deterministic=not training)
        h = []
        for l, (time_mixing, channel_mixing) in enumerate(zip(self.time_mixing_layers, self.channel_mixing_layers)):
            h_l, x = time_mixing(x, training, carry=(carry[:, l, :] if carry is not None else None))
            x = channel_mixing(x, training)
            h.append(h_l)
        h = jnp.stack(h, axis=1)

        text_logits = self.text_head(x)
        speech_logits = self.speech_head(x)
        return h, text_logits, speech_logits

