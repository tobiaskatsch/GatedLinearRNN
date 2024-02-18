from flax import linen as nn
import jax.numpy as jnp

from . import GateLoopLM
from .attention import MultiHeadCrossAttention
from .base_models.channel_mixing import ChannelMixing
from .base_models.sequence_model import SinusoidalPositionalEncoding
from .gate_loop import GateLoop
from .base_models import CausalTimeMixing
from typing import Optional, Callable


class CrossAttentionDecoder(nn.Module):
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
    use_head: bool
    n_head: int
    d_h: int

    def setup(self):
        self.channel_mixing_layers = [ChannelMixing(
            d_models=[self.d_model, self.d_channel_mixing, self.d_model],
            dropout=self.channel_mixing_dropout,
            eps=self.eps
        ) for _ in range(self.n_layer)]
        self.cross_attention_layers = [MultiHeadCrossAttention(
            d_model=self.d_model,
            d_h=self.d_h,
            n_head=self.n_head
        ) for _ in range(self.n_layer)]
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
        if self.use_head is True:
            self.head = nn.Dense(self.output_vocab_size)
        self.embedding_dropout_function = nn.Dropout(rate=self.embedding_dropout)
        if self.use_word_embedding:
            self.input_function = nn.Embed(self.input_vocab_size, self.d_model)
        else:
            self.input_function = nn.Dense(self.d_model)

    def __call__(self, x, encoding, training: bool, carry=None, encoding_mask=None):
        seq_length = x.shape[1]
        x = self.input_function(x)
        if self.positional_encoding_mode == 'sinusoidal' or self.positional_encoding_mode == 'learned':
            x = x + self.wpe(seq_length)
        x = self.embedding_dropout_function(x, deterministic=not training)
        h = []
        for l, (cross_attention, time_mixing, channel_mixing) in enumerate(zip(self.cross_attention_layers, self.time_mixing_layers, self.channel_mixing_layers)):
            h_l, x = time_mixing(x, training, carry=(carry[:, l, :] if carry is not None else None))
            x = cross_attention(x, encoding, encoding_mask=encoding_mask)
            x = channel_mixing(x, training)
            h.append(h_l)
        h = jnp.stack(h, axis=1)
        if self.use_head is True:
            x = self.head(x)
        return h, x

class GateLoopCrossAttentionDecoder(CrossAttentionDecoder):
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
    use_head: bool
    n_head: int
    d_h: int

    input_activation: Optional[Callable] = nn.tanh
    hidden_activation: Optional[Callable] = nn.tanh
    gate_activation: Optional[Callable] = nn.sigmoid
    use_true_recurrence: Optional[bool] = False
    use_tied_gates: Optional[bool] = True

    def setup(self):
        super().setup()
        time_mixing_layers = []
        for layer in range(self.n_layer):
            time_mixing_layers.append(
                CausalTimeMixing(
                    eps=self.eps,
                    dropout=self.time_mixing_dropout,
                    model=GateLoop(
                        d_model=self.d_model,
                        d_h=self.d_h,
                        input_activation=self.input_activation,
                        hidden_activation=self.hidden_activation,
                        gate_activation=self.gate_activation,
                        use_true_recurrence=self.use_true_recurrence,
                        use_tied_gates=self.use_tied_gates,
                    )
                )
            )
        self.time_mixing_layers = time_mixing_layers


class GateLoopText2SpeechModel(nn.Module):
    encoder_n_layer: int
    decoder_n_layer: int
    d_model: int
    d_channel_mixing: int
    eps: float
    channel_mixing_dropout: float
    time_mixing_dropout: float
    encoder_vocab_size: int
    decoder_vocab_size: int
    encoder_max_seq_length: int
    decoder_max_seq_length: int
    encoder_embedding_dropout: float
    decoder_embedding_dropout: float
    n_head: int

    d_h: int
    input_activation: Optional[Callable] = nn.tanh
    hidden_activation: Optional[Callable] = nn.tanh
    gate_activation: Optional[Callable] = nn.sigmoid
    use_true_recurrence: Optional[bool] = False
    use_tied_gates: Optional[bool] = True

    def setup(self):
        general_model_params = dict(
            d_model=self.d_model,
            d_channel_mixing=self.d_channel_mixing,
            eps=self.eps,
            channel_mixing_dropout=self.channel_mixing_dropout,
            time_mixing_dropout=self.time_mixing_dropout,
            use_word_embedding=True,
            positional_encoding_mode="none",
            d_h=self.d_h,
            input_activation=self.input_activation,
            hidden_activation=self.hidden_activation,
            gate_activation=self.gate_activation,
            use_true_recurrence=self.use_true_recurrence,
            use_tied_gates=self.use_tied_gates,
        )

        self.encoder = GateLoopLM(
            n_layer=self.encoder_n_layer,
            input_vocab_size=self.encoder_vocab_size,
            output_vocab_size=self.encoder_vocab_size,
            max_seq_length=self.encoder_max_seq_length,
            embedding_dropout=self.encoder_embedding_dropout,
            use_head=False,
            **general_model_params
        )

        self.decoder = GateLoopCrossAttentionDecoder(
            n_layer=self.decoder_n_layer,
            input_vocab_size=self.decoder_vocab_size,
            output_vocab_size=self.decoder_vocab_size,
            max_seq_length=self.decoder_max_seq_length,
            embedding_dropout=self.decoder_embedding_dropout,
            n_head=self.n_head,
            use_head=True,
            **general_model_params
        )

    def __call__(self, speech_tokens, training, text_tokens=None, text_masks=None, encoding=None, carry=None):
        if encoding is None:
            if text_tokens is None:
                raise AttributeError("Either text_tokens or encoding must be supplied!")
            _, encoding = self.encoder(text_tokens, training)
        h, x = self.decoder(speech_tokens, encoding, training, carry=carry, encoding_mask=text_masks)
        return encoding, h, x


