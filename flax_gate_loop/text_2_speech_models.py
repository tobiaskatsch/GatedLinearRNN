from flax import linen as nn
import jax.numpy as jnp
from .base_models.channel_mixing import ChannelMixing
from .base_models.sequence_model import SinusoidalPositionalEncoding
from .gate_loop import GateLoop
from .base_models import CausalTimeMixing
from typing import Optional, Callable

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
    speech_embedding_dropout: float
    text_embedding_dropout: float
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

        #self.text_embedding_dropout_function = nn.Dropout(rate=self.text_embedding_dropout)
        self.speech_embedding_dropout_function = nn.Dropout(rate=self.speech_embedding_dropout)

        #self.text_embedding = nn.Embed(self.text_vocab_size, self.text_embedding_size)
        self.speech_embedding = nn.Embed(self.speech_vocab_size, self.speech_embedding_size)
        self.input_proj = nn.Dense(self.d_model)

        #self.text_head = nn.Dense(self.text_vocab_size)
        self.speech_head = nn.Dense(self.speech_vocab_size)

    def __call__(self, x, training: bool, carry=None):
        b, _, seq_length = x.shape  # (b, 2, seq_length)
        #x_text = self.text_embedding(x[:, 0, :])
        #x_text = self.text_embedding_dropout_function(x_text, deterministic=not training)
        x_speech = self.speech_embedding(x[:, 1, :])
        x_speech = self.speech_embedding_dropout_function(x_speech, deterministic=not training)
        #x = jnp.concatenate((x_text, x_speech), axis=2)
        x = self.input_proj(x_speech)

        if self.positional_encoding_mode == 'sinusoidal' or self.positional_encoding_mode == 'learned':
            x = x + self.wpe(seq_length)
        h = []
        for l, (time_mixing, channel_mixing) in enumerate(zip(self.time_mixing_layers, self.channel_mixing_layers)):
            h_l, x = time_mixing(x, training, carry=(carry[:, l, :] if carry is not None else None))
            x = channel_mixing(x, training)
            h.append(h_l)
        h = jnp.stack(h, axis=1)

        #text_logits = self.text_head(x)
        speech_logits = self.speech_head(x)
        return h, speech_logits  # h, text_logits, speech_logits



class GateLoopText2SpeechModel(Text2SpeechModel):
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
    speech_embedding_dropout: float
    text_embedding_dropout: float
    positional_encoding_mode: str

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



