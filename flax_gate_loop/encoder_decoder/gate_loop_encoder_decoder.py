from .. import GateLoop
from ..attention import MultiHeadCrossAttention
from ..base_models import CausalTimeMixing
from ..base_models.cross_attention_sequence_model import CrossAttentionSequenceModel
from ..base_models.sequence_model import SinusoidalPositionalEncoding, SequenceModel
from ..encoder.gate_loop_encoder import GateLoopEncoder
from typing import Optional, Callable
from flax import linen as nn
from flax_gate_loop.base_models.channel_mixing import ChannelMixing
import jax.numpy as jnp

class GateLoopEncoderDecoder(nn.Module):
    n_head: int
    n_layer_encoder: int
    n_layer_decoder: int
    d_model: int
    d_channel_mixing: int
    eps: float
    channel_mixing_dropout: float
    time_mixing_dropout: float
    cross_attention_dropout: float
    input_vocab_size_encoder: int
    input_vocab_size_decoder: int
    output_vocab_size: int
    max_seq_length_encoder: int
    max_seq_length_decoder: int
    embedding_dropout: float
    use_word_embedding_encoder: bool
    use_word_embedding_decoder: bool
    positional_encoding_mode: str
    use_head: bool

    d_h: int
    input_activation: Optional[Callable] = nn.tanh
    hidden_activation: Optional[Callable] = nn.tanh
    gate_activation: Optional[Callable] = nn.sigmoid
    use_true_recurrence: Optional[bool] = False
    use_tied_gates: Optional[bool] = True
    def setup(self):
        shared_kwargs = {
            "d_model": self.d_model,
            "d_channel_mixing": self.d_channel_mixing,
            "eps": self.eps,
            "channel_mixing_dropout": self.channel_mixing_dropout,
            "time_mixing_dropout": self.time_mixing_dropout,
            "embedding_dropout": self.embedding_dropout,
            "positional_encoding_mode": self.positional_encoding_mode,
            "d_h": self.d_h,
            "input_activation": self.input_activation,
            "hidden_activation": self.hidden_activation,
            "gate_activation": self.gate_activation,
            "use_true_recurrence": self.use_true_recurrence,
            "use_tied_gates": self.use_tied_gates,
        }

        # Encoder specific initialization
        self.encoder = GateLoopEncoder(
            n_layer=self.n_layer_encoder,
            input_vocab_size=self.input_vocab_size_encoder,
            output_vocab_size=-1,  # Encoder specific parameter
            max_seq_length=self.max_seq_length_encoder,
            use_word_embedding=self.use_word_embedding_encoder,
            use_head=False,  # Encoder specific parameter
            **shared_kwargs
        )

        # Decoder specific initialization
        self.decoder = GateLoopCrossAttentionSequenceModel(
            n_head=self.n_head,
            n_layer=self.n_layer_decoder,
            input_vocab_size=self.input_vocab_size_decoder,
            output_vocab_size=self.output_vocab_size,
            max_seq_length=self.max_seq_length_decoder,
            use_word_embedding=self.use_word_embedding_decoder,
            use_head=self.use_head,
            cross_attention_dropout=self.cross_attention_dropout,
            **shared_kwargs
        )

    def __call__(self, decoder_input, training: bool, encoder_input=None, encoding=None, decoder_carry=None, mask=None):
        if encoding is None:
            if encoder_input is None:
                raise AttributeError("Either encoder_input or precomputed encoding required!")
            encoding = self.encoder(encoder_input, training)
        h, x = self.decoder(decoder_input, training, encoding, carry=decoder_carry)
        return encoding, h, x


class GateLoopCrossAttentionSequenceModel(CrossAttentionSequenceModel):
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

