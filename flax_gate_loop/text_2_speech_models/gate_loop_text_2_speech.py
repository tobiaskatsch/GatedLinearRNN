from .text_2_speech_model import Text2SpeechModel
from .. import GateLoop
from ..base_models import CausalTimeMixing
from typing import Optional, Callable
from flax import linen as nn


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
    embedding_dropout: float
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




