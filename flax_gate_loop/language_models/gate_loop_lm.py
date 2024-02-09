from flax_gate_loop.base_models.language_model import LanguageModel
from flax_gate_loop.base_models.time_mixing import CausalTimeMixing
from typing import Optional, Callable
from flax_gate_loop.gate_loop import GateLoop
from flax import linen as nn

class GateLoopLM(LanguageModel):
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