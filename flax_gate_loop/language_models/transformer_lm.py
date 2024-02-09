from typing import Optional, Callable
from flax_gate_loop.base_models.language_model import LanguageModel
from flax_gate_loop.base_models.time_mixing import CausalTimeMixing
from flax_gate_loop.transformer import MultiHeadAttention


class TransformerLM(LanguageModel):
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
    n_head: int
    use_causal_mask: Optional[bool] = True


    def setup(self):
        super().setup()
        self.time_mixing_layers = [CausalTimeMixing(
            eps=self.eps,
            dropout=self.time_mixing_dropout,
            model=MultiHeadAttention(
                d_h=self.d_h,
                n_head=self.n_head,
                use_causal_mask=self.use_causal_mask,
            )
        ) for _ in range(self.n_layer)]