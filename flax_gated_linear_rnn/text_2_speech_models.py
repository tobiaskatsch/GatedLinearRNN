from flax import linen as nn
import jax.numpy as jnp
import math
from . import GateLoopLM
from .attention import MultiHeadCrossAttention, scaled_dot_product
from .base_models.channel_mixing import ChannelMixing
from .base_models.sequence_model import SinusoidalPositionalEncoding
from .gated_linear_rnn import GatedLinearRNN
from .base_models import CausalTimeMixing
from typing import Optional, Callable


class GatedLinearRNNText2SpeechModel(nn.Module):
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
    cross_attention_layers_ids: list
    cross_attention_dropout: float
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
            positional_encoding_mode="none",
            d_h=self.d_h,
            input_activation=self.input_activation,
            hidden_activation=self.hidden_activation,
            gate_activation=self.gate_activation,
            use_true_recurrence=self.use_true_recurrence,
            use_tied_gates=self.use_tied_gates,
        )

        self.embedding_fn = nn.Embed(self.encoder_vocab_size, self.d_model)

        self.encoder = GateLoopLM(
            n_layer=self.encoder_n_layer,
            input_vocab_size=self.encoder_vocab_size,
            output_vocab_size=self.encoder_vocab_size,
            max_seq_length=self.encoder_max_seq_length,
            embedding_dropout=self.encoder_embedding_dropout,
            use_word_embedding=False,
            use_head=False,
            **general_model_params
        )

        self.decoder = GateLoopCrossAttentionDecoder(
            n_layer=self.decoder_n_layer,
            input_vocab_size=self.decoder_vocab_size,
            output_vocab_size=self.decoder_vocab_size,
            encoder_max_seq_length=self.encoder_max_seq_length,
            decoder_max_seq_length=self.decoder_max_seq_length,
            embedding_dropout=self.decoder_embedding_dropout,
            cross_attention_layers_ids=self.cross_attention_layers_ids,
            cross_attention_dropout=self.cross_attention_dropout,
            use_word_embedding=True,
            n_head=self.n_head,
            use_head=True,
            **general_model_params
        )

    def __call__(self, speech_tokens, training, text_tokens=None, text_masks=None, encoding=None, carry=None):
        if encoding is None:
            if text_tokens is None:
                raise AttributeError("Either text_tokens or encoding must be supplied!")
            e = self.embedding_fn(text_tokens)
            _, encoding = self.encoder(e, training)
            encoding = (encoding + e) * math.sqrt(0.5)
        h, x = self.decoder(speech_tokens, encoding, training, carry=carry, encoding_mask=text_masks)
        return encoding, h, x



class CrossAttentionDecoder(nn.Module):
    n_layer: int
    d_model: int
    d_channel_mixing: int
    eps: float
    channel_mixing_dropout: float
    time_mixing_dropout: float
    input_vocab_size: int
    output_vocab_size: int
    encoder_max_seq_length: int
    decoder_max_seq_length: int
    embedding_dropout: float
    use_word_embedding: bool
    positional_encoding_mode: str
    use_head: bool
    n_head: int
    d_h: int
    cross_attention_layers_ids: list
    cross_attention_dropout: float

    def setup(self):
        self.channel_mixing = ChannelMixing(
            d_models=[self.d_model, self.d_channel_mixing, self.d_model],
            dropout=self.channel_mixing_dropout,
            eps=self.eps
        )
        self.cross_attention_layers = [PositionalEncodedMultiHeadCrossAttention(
            d_model=self.d_model,
            d_h=self.d_h,
            n_head=self.n_head,
            encoder_max_seq_length=self.encoder_max_seq_length,
            decoder_max_seq_length=self.decoder_max_seq_length,
        ) for _ in self.cross_attention_layers_ids]
        if self.positional_encoding_mode == 'learned':
            self.wpe = nn.Embed(self.max_seq_length, self.d_model)
        elif self.positional_encoding_mode == 'sinusoidal':
            self.wpe = SinusoidalPositionalEncoding(max_seq_length=self.decoder_max_seq_length, emb_dim=self.d_model)
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
        k = 0
        for l, time_mixing in enumerate(self.time_mixing_layers):
            h_l, x = time_mixing(x, training, carry=(carry[:, l, :] if carry is not None else None))
            if l in self.cross_attention_layers_ids:
                x = x + self.cross_attention_layers[k](x, encoding, training, encoding_mask=encoding_mask)
                k += 1
            h.append(h_l)
        x = self.channel_mixing(x, training)
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
    encoder_max_seq_length: int
    decoder_max_seq_length: int
    embedding_dropout: float
    use_word_embedding: bool
    positional_encoding_mode: str
    use_head: bool
    n_head: int
    d_h: int
    cross_attention_layers_ids: list
    cross_attention_dropout: float

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
                    model=GatedLinearRNN(
                        d_model=self.d_model,
                        d_h=self.d_h,
                        input_activation=self.input_activation,
                        hidden_activation=self.hidden_activation,
                        gate_activation=self.gate_activation,
                        use_true_recurrence=self.use_true_recurrence,
                        use_tied_gates=self.use_tied_gates,
                        reversed=False,
                    )
                )
            )
        self.time_mixing_layers = time_mixing_layers



class PositionalEncodedMultiHeadCrossAttention(nn.Module):
    d_model: int
    d_h: int  # Dimensionality of the model / output size of each head
    n_head: int  # Number of attention heads
    encoder_max_seq_length: int
    decoder_max_seq_length: int

    def setup(self):
        self.q_proj = nn.Dense(self.d_h, kernel_init=nn.initializers.xavier_uniform(), bias_init=nn.initializers.zeros)
        self.kv_proj = nn.Dense(2 * self.d_h, kernel_init=nn.initializers.xavier_uniform(), bias_init=nn.initializers.zeros)
        self.k_proj = nn.Dense(self.d_h, kernel_init=nn.initializers.xavier_uniform(), bias_init=nn.initializers.zeros)
        self.out_proj = nn.Dense(self.d_model)
        self.d_head = self.d_h // self.n_head
        self.q_positional_encoding = PositionalEncoding(d_model=self.d_model, max_seq_length=self.decoder_max_seq_length)
        self.k_positional_encoding = PositionalEncoding(d_model=self.d_h, max_seq_length=self.encoder_max_seq_length)

    def __call__(self, query, key_value, training: bool, encoding_mask=None):
        batch_size, seq_len_query, _ = query.shape
        _, seq_len_kv, _ = key_value.shape

        query = query + self.q_positional_encoding(seq_len_query)[None, :, :]

        # Project queries
        q = self.q_proj(query)
        q = q.reshape(batch_size, seq_len_query, self.n_head, -1)
        q = q.transpose(0, 2, 1, 3)  # [Batch, Head, SeqLenQuery, Dims]

        # Project keys and values
        kv = self.kv_proj(key_value)
        k, v = jnp.split(kv, 2, axis=-1)  # b, l, d_h
        k = k + self.k_positional_encoding(seq_len_kv)[None, :, :]
        k = self.k_proj(k)

        k = k.reshape(batch_size, seq_len_kv, self.n_head, -1).transpose(0, 2, 1, 3)
        v = v.reshape(batch_size, seq_len_kv, self.n_head, -1).transpose(0, 2, 1, 3)

        if encoding_mask is not None:
            v = v * encoding_mask[:, None, :, None]

        output = scaled_dot_product(q, k, v, mask=None)
        output = output.transpose(0, 2, 1, 3)  # Back to [Batch, SeqLenQuery, Head, Dims]
        output = output.reshape(batch_size, seq_len_query, -1)

        if encoding_mask is not None:
            output = output / ((jnp.sum(encoding_mask, axis=1)[:, None, None]) + 0.000001)

        output = self.out_proj(output)
        return output


class PositionalEncoding(nn.Module):
    d_model: int
    max_seq_length: int

    def setup(self):
        # Initialize the learnable parameter omega (ω_s in the text)
        self.omega = self.param('omega', nn.initializers.ones, (1,))

    def __call__(self, seq_len):
        # Create a matrix of shape [max_len, d_model] with positional indices
        position = jnp.arange(self.max_seq_length)[:, jnp.newaxis]
        div_term = jnp.exp(jnp.arange(0, self.d_model, 2) * -(jnp.log(10000.0) / self.d_model))

        # Compute the positional encodings using the sine and cosine functions
        positional_encoding = jnp.zeros((self.max_seq_length, self.d_model))
        positional_encoding = positional_encoding.at[:, 0::2].set(jnp.sin(position * div_term * self.omega))
        positional_encoding = positional_encoding.at[:, 1::2].set(jnp.cos(position * div_term * self.omega))

        # Return the positional encoding for the specified sequence length
        return positional_encoding[:seq_len, :]


