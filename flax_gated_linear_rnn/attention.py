import math
from flax import linen as nn
import jax.numpy as jnp

import jax.numpy as jnp
from jax.nn import softmax
from flax import linen as nn
import math


def scaled_dot_product(q, k, v, mask=None):
    d_k = q.shape[-1]
    attn_logits = jnp.matmul(q, jnp.swapaxes(k, -2, -1))
    attn_logits = attn_logits / math.sqrt(d_k)
    if mask is not None:
        attn_logits = jnp.where(mask == 0, float('-inf'), attn_logits)
    attention_weights = softmax(attn_logits, axis=-1)
    output = jnp.matmul(attention_weights, v)
    return output


class MultiHeadSelfAttention(nn.Module):
    d_model: int
    d_h: int
    n_head: int
    use_causal_mask: bool

    def setup(self):
        self.qkv_proj = nn.Dense(
            3 * self.d_h,
            kernel_init=nn.initializers.xavier_uniform(),
            bias_init=nn.initializers.zeros
        )
        self.out_proj = nn.Dense(self.d_model)

    def __call__(self, x, mask=None):
        batch_size, seq_len, d_model = x.shape

        qkv = self.qkv_proj(x)
        qkv = qkv.reshape(batch_size, seq_len, self.n_head, 3 * d_model // self.n_head)
        qkv = qkv.transpose(0, 2, 1, 3)  # [Batch, Head, SeqLen, Dims]
        q, k, v = jnp.split(qkv, 3, axis=-1)

        combined_mask = None
        if self.use_causal_mask:
            causal_mask = jnp.tril(jnp.ones((seq_len, seq_len), dtype=jnp.float32)).reshape((1, 1, seq_len, seq_len))
            combined_mask = causal_mask
        if mask is not None:
            mask = mask.reshape((1, 1, seq_len, seq_len))  # Adjust shape to be broadcastable
            combined_mask = mask if combined_mask is None else jnp.logical_and(combined_mask, mask)

        output = scaled_dot_product(q, k, v, mask=combined_mask)
        output = output.transpose(0, 2, 1, 3)  # [Batch, SeqLen, Head, Dims]
        output = output.reshape(batch_size, seq_len, -1)
        output = self.out_proj(output)
        return output


class MultiHeadCrossAttention(nn.Module):
    d_model: int
    d_h: int  # Dimensionality of the model / output size of each head
    n_head: int  # Number of attention heads

    def setup(self):
        self.q_proj = nn.Dense(self.d_h, kernel_init=nn.initializers.xavier_uniform(), bias_init=nn.initializers.zeros)
        self.kv_proj = nn.Dense(2 * self.d_h, kernel_init=nn.initializers.xavier_uniform(), bias_init=nn.initializers.zeros)
        self.out_proj = nn.Dense(self.d_model)

    def __call__(self, query, key_value, encoding_mask=None):
        batch_size, seq_len_query, _ = query.shape
        _, seq_len_kv, _ = key_value.shape

        # Project queries
        q = self.q_proj(query)
        q = q.reshape(batch_size, seq_len_query, self.n_head, -1)
        q = q.transpose(0, 2, 1, 3)  # [Batch, Head, SeqLenQuery, Dims]

        # Project keys and values
        kv = self.kv_proj(key_value)
        kv = kv.reshape(batch_size, seq_len_kv, self.n_head, -1)
        kv = kv.transpose(0, 2, 1, 3)  # [Batch, Head, SeqLenKV, d_h*2]
        k, v = jnp.split(kv, 2, axis=-1)  # [Batch, Head, SeqLenKV, d_h]

        if encoding_mask is not None:
            v = v * encoding_mask[:, None, :, None]

        output = scaled_dot_product(q, k, v, mask=None)
        output = output.transpose(0, 2, 1, 3)  # Back to [Batch, SeqLenQuery, Head, Dims]
        output = output.reshape(batch_size, seq_len_query, -1)
        output = self.out_proj(output)
        return output





