import math
from flax import linen as nn
import jax.numpy as jnp


def scaled_dot_product(q, k, v, mask=None):
    d_k = q.shape[-1]
    attn_logits = jnp.matmul(q, jnp.swapaxes(k, -2, -1))
    attn_logits = attn_logits / math.sqrt(d_k)
    if mask is not None:
        attn_logits = jnp.where(mask == 0, float('-inf'), attn_logits)
    attention_weights = nn.softmax(attn_logits, axis=-1)
    output = jnp.matmul(attention_weights, v)
    return output
    
class MultiHeadAttention(nn.Module):
    d_h: int
    n_head: int
    use_causal_mask: bool

    def setup(self):
        self.qkv_proj = nn.Dense(3*self.d_h,
            kernel_init=nn.initializers.xavier_uniform(),
            bias_init=nn.initializers.zeros
        )

    def __call__(self, x, *args, **kwargs):
        batch_size, seq_len, d_model = x.shape

        qkv = self.qkv_proj(x)
        qkv = qkv.reshape(batch_size, seq_len, self.n_head, -1)
        qkv = qkv.transpose(0, 2, 1, 3)  # [Batch, Head, SeqLen, Dims]
        q, k, v = jnp.array_split(qkv, 3, axis=-1)

        if self.use_causal_mask is True:
            mask = jnp.tril(jnp.ones((seq_len, seq_len), dtype=bool)).reshape((1, 1, seq_len, seq_len))
        else:
            mask = None

        output = scaled_dot_product(q, k, v, mask=mask)
        output = output.transpose(0, 2, 1, 3) # [Batch, SeqLen, Head, Dims]
        output = output.reshape(batch_size, seq_len, self.d_h)
        return output

