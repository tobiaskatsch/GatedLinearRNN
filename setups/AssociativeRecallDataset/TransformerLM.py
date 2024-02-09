import jax.numpy as jnp

def get_model_hparams(model_variation_name):
    return dict(
        positional_encoding_mode="sinusoidal",
        d_h=128,
        n_head=4,
        use_causal_mask=True,
    )












