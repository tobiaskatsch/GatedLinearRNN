import jax.numpy as jnp

def get_model_hparams(model_variation_name):
    if model_variation_name == "default":
        return dict(
            positional_encoding_mode="sinusoidal",
            n_head=6,
            d_h=384 * 2,
            use_causal_mask=True,
        )
    else:
        raise NotImplementedError












