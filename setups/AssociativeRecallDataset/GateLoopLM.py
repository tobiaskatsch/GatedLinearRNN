import jax.numpy as jnp

def get_model_hparams(model_variation_name):
    if model_variation_name == "expressive":
        return dict(
            positional_encoding_mode="none",
            d_h=128*2,
            use_true_recurrence=True,
        )
    elif model_variation_name == "efficient":
        return dict(
            positional_encoding_mode="none",
            d_h=128*2,
            use_true_recurrence=False,
        )
    else:
        raise NotImplementedError












