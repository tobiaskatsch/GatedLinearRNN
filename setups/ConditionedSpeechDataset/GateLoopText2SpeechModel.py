import jax.numpy as jnp

def get_model_hparams(model_variation_name):
    if model_variation_name == "expressive":
        return dict(
            d_h=512*2,
            use_true_recurrence=True,
        )
    elif model_variation_name == "efficient":
        return dict(
            d_h=512*2,
            use_true_recurrence=False,
        )
    else:
        raise NotImplementedError
















