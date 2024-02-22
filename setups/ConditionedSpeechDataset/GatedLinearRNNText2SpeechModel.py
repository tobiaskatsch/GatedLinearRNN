import jax.numpy as jnp

def get_model_hparams(model_variation_name):
    if model_variation_name == "default":
        return dict(
            d_h=256,
            use_true_recurrence=True,
            use_tied_gates=True,
        )
    else:
        raise NotImplementedError
















