import jax.numpy as jnp

def get_model_hparams(model_variation_name):
    if model_variation_name == "default":
        return dict(
            positional_encoding_mode="none",
            d_h=384*2,
            use_true_recurrence=True,
            use_tied_gates=True,
        )
    elif model_variation_name == "untied":
        return dict(
            positional_encoding_mode="none",
            d_h=384*2,
            use_true_recurrence=False,
            use_tied_gates=False,
        )
    elif model_variation_name == "true_recurrent":
        return dict(
            positional_encoding_mode="none",
            d_h=384*2,
            use_true_recurrence=True,
            use_tied_gates=True,
        )
    else:
        raise NotImplementedError
















