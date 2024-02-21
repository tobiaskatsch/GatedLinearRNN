from run_experiment import get_class_from_name
from setups.get_setup_dict import get_model_setup_dict
from jax import random
import orbax

def load_model(dataset_class_name, model_class_name, model_variation_name, exmp_input_args, checkpoint_path, exmp_input_kwargs={}):
    model_hparams = get_model_setup_dict(dataset_class_name, model_class_name, model_variation_name)
    model_class = get_class_from_name(model_class_name)
    model = model_class(**model_hparams)
    init_rng = random.PRNGKey(0)
    _ = model.init(init_rng, *exmp_input_args, False, **exmp_input_kwargs)
    checkpointer = orbax.checkpoint.PyTreeCheckpointer()
    raw_restored = checkpointer.restore(checkpoint_path)
    params = raw_restored["state"]["params"]
    return model, params
