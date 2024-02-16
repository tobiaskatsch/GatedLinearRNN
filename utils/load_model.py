from flax_gate_loop.encoder.gate_loop_encoder import *
from flax_gate_loop.encoder.transformer_encoder import *
from flax_gate_loop.encoder_decoder.gate_loop_encoder_decoder import *
from setups.get_setup_dict import get_model_setup_dict
from util import get_home_directory
from jax import random
import orbax

def load_model(dataset_class_name, model_class_name, model_variation_name, exmp_input_args, exmp_input_kwargs, checkpoint_path):
    model_hparams = get_model_setup_dict(dataset_class_name, model_class_name, model_variation_name)
    model_class = GateLoopEncoder
    model = model_class(**model_hparams)
    init_rng = random.PRNGKey(0)
    _ = model.init(init_rng, *exmp_input_args, training=False, **exmp_input_kwargs)
    checkpointer = orbax.checkpoint.PyTreeCheckpointer()
    raw_restored = checkpointer.restore(checkpoint_path)
    params = raw_restored["state"]["params"]
    return model, params
