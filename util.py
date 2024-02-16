import os
from flax.training import train_state
from typing import Any, Sequence, Optional, Tuple, Iterator, Dict, Callable, List
from flax_gate_loop.gate_loop import *


def get_home_directory():
    try:
        import google.colab
        return ''
    except ImportError:
        return os.path.expanduser("~")


class TrainState(train_state.TrainState):
    rng: Any = None

def is_unpackable(var):
    try:
        _ = (*var,)
        return True
    except TypeError:
        return False

def get_class_from_name(class_name):
    return globals()[class_name]

def run_model_init(model, init_rng, exmp_input_args, exmp_input_kwargs):
    return model.init(init_rng, *exmp_input_args, training=False, **exmp_input_kwargs)

