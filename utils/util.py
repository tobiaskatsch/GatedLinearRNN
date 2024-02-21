import os
from flax.training import train_state
from typing import Any, Sequence, Optional, Tuple, Iterator, Dict, Callable, List
from gated_linear_rnn.gated_linear_rnn import *


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

def run_model_init(model, init_rng, exmp_input_args, exmp_input_kwargs):
    return model.init(init_rng, *exmp_input_args, training=False, **exmp_input_kwargs)

