import importlib
import os
from utils.util import get_home_directory

def get_setup_dict(dataset_class_name, model_class_name, model_variation_name, seed, num_workers, datasets_path, fresh_preprocess):
    if datasets_path is not None:
        datasets_path = os.path.join(get_home_directory(), datasets_path)  # construct absolute path
    module_name = f"setups.{dataset_class_name}.get_setup_dict"
    module = importlib.import_module(module_name)
    get_dataset_specific_setup_dict = getattr(module, "get_setup_dict")
    return get_dataset_specific_setup_dict(model_class_name, model_variation_name, seed, num_workers, datasets_path, fresh_preprocess)

def get_model_setup_dict(dataset_class_name, model_class_name, model_variation_name):
    module_name = f"setups.{dataset_class_name}.get_setup_dict"
    module = importlib.import_module(module_name)
    get_dataset_specific_model_setup_dict = getattr(module, "get_model_setup_dict")
    return get_dataset_specific_model_setup_dict(model_class_name, model_variation_name)

