import sys
import os
import jax
from data.numpy_data_loader import NumpyDataLoader
from flax_gate_loop.language_models.gate_loop_lm import *
from flax_gate_loop.language_models.transformer_lm import *
from setups.get_setup_dict import get_setup_dict
from util import get_home_directory
import ast
from datetime import datetime
import wandb

def get_class_from_name(class_name):
    return globals()[class_name]


def main(args):

    datetime_str = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

    print(f"Running experiment {args.dataset_class_name} {args.model_class_name} {args.model_variation_name} {datetime_str} in project {args.project_name}\n")

    if args.save_path is not None:
        args.save_path = os.path.join(get_home_directory(), args.save_path, args.dataset_class_name, f"{args.model_class_name}_{args.model_variation_name}", datetime_str)

    if args.start_from_checkpoint_path is not None:
        args.start_from_checkpoint_path = os.path.join(get_home_directory(), args.start_from_checkpoint_path)

    if args.wandb_api_key is not None:
        os.environ["WANDB_API_KEY"] = args.wandb_api_key
    else:
        wandb.login()

    setup_dict = get_setup_dict(
        args.dataset_class_name,
        args.model_class_name,
        args.model_variation_name,
        args.seed,
        args.num_workers,
        args.datasets_path,
        args.fresh_preprocess
    )

    model_trainer = setup_dict["model_trainer_class"](
        model_class=get_class_from_name(args.model_class_name),
        model_hparams=setup_dict["model_hparams"],
        optimizer_hparams=setup_dict["optimizer_hparams"],
        logger_params=dict(
            run_name=f"{args.dataset_class_name}_{args.model_class_name}_{args.model_variation_name}",
            project_name=args.project_name
        ),
        **setup_dict["model_trainer_hparams"],
        save_path=args.save_path,
        start_from_checkpoint_path=args.start_from_checkpoint_path,
        fixed_checkpoint_steps=args.fixed_checkpoint_steps,
    )

    model_trainer.train_model()


import argparse

def parse_bool(value):
    return str(value).lower() == 'true'

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run experiment")
    parser.add_argument("project_name", help="Name of the project")
    parser.add_argument("dataset_class_name", help="Name of the dataset class")
    parser.add_argument("model_class_name", help="Name of the model class")
    parser.add_argument("--model_variation_name", default="", help="Name of the model variation (optional)")
    parser.add_argument("--datasets_path", help="Path to the datasets (optional)")
    parser.add_argument("--num_workers", default=4, type=int, help="Number of workers (optional)")
    parser.add_argument("--fresh_preprocess", type=parse_bool, default=None, help="Fresh preprocess flag (optional, 'True' or 'False')")
    parser.add_argument("--save_path", help="Path to save (optional)")
    parser.add_argument("--start_from_checkpoint_path", help="Checkpoint path to start training from")
    parser.add_argument('--fixed_checkpoint_steps', type=str, help="List of (fixed) steps to save at (optional)")
    parser.add_argument("--seed", default=42, type=int, help="Seed (optional, default=42)")
    parser.add_argument("--wandb_api_key", default=None)

    args = parser.parse_args()

    # Simplify the handling of "None" string arguments
    for arg in ['datasets_path', 'save_path', 'start_from_checkpoint_path', 'fixed_checkpoint_steps', 'wandb_api_key']:
        if getattr(args, arg) == "None":
            setattr(args, arg, None)

    # Convert fixed_checkpoint_steps to a list of integers if not None
    if args.fixed_checkpoint_steps is not None:
        args.fixed_checkpoint_steps = args.fixed_checkpoint_steps.replace("_", " ")
        try:
            args.fixed_checkpoint_steps = ast.literal_eval(args.fixed_checkpoint_steps)
        except ValueError:
            print(f"Invalid fixed_checkpoint_steps '{args.fixed_checkpoint_steps}'")
            exit(1)

    main(args)




