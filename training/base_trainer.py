from flax import linen as nn
from jax import random
from copy import copy
import optax
import jax
import time
import jax.numpy as jnp
import orbax.checkpoint
from flax.training import orbax_utils
from tqdm import tqdm
from collections import defaultdict
from typing import Any, Sequence, Optional, Tuple, Iterator, Dict, Callable, List
from pytorch_lightning.loggers import WandbLogger
from util import run_model_init, is_unpackable, TrainState
import os
import time


class BaseTrainer:

    def __init__(self,
                 model_class: nn.Module,
                 model_hparams: Dict[str, Any],
                 optimizer_hparams: Dict[str, Any],
                 logger_params: Dict[str, Any],
                 exmp_input: Any,
                 val_every_n_steps: int,
                 log_every_n_steps: int,
                 num_epochs,
                 train_loader,
                 val_loader=None,
                 test_loader=None,
                 checkpoint_best_every_n_steps=1000,
                 fixed_checkpoint_steps=None,
                 precision=32,
                 batch_size="?",
                 test_every_n_steps=None,
                 train_step_kwargs: Dict[str, Any] = {},
                 eval_step_kwargs: Dict[str, Any] = {},
                 seed: int = 42,
                 debug: bool = False,
                 save_path=None,
                 start_from_checkpoint_path=None,
            ):
        super().__init__()
        self.model_class = model_class
        self.model_hparams = model_hparams
        self.optimizer_hparams = optimizer_hparams
        self.logger_params = logger_params
        self.exmp_input = exmp_input
        self.val_every_n_steps = val_every_n_steps
        self.log_every_n_steps = log_every_n_steps
        self.num_epochs = num_epochs
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.train_step_kwargs = train_step_kwargs
        self.eval_step_kwargs = eval_step_kwargs
        self.seed = seed
        self.test_every_n_steps = test_every_n_steps
        self.batch_size = batch_size
        self.debug = debug
        self.best_val_loss = float('inf')  # Initialize with infinity
        self.checkpoint_best_every_n_steps = checkpoint_best_every_n_steps
        self.steps_since_last_checkpoint = 0
        self.precision = precision
        self.save_path = save_path
        self.fixed_checkpoint_steps = fixed_checkpoint_steps
        self.start_from_checkpoint_path = start_from_checkpoint_path
        self.checkpointer = orbax.checkpoint.PyTreeCheckpointer()

        if self.save_path is not None:
            self.checkpoints_path = os.path.join(save_path, "checkpoints")

        self.config = {
            'model_class_name': model_class.__name__,
            'model_hparams': model_hparams,
            'optimizer_hparams': optimizer_hparams,
            'seed': self.seed,
            'batch_size': self.batch_size
        }
        # Create empty model without any parameters created yet
        self.model = self.model_class(**self.model_hparams)
        self.print_tabulate()
        # Init trainer parts
        self.create_jitted_functions()
        self.init_model()

        self.logger = WandbLogger(
            name=logger_params.get('run_name', None),
            project=logger_params.get('project_name', None),
            config=self.config,
        )

        if self.start_from_checkpoint_path is not None:
            self.restore_checkpoint(self.start_from_checkpoint_path)

    def init_model(self):
        # Prepare PRNG and input
        model_rng = random.PRNGKey(self.seed)
        model_rng, init_rng = random.split(model_rng)
        # Run model initialization
        params = run_model_init(self.model, init_rng, self.exmp_input)['params']
        # Create default state. Optimizer is initialized later

        hparams = copy(self.optimizer_hparams)

        lr = hparams.pop('lr')
        gradient_clip = hparams.pop('gradient_clip', 1.0)
        weight_decay = hparams.pop('weight_decay', 0.05)
        warumup_steps = hparams.pop('warumup_steps', 0)

        self.lr_schedule = optax.warmup_cosine_decay_schedule(
            init_value=0.0,
            peak_value=lr,
            warmup_steps=warumup_steps,
            decay_steps=int(self.num_epochs * len(self.train_loader)),
            end_value=1e-7
        )

        tx = optax.chain(
            optax.clip_by_global_norm(gradient_clip),
            optax.adamw(self.lr_schedule, weight_decay=weight_decay),
        )

        self.state = TrainState.create(
            apply_fn=self.model.apply,
            params=params,
            rng=model_rng,
            tx=tx,
        )

    def print_tabulate(self):
        if is_unpackable(self.exmp_input):
            print(self.model.tabulate(random.PRNGKey(0), *self.exmp_input, training=False))
        else:
            print(self.model.tabulate(random.PRNGKey(0), self.exmp_input, training=False))


    def create_jitted_functions(self):
        train_step, eval_step = self.create_functions()
        if self.debug:  # Skip jitting
            self.train_step = train_step
            self.eval_step = eval_step
        else:
            self.train_step = jax.jit(train_step)
            self.eval_step = jax.jit(eval_step)

    def create_functions(self):
        """
        train_step and eval_step MUST ONLY have JAX interpretable inputs (no strings, dicts,...)
        """

        def train_step(state: TrainState,
                       batch: Any,
                       **train_step_kwargs):
            metrics = {}
            return state, metrics

        def eval_step(state: TrainState,
                      batch: Any,
                      **eval_step_kwargs):
            metrics = {}
            return metrics

        raise NotImplementedError

    def train_model(self):

        if self.start_from_checkpoint_path is not None:
            skip_epochs = self.state.step // len(self.train_loader)
            skip_steps = self.state.step % len(self.train_loader)
            print(f"Resuming from epoch {skip_epochs+1} which is partially complete ({skip_steps+1}/{len(self.train_loader)})")
        else:
            skip_epochs = 0
            skip_steps = 0

        # Prepare training loop
        self.on_training_start()

        try:
            for epoch_idx in tqdm(range(1, self.num_epochs+1), desc='Epochs'):
                if epoch_idx < skip_epochs+1:
                    continue
                self.train_epoch(self.train_loader, self.val_loader, self.test_loader, epoch_idx, skip_steps=skip_steps)  # Assuming train_step handles a single batch
                skip_steps = 0
                self.on_training_epoch_end(epoch_idx)
            self.test_model(self.test_loader)
            self.logger.experiment.finish()
        except KeyboardInterrupt:
            self.test_model(self.test_loader)
            self.logger.experiment.finish()

    def test_model(self, test_loader):
        if test_loader is None:
            return
        if self.save_path is not None:
            for checkpoint_name in os.listdir(self.checkpoints_path):
                checkpoint_path = os.path.join(self.checkpoints_path, checkpoint_name)
                self.restore_checkpoint(checkpoint_path=checkpoint_path)
                test_metrics = self.eval_model(test_loader, log_prefix='test/')
                print(f"Test metrics for checkpoint '{checkpoint_name}': ", test_metrics)

    def train_epoch(self, train_loader: Iterator, val_loader: Iterator, test_loader, epoch_idx: int, skip_steps=0) -> Dict[str, Any]:
        # Train model for one epoch, and log avg loss and accuracy
        #start_time = time.time()
        metrics = defaultdict(float)
        steps_counter = skip_steps
        for batch_idx, batch in enumerate(train_loader):
            if batch_idx < skip_steps:
                continue
            self.state, step_metrics = self.train_step(self.state, batch, **self.train_step_kwargs)
            for key in step_metrics:
                metrics['train/' + key] += step_metrics[key]  # Accumulate the metrics
            steps_counter += 1  # Increment the steps counter
            self.steps_since_last_checkpoint += 1
            if self.state.step % self.log_every_n_steps == 0:
                metrics = {key: value / steps_counter for key, value in metrics.items()}  # Compute the average
                metrics["train/epoch"] = epoch_idx
                metrics["train/lr"] = self.lr_schedule(self.state.step)
                self.logger.log_metrics(metrics, step=self.state.step)
                metrics = defaultdict(float)  # Reset metrics
                steps_counter = 0  # Reset the steps counter
            if self.state.step % self.val_every_n_steps == 0:
                val_metrics = self.eval_model(val_loader, log_prefix='val/')
                self.logger.log_metrics(val_metrics, step=self.state.step)
                self.maybe_store_checkpoint(val_metrics["val/loss"])
            if self.test_every_n_steps is not None and self.state.step % self.test_every_n_steps == 0:
                test_metrics = self.eval_model(test_loader, log_prefix='test/')
                self.logger.log_metrics(test_metrics, step=self.state.step)
            if self.fixed_checkpoint_steps is not None:
                if self.state.step in self.fixed_checkpoint_steps:
                    checkpoint_path = os.path.join(self.checkpoints_path, str(self.state.step))
                    self.store_checkpoint(checkpoint_path=checkpoint_path)

        #end_time = time.time()
        #elapsed_time = (end_time - start_time) / 60  # Elapsed time in minutes
        #metrics["train/epoch_time_minutes"] = elapsed_time
        #self.logger.log_metrics(metrics, step=self.state.step)
        return metrics

    def eval_model(self, data_loader: Iterator, log_prefix: Optional[str] = '') -> Dict[str, Any]:
        # Test model on all images of a data loader and return avg loss
        metrics = defaultdict(float)
        for batch in data_loader:
            step_metrics = self.eval_step(self.state, batch, **self.eval_step_kwargs)
            for key in step_metrics:
                metrics[key] += step_metrics[key]
        metrics = {(log_prefix + key): (metrics[key] / len(data_loader)).item() for key in metrics}
        return metrics

    def maybe_store_checkpoint(self, val_loss):
        if self.save_path is None or self.state.step == 1:
            return
        if val_loss < self.best_val_loss and self.steps_since_last_checkpoint > self.checkpoint_best_every_n_steps:
            self.best_val_loss = val_loss
            self.steps_since_last_checkpoint = 0
            checkpoint_path = os.path.join(self.checkpoints_path, "best")
            self.store_checkpoint(checkpoint_path=checkpoint_path)

    def get_checkpoint(self):
        return dict(
            model_class_name=self.model_class.__name__,
            model_hparams=self.model_hparams,
            exmp_input=self.exmp_input,
            state=self.state,
        )

    def store_checkpoint(self, checkpoint_path):
        if self.save_path is None:
            return
        ckpt = self.get_checkpoint()
        save_args = orbax_utils.save_args_from_target(ckpt)
        self.checkpointer.save(checkpoint_path, ckpt, save_args=save_args, force=True)

    def restore_checkpoint(self, checkpoint_path):
        item = self.get_checkpoint()
        restore_args = orbax_utils.restore_args_from_target(item)
        ckpt = self.checkpointer.restore(checkpoint_path, item=item, restore_args=restore_args, transforms={})
        self.state = ckpt["state"]

    def on_training_start(self):
        pass

    def on_training_epoch_end(self, epoch_idx: int):
        pass
