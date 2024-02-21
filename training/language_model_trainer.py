from training.base_trainer import BaseTrainer
import jax.numpy as jnp
import jax
import optax

class LanguageModelTrainer(BaseTrainer):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def create_functions(self):

        def cross_entropy_loss(logits, targets):
            return jnp.mean(optax.softmax_cross_entropy_with_integer_labels(logits, targets))

        def cross_entropy_batch_loss(params, step_rng, batch, training: bool):
            targets, inputs = batch
            hidden_states, logits = self.model.apply(
                {'params': params}, inputs, training, rngs={'dropout': step_rng},
            )
            logits = jnp.reshape(logits, (-1, logits.shape[-1]))    # (batch_size * seq_length, vocab_size)
            targets = jnp.reshape(targets, (-1))                    # (batch_size * sequence_length)
            loss = cross_entropy_loss(logits, targets)
            return loss

        def cross_entropy_batch_loss_and_acc(params, batch, training: bool):
            targets, inputs = batch
            hidden_states, logits = self.model.apply({'params': params}, inputs, training)
            logits = jnp.reshape(logits, (-1, logits.shape[-1]))    # (batch_size * seq_length, vocab_size)
            targets = jnp.reshape(targets, (-1))                    # (batch_size * sequence_length)
            loss = cross_entropy_loss(logits, targets)
            correct = jnp.sum(jnp.argmax(logits, axis=-1) == targets)
            total = targets.size
            acc = correct / total
            return loss, acc

        def train_step(state, batch):
            step_rng = jax.random.fold_in(key=state.rng, data=state.step)
            loss_fn = lambda params: cross_entropy_batch_loss(params, step_rng, batch, training=True)
            loss, grads = jax.value_and_grad(loss_fn)(state.params)
            # Update state
            state = state.apply_gradients(grads=grads)
            metrics = {'loss': loss, 'perplexity': jnp.exp(loss)}
            return state, metrics

        def eval_step(state, batch):
            loss, acc = cross_entropy_batch_loss_and_acc(state.params, batch, training=False)
            metrics = {'loss': loss, 'perplexity': jnp.exp(loss), 'accuracy': acc}
            return metrics

        return train_step, eval_step

