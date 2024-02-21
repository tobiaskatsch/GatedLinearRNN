from training.base_trainer import BaseTrainer
import jax.numpy as jnp
import jax
import optax

class Text2SpeechModelTrainer(BaseTrainer):

    def __init__(self, *args, **kwargs):
        self.top_k = 1
        super().__init__(*args, **kwargs)

    def create_functions(self):

        def cross_entropy_loss(logits, targets):
            return jnp.mean(optax.softmax_cross_entropy_with_integer_labels(logits, targets))

        def reshape_and_cross_entropy_loss(logits, targets):
            logits = jnp.reshape(logits, (-1, logits.shape[-1]))  # (batch_size * seq_length, vocab_size)
            targets = jnp.reshape(targets, (-1))  # (batch_size * sequence_length)
            return cross_entropy_loss(logits, targets)

        def cross_entropy_batch_loss(params, step_rng, batch, training: bool):
            speech_targets, speech_tokens, text_tokens, text_masks = batch
            _, _, speech_logits = self.model.apply(
                {'params': params}, speech_tokens, training, text_tokens=text_tokens, text_masks=text_masks, rngs={'dropout': step_rng},
            )
            loss = reshape_and_cross_entropy_loss(speech_logits, speech_targets)
            return loss

        def accuracy(logits, targets, top_k=1):
            top_k_predictions = jnp.argsort(logits, axis=-1)[..., -top_k:]
            expanded_targets = jnp.expand_dims(targets, axis=-1)
            correct = jnp.sum(jnp.any(top_k_predictions == expanded_targets, axis=-1))
            total = targets.size
            accuracy = correct / total
            return accuracy

        def cross_entropy_batch_loss_and_acc(params, batch):
            speech_targets, speech_tokens, text_tokens, text_masks = batch
            _, _, speech_logits = self.model.apply(
                {'params': params}, speech_tokens, False, text_tokens=text_tokens, text_masks=text_masks,
            )
            loss = reshape_and_cross_entropy_loss(speech_logits, speech_targets)
            acc = accuracy(speech_logits, speech_targets, self.top_k)
            return loss, acc

        def train_step(state, batch):
            step_rng = jax.random.fold_in(key=state.rng, data=state.step)
            loss_fn = lambda params: cross_entropy_batch_loss(params, step_rng, batch, training=True)
            loss, grads = jax.value_and_grad(loss_fn)(state.params)
            state = state.apply_gradients(grads=grads)
            metrics = {'loss': loss}
            return state, metrics

        def eval_step(state, batch):
            loss, acc = cross_entropy_batch_loss_and_acc(state.params, batch)
            metrics = {'loss': loss, f'acc@{self.top_k}': acc}
            return metrics

        return train_step, eval_step

