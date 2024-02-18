from data.speech import ConditionedSpeechDataset, preprocess_speech
from data.numpy_data_loader import NumpyDataLoader
import torch
from torch.utils.data import random_split
import importlib
from training.text_2_speech_model_trainer import Text2SpeechModelTrainer
import os
import numpy as np

def get_setup_dict(model_class_name, model_variation_name, seed, num_workers, datasets_path, fresh_preprocess):

    model_hparams = get_model_setup_dict(model_class_name, model_variation_name)

    batch_size = 16
    val_fraction = 0.05

    data_folder_path = os.path.join(datasets_path, "speech")

    if fresh_preprocess:
        raise AttributeError("This dataset needs to be preprocessed manually!")

    dataset = ConditionedSpeechDataset(data_folder_path)
    val_size = int(len(dataset) * val_fraction)
    train_size = len(dataset) - val_size
    torch.manual_seed(seed)
    train_set, val_set = random_split(dataset, [train_size, val_size])


    train_loader = NumpyDataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers,
                               generator=torch.Generator().manual_seed(seed))
    val_loader = NumpyDataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=num_workers,
                             generator=torch.Generator().manual_seed(seed))

    num_epochs = 50


    speech_targets, speech_tokens, text_tokens = next(iter(train_loader))

    model_trainer_hparams = dict(
        exmp_input_args=(speech_tokens,),
        exmp_input_kwargs=dict(text_tokens=text_tokens),
        val_every_n_steps=50,
        log_every_n_steps=50,
        num_epochs=num_epochs,
        train_loader=train_loader,
        val_loader=val_loader,
        batch_size=batch_size,
        seed=seed,
        debug=False,
    )


    optimizer_hparams = dict(
        lr=0.001,
        warumup_steps=(0.1 * len(train_set) * num_epochs) / batch_size,
        weight_decay=0.05,
        b1=0.9,
        b2=0.98,
    )

    return dict(
        model_trainer_class=Text2SpeechModelTrainer,
        model_hparams=model_hparams,
        optimizer_hparams=optimizer_hparams,
        model_trainer_hparams=model_trainer_hparams,
    )


def get_model_setup_dict(model_class_name, model_variation_name):

    general_model_hparams = dict(
        encoder_n_layer=3,
        decoder_n_layer=3,
        d_model=384,
        d_channel_mixing=384 * 4,
        eps=1e-5,
        channel_mixing_dropout=0.1,
        time_mixing_dropout=0.1,
        encoder_vocab_size=72,
        decoder_vocab_size=1024,
        encoder_max_seq_length=100,
        decoder_max_seq_length=2000,
        encoder_embedding_dropout=0.1,
        decoder_embedding_dropout=0.1,
        n_head=6,
    )

    module_name = f"setups.ConditionedSpeechDataset.{model_class_name}"
    module = importlib.import_module(module_name)
    get_model_hparams = getattr(module, "get_model_hparams")

    specific_model_hparams = get_model_hparams(model_variation_name)

    model_hparams = dict(
        **general_model_hparams,
        **specific_model_hparams
    )

    return model_hparams




