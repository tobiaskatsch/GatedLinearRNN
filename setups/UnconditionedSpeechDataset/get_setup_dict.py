from data.speech import UnconditionedSpeechDataset, preprocess_speech
from data.numpy_data_loader import NumpyDataLoader
import torch
from torch.utils.data import random_split
import importlib
from training.language_model_trainer import LanguageModelTrainer
import os

def get_setup_dict(model_class_name, model_variation_name, seed, num_workers, datasets_path, fresh_preprocess):

    model_hparams = get_model_setup_dict(model_class_name, model_variation_name)

    batch_size = 16
    val_fraction = 0.05

    data_folder_path = os.path.join(datasets_path, "speech")

    if fresh_preprocess:
        playlist_url = "https://youtube.com/playlist?list=PL6Sm8cBIf-5HXswvAhof-g1iihU3aJdKs&si=oNBfRFPW7FRG3eiY"
        speech_tokenizer_path = "/content/SpeechTokenizer"
        preprocess_speech(data_folder_path, speech_tokenizer_path, playlist_url, conditioned=False, snippet_length=10, num_quantizers=4)

    dataset = UnconditionedSpeechDataset(data_folder_path)
    val_size = int(len(dataset) * val_fraction)
    train_size = len(dataset) - val_size
    torch.manual_seed(seed)
    train_set, val_set = random_split(dataset, [train_size, val_size])

    train_loader = NumpyDataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers,
                               generator=torch.Generator().manual_seed(seed))
    val_loader = NumpyDataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=num_workers,
                             generator=torch.Generator().manual_seed(seed))

    num_epochs = 50

    model_trainer_hparams = dict(
        exmp_input_args=(next(iter(train_loader))[1:],),
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
        warumup_steps=(0.2 * len(train_set) * num_epochs) / batch_size,
        weight_decay=0.05,
        b1=0.9,
        b2=0.98,
    )

    return dict(
        model_trainer_class=LanguageModelTrainer,
        model_hparams=model_hparams,
        optimizer_hparams=optimizer_hparams,
        model_trainer_hparams=model_trainer_hparams,
    )


def get_model_setup_dict(model_class_name, model_variation_name):

    vocab_size = 1024
    max_seq_length = 2000

    general_model_hparams = dict(
        n_layer=6,
        d_model=384,
        d_channel_mixing=384 * 4,
        eps=1e-5,
        channel_mixing_dropout=0.1,
        time_mixing_dropout=0.1,
        input_vocab_size=vocab_size,
        output_vocab_size=vocab_size,
        max_seq_length=max_seq_length,
        embedding_dropout=0.1,
        use_word_embedding=True,
        use_head=True,
    )

    module_name = f"setups.UnconditionedSpeechDataset.{model_class_name}"
    module = importlib.import_module(module_name)
    get_model_hparams = getattr(module, "get_model_hparams")

    specific_model_hparams = get_model_hparams(model_variation_name)

    model_hparams = dict(
        **general_model_hparams,
        **specific_model_hparams
    )

    return model_hparams


