import argparse
import logging
import os
from pathlib import Path
import uuid

import numpy as np

from models import AWDLSTMLanguageModel, LanguageModel
from sources import TextDataSource, WikidumpDataSource
from text import Dictionary, LanguageModelLoader, Tokenizer


logging.basicConfig(level=logging.INFO)


def process_source(source, log_dir="logs", out_dir="models", dictionary=None, batch_size=16, max_vocab=30000):
    exp_id = str(uuid.uuid4())

    path_logs = Path(log_dir) / exp_id
    path_logs.mkdir()
    path_model = Path(out_dir) / exp_id
    path_model.mkdir()

    data_train = TextDataSource(os.path.join(source, "train.txt"))
    data_valid = TextDataSource(os.path.join(source, "valid.txt"))
    data_test = TextDataSource(os.path.join(source, "test.txt"))

    if not dictionary:
        dictionary = Dictionary(data_train)
        dictionary.save("dict.pkl")
    else:
        dictionary = Dictionary.load(dictionary)
    dictionary.prune(max_vocab=max_vocab)

    loader_train = LanguageModelLoader(
        np.concatenate(np.array(list(dictionary.numericalize(data_train, np=True)))), batch_size, 70
    )
    loader_valid = LanguageModelLoader(
        np.concatenate(np.array(list(dictionary.numericalize(data_valid, np=True)))), batch_size, 70
    )
    loader_test = LanguageModelLoader(
        np.concatenate(np.array(list(dictionary.numericalize(data_test, np=True)))), batch_size, 70
    )

    model = AWDLSTMLanguageModel(len(dictionary), 200, log_dir=str(path_logs))

    model.fit(loader_train, loader_valid, bs=batch_size)

    model.eval(loader_test)

    model.save(str(path_model))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Keras Language Model")
    parser.add_argument("source", type=str, default=None, help="Path to source directory")
    parser.add_argument("--dictionary", type=str, default=None, help="Persisted dictionnary to load back")
    parser.add_argument("--max-vocab", type=int, default=30000, help="Maximum vocabulary size")
    parser.add_argument("-b", "--batch-size", type=int, default=16, help="Batch Size")
    parser.add_argument("-i", "--epochs", type=int, default=1, help="Maximum number of epochs")
    parser.add_argument(
        "-l", "--log-dir", type=str, default=os.path.dirname(os.path.abspath(__file__)), help="Logging folder"
    )
    parser.add_argument(
        "-o", "--out-dir", type=str, default=os.path.dirname(os.path.abspath(__file__)), help="Model output folder"
    )

    args = parser.parse_args()

    try:
        process_source(args.source, args.log_dir, args.out_dir, args.dictionary, args.batch_size, args.max_vocab)
    except KeyboardInterrupt:
        print("Bye!")
