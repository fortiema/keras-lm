import argparse
import logging
import os
import sys

import numpy as np

from models import LanguageModel
from sources import TextDataSource, WikidumpDataSource
from text import Dictionary, LanguageModelLoader, Tokenizer


logging.basicConfig(level=logging.INFO)

BATCH_SIZE = 16


def process_source(source, dictionary=None):
    data_train = TextDataSource(os.path.join(source, "train.txt"))
    data_valid = TextDataSource(os.path.join(source, "valid.txt"))
    # data_test = TextDataSource(os.path.join(source, "test.txt"))

    if not dictionary:
        dictionary = Dictionary(data_train)
        dictionary.save("dict.pkl")
    else:
        dictionary = Dictionary.load(dictionary)
    dictionary.prune(max_vocab=50000)

    loader_train = LanguageModelLoader(
        np.concatenate(np.array(list(dictionary.numericalize(data_train, np=True)))), BATCH_SIZE, 70
    )
    loader_valid = LanguageModelLoader(
        np.concatenate(np.array(list(dictionary.numericalize(data_valid, np=True)))), BATCH_SIZE, 70
    )

    model = LanguageModel(len(dictionary), 100)

    model.fit(loader_train, loader_valid, bs=BATCH_SIZE)

    # model.eval(loader_test)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Keras Language Model')
    parser.add_argument('source', type=str, default=None, help='Path to source directory')
    parser.add_argument('--dictionary', type=str, default=None, help='Persisted dictionnary to load back')

    args = parser.parse_args()

    try:
        process_source(args.source, args.dictionary)
    except KeyboardInterrupt:
        print("Bye!")
