import logging

from sources import WikidumpDataSource
from text import Dictionary, Tokenizer


logging.basicConfig(level=logging.INFO)


def process_source(source, out):
    data = WikidumpDataSource('', name_pattern=r".*")
    tokenizer = Tokenizer()

    with open('processed.txt', 'w') as fout:
        for doc in data:
            tok = tokenizer.proc_text(doc)
            print(tok)
            fout.write(" ".join(tok))
