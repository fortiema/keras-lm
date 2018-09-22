from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import logging
from pathlib import Path
import re

import numpy as np
import spacy
from spacy.attrs import ORTH


logger = logging.getLogger(__name__)


class TextDataSource:
    """Generic source of text data (iterator)

    Raises:
        IOError: [description]
        IOError: [description]

    Returns:
        [type]: [description]
    """

    def __init__(self, source, name_pattern=r".*\.txt"):
        self.source = Path(source)
        self.files = None

        if not self.source.exists():
            raise IOError(f"Ressource {str(self.source)} does not exist!")

        if self.source.is_dir():
            self.files = [f for f in self.source.iterdir() if re.match(name_pattern, str(f))]
        elif self.source.is_file() and re.match(name_pattern, str(self.source.resolve())):
            self.files = [self.source.resolve()]
        else:
            raise IOError(
                f"Ressource {str(self.source)} must contain at least one file matching pattern {name_pattern}."
            )

        logger.info(f"Source {str(self.source)} initialized with {len(self.files)} files.")

    def __iter__(self):
        for f in self.files:
            with open(f, "r") as fin:
                for line in fin:
                    if line:
                        yield line

    def __len__(self):
        return len(list(self.__iter__()))


class Dictionary:

    def __init__(self, source=None):
        self.freq = Counter()
        self.bos, self.eos, self.pad, self.unk = "<bos>", "<eos>", "<pad>", "<unk>"
        self.itos = []
        self.stoi = {}

        if source is not None:
            self.fit(source)

    def fit(self, source):
        logger.info(f"Fitting vocabulary...")
        self.freq = Counter([t for t in source])
        self.itos = [o for o, c in self.freq.most_common()]
        self.itos.insert(0, self.bos)
        self.itos.insert(1, self.eos)
        self.itos.insert(2, self.pad)
        self.itos.insert(3, self.unk)
        self.stoi = {v: i for i, v in enumerate(self.itos)}

    def prune(self, max_vocab, min_freq):
        logger.info(f"Pruning vocabulary to keep at most {max_vocab} tokens...")
        self.itos = [o for o, c in self.freq.most_common(max_vocab) if c > min_freq]
        self.itos.insert(0, self.bos)
        self.itos.insert(1, self.eos)
        self.itos.insert(2, self.pad)
        self.itos.insert(3, self.unk)
        self.stoi = {v: i for i, v in enumerate(self.itos)}
        logger.info("Pruning completed!")

    def __len__(self):
        return len(self.itos)


class LanguageModelLoader:
    """ Returns a language model iterator that iterates through batches that are of length N(bptt,5)

    The first batch returned is always bptt+25; the max possible width.
    """

    def __init__(self, nums, bs, bptt, backwards=False):
        self.bs, self.bptt, self.backwards = bs, bptt, backwards
        self.data = self.batchify(nums)
        self.i, self.iter = 0, 0
        self.n = len(self.data)

    def __iter__(self):
        self.i, self.iter = 0, 0
        while self.i < self.n - 1 and self.iter < len(self):
            if self.i == 0:
                seq_len = self.bptt + 5 * 5
            else:
                bptt = self.bptt if np.random.random() < 0.95 else self.bptt / 2.
                seq_len = max(5, int(np.random.normal(bptt, 5)))
            res = self.get_batch(self.i, seq_len)
            self.i += seq_len
            self.iter += 1
            yield res

    def __len__(self):
        return self.n // self.bptt - 1

    def batchify(self, data):
        nb = data.shape[0] // self.bs
        data = np.array(data[: nb * self.bs])
        data = data.reshape(self.bs, -1).T
        if self.backwards:
            data = data[::-1]
        return data

    def get_batch(self, i, seq_len):
        source = self.data
        seq_len = min(seq_len, len(source) - 1 - i)
        return source[i : i + seq_len], source[i + 1 : i + 1 + seq_len].view(-1)


class Tokenizer:
    def __init__(self, lang="en"):
        self.re_br = re.compile(r"<\s*br\s*/?>", re.IGNORECASE)
        self.tok = spacy.load(lang)
        for w in ("<eos>", "<bos>", "<unk>"):
            self.tok.tokenizer.add_special_case(w, [{ORTH: w}])

    def sub_br(self, x):
        return self.re_br.sub("\n", x)

    def spacy_tok(self, x):
        return [t.text for t in self.tok.tokenizer(self.sub_br(x))]

    re_rep = re.compile(r"(\S)(\1{3,})")
    re_word_rep = re.compile(r"(\b\w+\W+)(\1{3,})")

    @staticmethod
    def replace_rep(m):
        TK_REP = "tk_rep"
        c, cc = m.groups()
        return f" {TK_REP} {len(cc)+1} {c} "

    @staticmethod
    def replace_wrep(m):
        TK_WREP = "tk_wrep"
        c, cc = m.groups()
        return f" {TK_WREP} {len(cc.split())+1} {c} "

    @staticmethod
    def do_caps(ss):
        TOK_UP, TOK_SENT, TOK_MIX = " t_up ", " t_st ", " t_mx "
        res = []
        prev = "."
        re_word = re.compile("\w")
        re_nonsp = re.compile("\S")
        for s in re.findall(r"\w+|\W+", ss):
            res += [TOK_UP, s.lower()] if (s.isupper() and (len(s) > 2)) else [s.lower()]
        return "".join(res)

    def proc_text(self, s):
        s = self.re_rep.sub(Tokenizer.replace_rep, s)
        s = self.re_word_rep.sub(Tokenizer.replace_wrep, s)
        s = Tokenizer.do_caps(s)
        s = re.sub(r"([/#])", r" \1 ", s)
        s = re.sub(" {2,}", " ", s)
        return self.spacy_tok(s)

    @staticmethod
    def proc_all(ss, lang):
        tok = Tokenizer(lang)
        return [tok.proc_text(s) for s in ss]

    @staticmethod
    def proc_all_mp(ss, lang="en", ncpus=1):
        with ProcessPoolExecutor(ncpus) as e:
            return sum(e.map(Tokenizer.proc_all, ss, [lang] * len(ss)), [])
