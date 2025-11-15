import datasets
from datasets import load_dataset
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification
from torch.optim import AdamW
from transformers import get_scheduler
import torch
from tqdm.auto import tqdm
import evaluate
import random
import argparse
from nltk.corpus import wordnet
from nltk import word_tokenize
from nltk.tokenize.treebank import TreebankWordDetokenizer

random.seed(0)


def example_transform(example):
    example["text"] = example["text"].lower()
    return example

def custom_transform(example):
    # ----- Config -----
    p_transform = 0.3     
    p_typo = 0.8     
    rng = random

    qwerty_neighbors = {
        'a': list("qwsz"), 'b': list("vghn"), 'c': list("xdfv"),
        'd': list("erfcxs"), 'e': list("wsdr"), 'f': list("rtgvcd"),
        'g': list("tyhbvf"), 'h': list("yujnbg"), 'i': list("ujko"),
        'j': list("uikhmn"), 'k': list("ijolm"), 'l': list("kop"),
        'm': list("njk"), 'n': list("bhjm"), 'o': list("iklp"),
        'p': list("ol"), 'q': list("wa"), 'r': list("edft"),
        's': list("wedxza"), 't': list("rfgy"), 'u': list("yhji"),
        'v': list("cfgb"), 'w': list("qase"), 'x': list("zsdc"),
        'y': list("tugh"), 'z': list("asx")
    }

    def same_case(dst, src):
        if src.istitle():
            return dst.capitalize()
        if src.isupper():
            return dst.upper()
        return dst

    def try_synonym(word):
        base = word.lower()
        syns = wordnet.synsets(base)
        cands = []
        for s in syns:
            for lemma in s.lemmas():
                name = lemma.name()
                if "_" in name:   
                    continue
                if not name.isalpha():
                    continue
                if name.lower() == base:
                    continue
                cands.append(name.lower())
        if not cands:
            return word
        choice = rng.choice(cands)
        return same_case(choice, word)

    def try_typo(word):
        idxs = [i for i, ch in enumerate(word) if ch.isalpha()]
        if not idxs:
            return word
        i = rng.choice(idxs)
        ch = word[i]
        low = ch.lower()
        if low not in qwerty_neighbors or not qwerty_neighbors[low]:
            return word
        repl = rng.choice(qwerty_neighbors[low])
        repl = repl.upper() if ch.isupper() else repl
        return word[:i] + repl + word[i+1:]

    tokens = word_tokenize(example["text"])
    new_tokens = []
    for tok in tokens:
        if tok.isalpha() and rng.random() < p_transform:
            if rng.random() < p_typo:
                new_tok = try_typo(tok)
            else:
                new_tok = try_synonym(tok)
            new_tokens.append(new_tok)
        else:
            new_tokens.append(tok)

    detok = TreebankWordDetokenizer()
    example["text"] = detok.detokenize(new_tokens)
    return example

