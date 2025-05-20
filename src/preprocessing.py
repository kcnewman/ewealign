import numpy as np
import pandas as pd
import re
import tqdm
import unicodedata


def load_corpus(file_path):
    df = pd.read_csv(file_path, names=["english", "twi"])
    df["english"] = df["english"].str.strip().str.lower()
    df["english"] = df["english"].str.strip().str.lower()
    df = df.dropna()

    print(f"Loaded {len(df)} parallel sentences")
    return df


def clean_en(text):
    text = text.lower()
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r'[^\w\s,.?!\'"-]', "", text)
    return text


def clean_ak(text):
    def strip_accents(text):
        return "".join(
            t
            for t in unicodedata.normalize("NFD", text)
            if unicodedata.category(t) != "Mn" or t in "ɛɔƐƆŋŊ"
        )

    chars = "a-zA-ZɛɔƐƆŋŊ'"
    text = strip_accents(text)
    text = text.lower()
    text = re.sub(rf"[^{chars}\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def prep_dataset(df):
    eng_corpora = [clean_en(corpus) for corpus in df["english"].tolist()]
    ak_corpora = [clean_ak(corpus) for corpus in df["twi"].tolist()]

    return eng_corpora, ak_corpora


def load_glove(file_path, vocab_size=None):
    print(f"Loading GloVe embeddings from {file_path}...")
    embeddings = {}
    with open(file_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(tqdm.tqdm(f)):
            if vocab_size is not None and i >= vocab_size:
                break
            values = line.strip().split()
            word = values[0]
            vector = np.array(values[1:], dtype="float32")
            embeddings[word] = vector

    embeddings_dim = len(next(iter(embeddings.values())))
    print(f"Loaded {len(embeddings)} GloVe embeddings with dimension {embeddings_dim}")
    return embeddings


def load_fasttext(file_path, vocab_size=None):
    print(f"Loading GloVe embeddings from {file_path}...")
    embeddings = {}
    with open(file_path, "r", encoding="utf-8") as f:
        header = f.readline().strip().split()
        vocab, dim = int(header[0]), int(header[1])

        if vocab_size is None:
            vocab_size = vocab
        else:
            vocab_size = min(vocab_size, vocab)

        for i in tqdm.tqdm(range(vocab_size)):
            line = f.readline().strip().split()
            word = line[0]
            vector = np.array(line[1:], dtype="float32")
            embeddings[word] = vector

    print(f"Loaded {len(embeddings)} FastText embeddings with dimension {dim}")
    return embeddings
