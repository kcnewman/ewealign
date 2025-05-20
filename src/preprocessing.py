import numpy as np
import pandas as pd
from typing import List, Tuple, Dict
import re
import os
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
