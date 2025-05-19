import re
import unicodedata


class TwiTextCleaner:
    def __init__(self, keep_chars="a-zA-ZɛɔƐƆŋŊ'"):
        self.keep_chars = keep_chars

    def strip_accents_but_keep_twi(self, text):
        return "".join(
            c
            for c in unicodedata.normalize("NFD", text)
            if unicodedata.category(c) != "Mn" or c in "ɛɔƐƆŋŊ"
        )

    def normalize(self, text):
        text = self.strip_accents_but_keep_twi(text)
        text = text.lower()
        text = re.sub(rf"[^{self.keep_chars}\s]", " ", text)
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    def clean_file(self, filepath):
        with open(filepath, "r", encoding="utf-8") as f:
            lines = f.readlines()
        return [self.normalize(line) for line in lines]

    def save(self, output_path, filepath):
        with open(output_path, "w", encoding="utf-8") as f:
            for line in self.clean_file(filepath):
                f.write(line + "\n")
