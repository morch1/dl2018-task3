import re
import random


class CorpusPreprocessor:
    MASK = '?'
    non_letters_regex = re.compile('[^a-ząćęłńóśźż ]')
    multi_spaces_regex = re.compile(' +')

    def __init__(self, dictionary):
        self.dictionary = dictionary

    def transform_text(self, text):
        return self.multi_spaces_regex.sub(' ', self.non_letters_regex.sub('', text.lower())).strip().split()

    def mask_text(self, text):
        i = random.randint(0, len(text))
        original_word = text[i]
        text[i] = self.MASK
        if bool(random.randint(0, 1)):
            return text, original_word, 1
        else:
            return text, random.choice(self.dictionary), 0
