import re
import random
import json

random.seed(420)


class CorpusPreprocessor:
    MASK = '?'
    non_letters_regex = re.compile('[^a-ząćęłńóśźż ]')
    multi_spaces_regex = re.compile(' +')

    def __init__(self, path=None, n_sentences=None, max_word_length=25, max_sentence_length=25):
        if path is None:
            return
        self.sentences = []
        with open(path, encoding='utf-8') as f:
            sentences = f.readlines()
        if n_sentences is None:
            random.shuffle(sentences)
        else:
            sentences = random.sample(sentences, 2 * n_sentences)
        words = set()
        for s in sentences:
            ts = self.transform_text(s)
            if len(ts) <= max_sentence_length and all(len(w) <= max_word_length for w in ts):
                self.sentences.append(ts)
                words.update(ts)
            if len(self.sentences) == n_sentences:
                break
        self.words = list(words)

    def transform_text(self, text):
        return self.multi_spaces_regex.sub(' ', self.non_letters_regex.sub('', text.lower())).strip().split()

    def mask_text(self, text):
        i = random.randint(0, len(text))
        original_word = text[i]
        text[i] = self.MASK
        if bool(random.randint(0, 1)):
            return text, original_word, 1
        else:
            return text, random.choice(self.words), 0

    def save(self, path):
        data = {'words': self.words, 'sentences': self.sentences}
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f)

    def load(self, path):
        with open(path, encoding='utf-8') as f:
            data = json.load(f)
        self.words = data['words']
        self.sentences = data['sentences']


def main():
    c = CorpusPreprocessor("D:\\data\\train_shuf.txt", 50000)
    c.save('corpus.json')


if __name__ == '__main__':
    main()
