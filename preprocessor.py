import argparse
import re
import random
import torch


class CorpusPreprocessor:
    MASK = '?'
    alphabet = 'aąbcćdeęfghijklłmnńoópqrsśtuvwxyzźż'
    non_letters_regex = re.compile(f'[^{alphabet} ]')
    multi_spaces_regex = re.compile(' +')

    def __init__(self, path=None, n_sentences=None, max_word_length=25, max_sentence_length=25):
        if path is None:
            return
        self.max_word_length = max_word_length
        self.max_sentence_length = max_sentence_length
        self.sentences = []
        with open(path, encoding='utf-8') as f:
            sentences = f.readlines()
        if n_sentences is None or n_sentences > len(sentences) / 2:
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
        self.words = sorted(list(words))

    def transform_text(self, text):
        return self.multi_spaces_regex.sub(' ', self.non_letters_regex.sub('', text.lower())).strip().split()

    def mask_text(self, text):
        i = random.randint(0, len(text) - 1)
        original_word = text[i]
        masked_sent = text.copy()
        masked_sent[i] = self.MASK
        if bool(random.randint(0, 1)):
            return masked_sent, original_word, 1
        else:
            return masked_sent, random.choice(self.words), 0

    def save(self, path):
        torch.save({
            'max_word_length': self.max_word_length,
            'max_sentence_length': self.max_sentence_length,
            'words': self.words,
            'sentences': self.sentences
        }, path)

    def load(self, path):
        data = torch.load(path)
        self.max_word_length = data['max_word_length']
        self.max_sentence_length = data['max_sentence_length']
        self.words = data['words']
        self.sentences = data['sentences']


def main():
    random.seed(420)
    parser = argparse.ArgumentParser(description='Preprocess data',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--source', default='train_shuf.txt', help='file containing original dataset')
    parser.add_argument('--destination', default='corpus.pt', help='file to store preprocessed data')
    parser.add_argument('--nsentences', default=50000, help='how many sentences to store')
    args = parser.parse_args()
    cp = CorpusPreprocessor(args.source, args.nsentences)
    cp.save(args.destination)


if __name__ == '__main__':
    main()
