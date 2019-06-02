import torch
from torch.utils import data
from preprocessor import CorpusPreprocessor


class CorpusDataset(data.Dataset):
    def __init__(self, cp: CorpusPreprocessor, range):
        self.cp = cp
        self.sentences = cp.sentences[range[0]:range[1]]
        self.chr_to_idx = dict((c, i) for i, c in enumerate(cp.alphabet))

    def word2tensor(self, word):
        t_word = torch.zeros(self.cp.max_word_length, len(self.cp.alphabet))
        for j, c in enumerate(word):
            t_word[j, self.chr_to_idx[c]] = 1.0
        return t_word

    def sentence2tensor(self, sentence):
        t_sentence = torch.zeros(self.cp.max_sentence_length, self.cp.max_word_length, len(self.cp.alphabet))
        for i, w in enumerate(sentence):
            if w != self.cp.MASK:
                for j, c in enumerate(w):
                    t_sentence[i, j, self.chr_to_idx[c]] = 1.
        return t_sentence

    def __getitem__(self, i):
        masked_sent, word, label = self.cp.mask_text(self.sentences[i])
        t_masked_sent = self.sentence2tensor(masked_sent)
        t_word = self.word2tensor(word)
        t_label = torch.Tensor([label])
        return t_masked_sent, t_word, t_label

    def __len__(self):
        return len(self.sentences)

    @classmethod
    def split(cls, cp: CorpusPreprocessor, ratio):
        split_point = int(ratio * len(cp.sentences))
        train_set = cls(cp, (0, split_point))
        test_set = cls(cp, (split_point + 1, len(cp.sentences)))
        return train_set, test_set
