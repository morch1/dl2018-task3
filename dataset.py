import random
from torch.utils import data
from preprocessor import CorpusPreprocessor


class CorpusDataset(data.Dataset):
    def __init__(self, cp: CorpusPreprocessor, range):
        self.cp = cp
        self.sentences = cp.sentences[range[0]:range[1]]

    def __getitem__(self, i):
        return self.cp.mask_text(self.sentences[i])

    def __len__(self):
        return len(self.sentences)

    @classmethod
    def split(cls, cp: CorpusPreprocessor, ratio):
        split_point = int(ratio * len(cp.sentences))
        train_set = cls(cp, (0, split_point))
        test_set = cls(cp, (split_point + 1, len(cp.sentences)))
        return train_set, test_set


def main():
    # dataset preview
    random.seed(420)
    cp = CorpusPreprocessor()
    cp.load('corpus.json')
    train_set, test_set = CorpusDataset.split(cp, 0.8)
    print('train:')
    for i in range(10):
        print(train_set[i])
    print('\ntest:')
    for i in range(10):
        print(test_set[i])


if __name__ == '__main__':
    main()
