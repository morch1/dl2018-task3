import torch
from torch import nn


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.shape[0], -1)


class WordEmbedder(nn.Module):
    """
    input:  batch_size x max_word_length x alphabet_size
    output: batch_size x embedding_size
    """

    def __init__(self, alphabet_size, max_word_length, embedding_size):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(3, alphabet_size), padding=(1, 0)), nn.BatchNorm2d(32), nn.ReLU(),
            Flatten(),
            nn.Linear(32 * max_word_length, embedding_size),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.net(x.unsqueeze(1))


class SentenceEmbedder(nn.Module):
    """
    input:  batch_size x max_sentence_length x max_word_length x alphabet_size
    output: 2 x batch_size x max_sentence_length x embedding_size
    """

    def __init__(self, word_embedder, alphabet_size, max_sentence_length, max_word_length, embedding_size):
        super().__init__()
        self.we = word_embedder
        self.lstm = nn.LSTM(embedding_size, embedding_size, bidirectional=True)
        self.alphabet_size = alphabet_size
        self.max_sentence_length = max_sentence_length
        self.max_word_length = max_word_length
        self.embedding_size = embedding_size

    def forward(self, x):
        batch_size = x.shape[0]
        words = x.view(batch_size * self.max_sentence_length, self.max_word_length, self.alphabet_size)
        words_emb = self.we(words).view(batch_size, self.max_sentence_length, self.embedding_size).transpose(0, 1)
        output, _ = self.lstm(words_emb)
        return output.view(self.max_sentence_length, batch_size, 2, self.embedding_size).transpose(0, 2)


class Net(nn.Module):
    """
    input:  batch_size x max_sentence_length x max_word_length x alphabet_size,
            batch_size x max_word_length x alphabet_size
    output: batch_size
    """

    embedding_size = 100

    def __init__(self, alphabet_size, max_sentence_length, max_word_length):
        super().__init__()
        self.we = WordEmbedder(alphabet_size, max_word_length, self.embedding_size)
        self.se = SentenceEmbedder(self.we, alphabet_size, max_sentence_length, max_word_length, self.embedding_size)
        self.lin = nn.Sequential(
            Flatten(),
            nn.Linear((2 * max_sentence_length + 1) * self.embedding_size, 1),
            nn.Sigmoid(),
        )

    def forward(self, masked_sents, words):
        masked_sents_emb = self.se(masked_sents)
        words_emb = self.we(words)
        x = torch.cat((masked_sents_emb[0], masked_sents_emb[1], words_emb.unsqueeze(1)), dim=1)
        output = self.lin(x)
        return output
