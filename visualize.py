import torch
import random
import argparse
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from model import Net
from preprocessor import CorpusPreprocessor
from dataset import CorpusDataset


def main():
    random.seed(420)
    parser = argparse.ArgumentParser(description='Visualize word embedder',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--model', default='checkpoint.pt', help='model checkpoint to use')
    parser.add_argument('--data', default='corpus.pt', help='preprocessed data file')
    parser.add_argument('--device', default='cuda:0' if torch.cuda.is_available() else 'cpu', help='device to use')
    parser.add_argument('--output', default='vis.png', help='where to save result')
    parser.add_argument('--show', action='store_true', help='display result')
    args = parser.parse_args()

    cp = CorpusPreprocessor()
    cp.load(args.data)

    net = Net(len(cp.alphabet), cp.max_sentence_length, cp.max_word_length)
    net.load_state_dict(torch.load(args.model, map_location=args.device))
    net.to(args.device)
    we = net.we

    _, dataset = CorpusDataset.split(cp, 0.8)
    testpairs = [
        ('warszawa', 'polska', 'paryż', 'francja'),
        ('niemcy', 'europa', 'chiny', 'azja'),
        ('stół', 'stołu', 'dom', 'domu'),
        ('król', 'mężczyzna', 'królowa', 'kobieta'),
        ('ojciec', 'mężczyzna', 'matka', 'kobieta'),
        ('on', 'mężczyzna', 'ona', 'kobieta'),
        ('ciepło', 'lato', 'zimno', 'zima'),
        ('ciemno', 'noc', 'jasno', 'dzień'),
        ('samochód', 'koła', 'samolot', 'skrzydła'),
        ('ojciec', 'mężczyzna', 'jezioro', 'ciasto'),
        ('pis', 'kaczyński', 'po', 'tusk'),
    ]
    testwords = list({w for doublepair in testpairs for w in doublepair})
    words = testwords + random.sample(cp.words, 1000)
    t_words = torch.stack(tuple(dataset.word2tensor(w) for w in words)).to(args.device)

    we.eval()
    with torch.no_grad():
        outputs = we(t_words).cpu().numpy()
    outputs_emb = TSNE(n_components=2).fit_transform(outputs)

    for pair in testpairs:
        t_pair = tuple(t_words[testwords.index(w)] for w in pair)
        distance = torch.sqrt(torch.sum(((t_pair[0] - t_pair[1]) - (t_pair[2] - t_pair[3])) ** 2)).item()
        print(f'({pair[0]} - {pair[1]}) - ({pair[2]} - {pair[3]}) = {distance}')

    plt.figure(figsize=(10, 10))
    ax = plt.axes(frameon=False)
    plt.setp(ax, xticks=(), yticks=())
    plt.subplots_adjust(left=0.0, bottom=0.0, right=1.0, top=0.9, wspace=0.0, hspace=0.0)

    xs = list(outputs_emb[:, 0])
    ys = list(outputs_emb[:, 1])
    plt.scatter(xs, ys, c='lightgrey')
    for x, y, w in zip(xs, ys, words):
        if w in testwords:
            ax.annotate(w, (x, y))
    plt.show()


if __name__ == '__main__':
    main()
