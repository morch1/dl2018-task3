import logging
import math
import random
import torch
import argparse
from torch import nn, optim
from torch.utils.data import DataLoader
from dataset import CorpusDataset
from model import Net
from preprocessor import CorpusPreprocessor
from evaluate import evaluate


def train(net, device, trainset, testset, batch_size, lr, max_epochs, early_stop, checkpoint_filename):
    logger = logging.getLogger('training')
    logger.setLevel(logging.INFO)
    log_format = logging.Formatter('%(asctime)s - %(message)s')
    fh = logging.FileHandler(checkpoint_filename + '.log')
    fh.setFormatter(log_format)
    logger.addHandler(fh)
    ch = logging.StreamHandler()
    ch.setFormatter(log_format)
    logger.addHandler(ch)

    def log_status(*args):
        logger.info('epoch {}, train loss {}, test loss {}, test accuracy {:.2f}'.format(*args))

    net.to(device)

    trainloader = DataLoader(trainset, batch_size=batch_size, num_workers=4)
    testloader = DataLoader(testset, batch_size=batch_size, num_workers=4)

    criterion = nn.BCELoss()
    optimizer = optim.Adam(net.parameters(), lr=lr)

    n_train_batches = len(trainloader)

    best = (0, math.inf, math.inf, 0.0)

    patience = early_stop

    logger.info('device {}, batch size {}, lr {}, max epochs {}, early stop {}'.format(device, batch_size, lr, max_epochs, early_stop))
    logger.info('started training')

    for epoch in range(max_epochs):
        net.train()
        train_loss = 0.0
        for (masked_sents, words, labels) in trainloader:
            masked_sents = masked_sents.to(device)
            words = words.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = net(masked_sents, words)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= n_train_batches

        accuracy, test_loss = evaluate(net, device, testloader, criterion)

        status = (epoch, train_loss, test_loss, accuracy)
        if test_loss < best[2]:
            patience = early_stop
            best = status
            torch.save(net.state_dict(), checkpoint_filename)
        else:
            patience -= 1

        log_status(*status)

        if patience == 0:
            break

    logger.info('finished training, best result (saved to {}):'.format(checkpoint_filename))
    log_status(*best)

    net.load_state_dict(torch.load(checkpoint_filename))


def main():
    random.seed(420)
    parser = argparse.ArgumentParser(description='Train model',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--data', default='corpus.pt', help='preprocessed data file')
    parser.add_argument('--checkpoint', default='checkpoint.pt', help='checkpoint filename')
    parser.add_argument('--device', default='cuda:0' if torch.cuda.is_available() else 'cpu', help='device to use')
    parser.add_argument('--batch', default=64, type=int, help='batch size')
    parser.add_argument('--lr', default=0.003, type=float, help='learning rate')
    parser.add_argument('--epochs', default=100, type=int, help='max number of epochs')
    parser.add_argument('--earlystop', default=5, type=int, help='early stop after this many epochs without improvement')
    parser.add_argument('--embedding_size', default=100, type=int, help='size of word embedding vector')
    args = parser.parse_args()

    cp = CorpusPreprocessor()
    cp.load(args.data)

    trainset, testset = CorpusDataset.split(cp, 0.8)

    net = Net(len(cp.alphabet), cp.max_sentence_length, cp.max_word_length, args.embedding_size)
    train(net, args.device, trainset, testset, args.batch, args.lr, args.epochs, args.earlystop, args.checkpoint)


if __name__ == '__main__':
    main()
