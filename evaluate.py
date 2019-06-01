import random
import torch
import argparse
from torch.utils.data import DataLoader
from dataset import CorpusDataset
from model import Net
from preprocessor import CorpusPreprocessor


def evaluate(net, device, testloader, criterion=None):
    if criterion is None:
        net.to(device)
    n_test_batches = len(testloader)

    net.eval()
    test_loss = 0.0
    correct = 0
    total = 0

    for (masked_sents, words, labels) in testloader:
        masked_sents = masked_sents.to(device)
        words = words.to(device)
        labels = labels.to(device)
        with torch.no_grad():
            outputs = net(masked_sents, words)
            if criterion is not None:
                test_loss += criterion(outputs, labels).item()
            predicted = outputs.round()
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    if criterion is not None:
        test_loss /= n_test_batches
        return accuracy, test_loss
    else:
        return accuracy


def main():
    random.seed(420)
    parser = argparse.ArgumentParser(description='Evaluate accuracy of trained model',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--model', default='checkpoint.pt', help='model to use')
    parser.add_argument('--data', default='corpus.pt', help='preprocessed data file')
    parser.add_argument('--device', default='cuda:0' if torch.cuda.is_available() else 'cpu', help='device to use')
    parser.add_argument('--batch', default=64, help='batch size')
    args = parser.parse_args()

    cp = CorpusPreprocessor()
    cp.load(args.data)

    net = Net(len(cp.alphabet), cp.max_sentence_length, cp.max_word_length, 100)
    net.load_state_dict(torch.load(args.model, map_location=args.device))

    _, testset = CorpusDataset.split(cp, 0.8)
    testloader = DataLoader(testset, batch_size=args.batch, num_workers=4)

    accuracy = evaluate(net, args.device, testloader)
    print('Model accuracy: {}'.format(accuracy))


if __name__ == '__main__':
    main()
