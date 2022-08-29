import os
from argparse import ArgumentParser
from tqdm import tqdm

import torch
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from esm import Alphabet
from sklearn.metrics import roc_curve, auc, confusion_matrix

from DeepSecE.dataset import TXSESequenceDataSet
from DeepSecE.model import EffectorTransformer
from DeepSecE.utils import label2index
from DeepSecE.trainer import set_seed, one_hot_encoding


def plot_roc(y, pred, colors, labels):

    num_classes = pred.shape[1]

    for i in range(num_classes):
        fpr, tpr, _ = roc_curve(one_hot_encoding(y)[:, i], pred[:, i])
        plt.plot(fpr, tpr, color=colors[i], label='%s (AUC = %0.3f)' % (
            labels[i], auc(fpr, tpr)), lw=1.5, alpha=.8)

    plt.plot([0, 1], [0, 1], linestyle='--', lw=1.5, color='k', alpha=.8)
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive rate')
    plt.grid(ls='--', alpha=0.5)
    plt.legend(loc='best')
    plt.title('Receiver Operating Characteristic Curve')


def viz_conf_matrix(cm, labels):
    cm_sum = np.sum(cm, axis=1, keepdims=True)
    cm_perc = cm / cm_sum * 100
    annot = np.empty_like(cm).astype(str)
    nrows, ncols = cm.shape
    for i in range(nrows):
        for j in range(ncols):
            p = cm_perc[i, j]
            if p < 1:
                annot[i, j] = ''
            else:
                annot[i, j] = '%.1f%%' % p
    cm = pd.DataFrame(cm, index=labels, columns=labels)
    cm.index.name = 'Actual'
    cm.columns.name = 'Predicted'
    sns.heatmap(cm, annot=annot, fmt='', cmap="Blues")
    plt.title('Confusion Matrix')


def main(args):

    set_seed(42)
    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)

    if args.no_cuda:
        device = torch.device("cpu")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    colors = ['#808080', '#ffbe0b', '#fb5607', '#ff006e', '#8338ec', '#3a86ff']
    plot_labels = ['Non-effector', 'T1SE', 'T2SE', 'T3SE', 'T4SE', 'T6SE']

    model = EffectorTransformer(1280, 33, 1, 4, 256, 0.4, num_classes=6)
    model.to(device)
    
    if args.no_cuda:
        model.load_state_dict(torch.load(args.model_location, map_location="cpu"))
    else:
        model.load_state_dict(torch.load(args.model_location))

    dataset = TXSESequenceDataSet(fasta_path=args.fasta_path,
                          transform=label2index, mode='test', seed=42)
    alphabet = Alphabet.from_architecture("roberta_large")
    loader = DataLoader(dataset, batch_size=args.batch_size,
                        collate_fn=alphabet.get_batch_converter(), num_workers=4, shuffle=False)

    model.eval()

    truth = []
    probs = []
    preds = []

    with torch.no_grad():
        for labels, strs, toks in tqdm(loader):
            toks = toks.to(device)
            out = model(strs, toks)
            prob = torch.softmax(out, dim=1)
            _, pred = torch.max(prob, 1)
            probs.append(prob)
            preds.append(pred)
            y = torch.tensor(labels, device=device).long()
            truth.append(y)

    probs = torch.cat(probs).cpu().numpy()
    truth = torch.cat(truth).cpu().numpy()
    preds = torch.cat(preds).cpu().numpy()

    plt.figure(figsize=(14, 6))
    plt.subplot(1, 2, 1)
    plot_roc(truth, probs, colors, plot_labels)
    plt.subplot(1, 2, 2)
    cm = confusion_matrix(truth, preds)
    viz_conf_matrix(cm, plot_labels)
    plt.savefig(os.path.join(args.out_dir, 'test.png'), dpi=300)


if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--fasta_path', required=True, type=str)
    parser.add_argument('--model_location', required=True, type=str)
    parser.add_argument('--out_dir', default='./', type=str)
    parser.add_argument('--no_cuda', action='store_true')

    args = parser.parse_args()

    main(args)
