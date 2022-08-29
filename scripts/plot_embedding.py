import os
import time
from tqdm import tqdm
from argparse import ArgumentParser

import torch
from torch.utils.data import DataLoader
from esm import Alphabet
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import umap

from DeepSecE.dataset import TXSESequenceDataSet
from DeepSecE.model import EffectorTransformer
from DeepSecE.utils import label2index
from DeepSecE.trainer import set_seed


def plot_tsne(x, y, color_dict, title, ignore_ylabel=False):
    
    tsne = TSNE(n_components=2, random_state=42)
    x_tsne = tsne.fit_transform(x)
    
    x_min, x_max = x_tsne.min(0), x_tsne.max(0)
    x_norm = (x_tsne - x_min) / (x_max - x_min)

    for i, (name, color) in enumerate(color_dict.items()):
        plt.scatter(x_norm[y == i, 0], x_norm[y == i, 1], c=color, alpha=0.8, s=10, label=name)
 
    plt.xlabel("T-SNE Dimension 1")
    if not ignore_ylabel:
        plt.ylabel("T-SNE Dimension 2")

    plt.title(title)
    plt.legend(loc="lower right")
 
 
def plot_umap(x, y, color_dict, title, ignore_ylabel=False):
    reducer = umap.UMAP(random_state=42)
    x_umap = reducer.fit_transform(x)
    
    x_min, x_max = x_umap.min(0), x_umap.max(0)
    x_norm = (x_umap - x_min) / (x_max - x_min)
    
    for i, (name, color) in enumerate(color_dict.items()):
        plt.scatter(x_norm[y == i, 0], x_norm[y == i, 1], c=color, alpha=0.8, s=10, label=name)
        
    plt.xlabel("UMAP Dimension 1")
    if not ignore_ylabel:
        plt.ylabel("UMAP Dimension 2")

    plt.title(title)
    plt.legend(loc="lower right")


def plot_esm_embedding(model, fasta, batch_size, device):
    
    print(f'Loading FASTA Dataset from {fasta}')
    dataset = TXSESequenceDataSet(fasta_path=fasta, transform=label2index, mode='test', seed=42)
    alphabet = Alphabet.from_architecture("roberta_large")
    loader = DataLoader(dataset, batch_size=batch_size, collate_fn=alphabet.get_batch_converter(), num_workers=4, shuffle=False)
    
    model.eval()
    
    truth = []
    embeddings = []
    
    with torch.no_grad():
        for labels, strs, toks in tqdm(loader):
            toks = toks.to(device)
            toks = toks[:, :1022]
            batch = toks.shape[0]
            out = model.pretrained_model(toks, repr_layers=[33], return_contacts=False)
            emb = torch.cat([out["representations"][33][i, 1: len(strs[i]) + 1].mean(0).unsqueeze(0) for i in range(batch)], dim=0)
            embeddings.append(emb)
            y = torch.tensor(labels, device=device).long()
            truth.append(y)

    embeddings = torch.cat(embeddings).cpu().numpy()
    truth = torch.cat(truth).cpu().numpy()
    
    color_dict = {'Non-effector':'#dddddd', 'T1SE': '#ffbe0b', 'T2SE': '#fb5607',
              'T3SE': '#ff006e', 'T4SE': '#8338ec', 'T6SE': '#3a86ff'}
    
    plot_umap(embeddings, truth, color_dict, title="UMAP projection of ESM1b embedding")
    # plot_tsne(embeddings, truth, color_dict, title="T-SNE projection of ESM1b embedding")


def plot_effector_embedding(model, fasta, batch_size, device):
    
    print(f'Loading FASTA Dataset from {fasta}')
    dataset = TXSESequenceDataSet(fasta_path=fasta, transform=label2index, mode='test', seed=42)
    alphabet = Alphabet.from_architecture("roberta_large")
    loader = DataLoader(dataset, batch_size=batch_size, collate_fn=alphabet.get_batch_converter(), num_workers=4, shuffle=False)
    
    model.eval()
    
    truth = []
    embeddings = []
    
    with torch.no_grad():
        for labels, strs, toks in tqdm(loader):
            toks = toks.to(device)
            out = model(strs, toks) # (bs, hid_dim)
            embeddings.append(out)
            y = torch.tensor(labels, device=device).long()
            truth.append(y)

    embeddings = torch.cat(embeddings).cpu().numpy()
    truth = torch.cat(truth).cpu().numpy()
    
    color_dict = {'Non-effector':'#dddddd', 'T1SE': '#ffbe0b', 'T2SE': '#fb5607',
              'T3SE': '#ff006e', 'T4SE': '#8338ec', 'T6SE': '#3a86ff'}
    
    plot_umap(embeddings, truth, color_dict, title="UMAP projection of effector embedding", ignore_ylabel=True)
    # plot_tsne(embeddings, truth, color_dict, title="T-SNE projection of effector embedding", ignore_ylabel=True)


def main(args):
    
    set_seed(42)
    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)
        
    if args.no_cuda:
        device = torch.device("cpu")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    print(f'Using device {device} for prediction')
    start_time = time.time()
    
    model = EffectorTransformer(1280, 33, 1, 4, 256, 0.4, num_classes=6, return_embedding=True)
    model.to(device)
    print(f'Loading model from {args.model_location}')
    if args.no_cuda:
        model.load_state_dict(torch.load(args.model_location, map_location="cpu"))
    else:
        model.load_state_dict(torch.load(args.model_location))
    
    plt.figure(figsize=(13, 6))
    plt.subplot(1, 2, 1)
    plot_esm_embedding(model, args.fasta_path, args.batch_size, device)
    plt.subplot(1, 2, 2)
    plot_effector_embedding(model, args.fasta_path, args.batch_size, device)
    plt.savefig(os.path.join(args.out_dir, 'embedding.png'), dpi=300)
    
    end_time = time.time()
    secs = end_time - start_time
    
    print(f'It took {secs:.1f}s to finish the prediction')
    
    
if __name__ == '__main__':
    
    parser = ArgumentParser()
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--fasta_path', required=True, type=str)
    parser.add_argument('--model_location', required=True, type=str)
    parser.add_argument('--out_dir', default='./', type=str)
    parser.add_argument('--no_cuda', action='store_true')
    
    args = parser.parse_args()

    main(args)
