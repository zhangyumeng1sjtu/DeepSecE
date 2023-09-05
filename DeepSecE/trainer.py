import os
import random

import numpy as np
import torch
from sklearn.metrics import accuracy_score

from DeepSecE.utils import metrics


def set_seed(seed):

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    
    
class EarlyStopping:

    def __init__(self, patience=10, checkpoint_dir='logs'):
        self.patience = patience
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.checkpoint_dir = checkpoint_dir

    def __call__(self, score, model, goal="maximize"):

        if goal == "minimize":
            score = -score

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(score, model)

        elif score < self.best_score:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(score, model)
            self.counter = 0

    def save_checkpoint(self, score, model):
        torch.save(model.state_dict(), os.path.join(
            self.checkpoint_dir, 'checkpoint.pt'))
        self.best_score = score


def train(model, iterator, criterion, optimizer, device):

    avg_loss = 0
    avg_acc = 0
    data_size = 0

    model.train()

    for labels, strs, toks in iterator:
        toks = toks.to(device)
        logits = model(strs, toks)
        y = torch.tensor(labels, device=device).long()

        _, pred = torch.max(logits.data, 1)

        loss = criterion(logits, y)
        acc = accuracy_score(y.cpu().numpy(), pred.cpu().numpy())

        avg_loss += loss.cpu().item() * len(labels)
        avg_acc += acc * len(labels)
        data_size += len(labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    avg_acc = avg_acc / data_size
    avg_loss = avg_loss / data_size

    return avg_loss, avg_acc


def test(model, iterator, criterion, device, return_array=False):

    avg_loss = 0
    data_size = 0

    truth = []
    probs = []
    preds = []

    model.eval()

    with torch.no_grad():
        for labels, strs, toks in iterator:
            toks = toks.to(device)
            logits = model(strs, toks)
            y = torch.tensor(labels, device=device).long()

            prob = torch.softmax(logits, dim=1)
            _, pred = torch.max(prob, 1)

            truth.append(y.detach().cpu().numpy())
            probs.append(prob.detach().cpu().numpy())
            preds.append(pred.detach().cpu().numpy())

            loss = criterion(logits, y)

            avg_loss += loss.cpu().item() * len(labels)
            data_size += len(labels)

    avg_loss = avg_loss / data_size

    truth = np.concatenate(truth)
    probs = np.concatenate(probs)
    preds = np.concatenate(preds)

    metrics_dict = metrics(truth, preds, probs)

    if return_array:
        return avg_loss, metrics_dict, truth, preds
    else:
        return avg_loss, metrics_dict
