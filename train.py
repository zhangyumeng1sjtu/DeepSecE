import os
import time
import logging
from argparse import ArgumentParser

import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
import torch.optim.lr_scheduler as lrs
from sklearn.metrics import confusion_matrix
from tensorboardX import SummaryWriter

from warmup_scheduler import GradualWarmupScheduler
from esm import Alphabet

from DeepSecE.model import EffectorTransformer, ESM1bModel
from DeepSecE.dataset import TXSESequenceDataSet
from DeepSecE.utils import  label2index, viz_conf_matrix
from DeepSecE.trainer import train, test, set_seed, EarlyStopping


def main(args):

    set_seed(args.seed)

    # Configure logging
    log_dir = os.path.join(args.log_dir, f'Fold_{args.fold_num+1}')
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    logging.basicConfig(handlers=[
        logging.FileHandler(filename=os.path.join(log_dir, "training.log"), encoding='utf-8', mode='w+')],
        format="%(asctime)s %(levelname)s:%(message)s", datefmt="%F %A %T", level=logging.INFO)
    writer = SummaryWriter(log_dir)

    # Configure model
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if args.model == "effectortransformer":
        model = EffectorTransformer(1280, 33, hid_dim=args.hid_dim, num_layers=args.num_layers,
                            heads=args.num_heads, dropout_rate=args.dropout_rate, num_classes=6)
    elif args.model == "esm1bmodel":
        model = ESM1bModel(1280, 33, unfreeze_last=True, hid_dim=args.hid_dim, dropout_rate=args.dropout_rate, num_classes=6)
    else:
        raise ValueError('Invalid model type!')
    model.to(device)

    # Configure datasets and dataloaders
    alphabet = Alphabet.from_architecture("roberta_large")
    train_dataset = TXSESequenceDataSet(fasta_path=os.path.join(args.data_dir, 'Train-2918.fasta'),
                                transform=label2index, mode='train', kfold=args.kfold, fold_num=args.fold_num, seed=args.seed)
    valid_dataset = TXSESequenceDataSet(fasta_path=os.path.join(args.data_dir, 'Train-2918.fasta'),
                                transform=label2index, mode='valid', kfold=args.kfold, fold_num=args.fold_num, seed=args.seed)
    test_dataset = TXSESequenceDataSet(fasta_path=os.path.join(args.data_dir, 'Test-260.fasta'),
                               transform=label2index, mode='test')
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                              collate_fn=alphabet.get_batch_converter(), num_workers=args.num_workers, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size,
                              collate_fn=alphabet.get_batch_converter(), num_workers=args.num_workers)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size,
                             collate_fn=alphabet.get_batch_converter(), num_workers=args.num_workers)

    # Configure loss, optimizer, scheduler, early stopping
    criterion = nn.CrossEntropyLoss()
    if args.model == "esm1bmodel":
        optimizer_settings = [{
            'params':  filter(lambda p: p.requires_grad, model.pretrained_model.parameters()), 'lr': args.lr / 10
        }, {
            'params':  filter(lambda p: p.requires_grad, model.clf.parameters()), 'lr': args.lr
        }]
        optimizer = optim.Adam(optimizer_settings, weight_decay=args.weight_decay)
    else:
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    if args.lr_scheduler is None:
        scheduler = GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=args.warm_epochs)
        return [optimizer], [scheduler]
    else:
        if args.lr_scheduler == 'step':
            after_scheduler = lrs.StepLR(optimizer, step_size=args.lr_decay_steps, gamma=args.lr_decay_rate)
        elif args.lr_scheduler == 'cosine':
            after_scheduler = lrs.CosineAnnealingLR(optimizer, T_max=args.lr_decay_steps, eta_min=args.lr_decay_min_lr)
        else:
            raise ValueError('Invalid lr_scheduler type!')
        scheduler = GradualWarmupScheduler(
            optimizer, multiplier=1, total_epoch=args.warm_epochs, after_scheduler=after_scheduler)

    early_stopping = EarlyStopping(
        patience=args.patience, checkpoint_dir=log_dir)

    # Training epochs
    for epoch in range(args.max_epochs):
        start_time = time.time()

        train_loss, train_acc = train(model, train_loader, criterion, optimizer, device)
        valid_loss, valid_metrics = test(model, valid_loader, criterion, device)

        scheduler.step()

        end_time = time.time()
        epoch_secs = end_time - start_time

        valid_acc = valid_metrics['Accuracy']
        valid_f1 = valid_metrics['F1-score']
        valid_map = valid_metrics['AUPRC']

        logging.info(f'Epoch: {epoch+1:02} | Epoch Time: {epoch_secs:.2f}s')
        logging.info(f'Train Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.2f}%')
        logging.info(f'Valid Loss: {valid_loss:.3f} | Valid Acc: {valid_acc*100:.2f}%')
        logging.info(f'Valid F1: {valid_f1:.3f} | Valid mAP: {valid_map:.3f}')

        writer.add_scalar('Train/Loss', train_loss, epoch+1)
        writer.add_scalar('Train/Accuracy', train_acc, epoch+1)
        writer.add_scalar('Valid/Loss', valid_loss, epoch+1)
        for key, value in valid_metrics.items():
            writer.add_scalar('Valid/' + key, value, epoch+1)

        early_stopping(valid_f1, model)
        if early_stopping.early_stop:
            logging.info(f"Early stopping at Epoch {epoch+1}")
            break

    # Testing and evaluation
    model.load_state_dict(torch.load(os.path.join(log_dir, 'checkpoint.pt')))
    valid_best_loss, valid_best_metrics, valid_truth, valid_pred = test(model, valid_loader, criterion, device, True)
    _, test_final_metrics, test_truth, test_pred = test(model, test_loader, criterion, device, True)

    logging.info(f'Best Valid Loss: {valid_best_loss:.3f} | Acc: {valid_best_metrics["Accuracy"]*100:.2f}% |'
                 f' F1: {valid_best_metrics["F1-score"]:.3f} | mAP: {valid_best_metrics["AUPRC"]:.3f}')
    logging.info(f'Final Test Acc: {test_final_metrics["Accuracy"]*100:.2f}% |'
                 f' F1: {test_final_metrics["F1-score"]:.3f} | mAP: {test_final_metrics["AUPRC"]:.3f}')

    for key, value in valid_best_metrics.items():
        writer.add_scalar('Valid/Best ' + key, value)
    for key, value in test_final_metrics.items():
        writer.add_scalar('Test/Final ' + key, value)

    valid_cm = confusion_matrix(valid_truth, valid_pred)
    test_cm = confusion_matrix(test_truth, test_pred)

    labels = ['Non-effector', 'T1SE', 'T2SE', 'T3SE', 'T4SE', 'T6SE']

    writer.add_figure('Valid/conf_matrix', viz_conf_matrix(valid_cm, labels))
    writer.add_figure('Test/conf_matrix', viz_conf_matrix(test_cm, labels))
    writer.close()


if __name__ == '__main__':

    parser = ArgumentParser(description="Train a DeepSecE model for secreted effector prediction.")
    
    # Select Model
    parser.add_argument('--model', required=True, choices=['effectortransformer', 'esm1bmodel'], type=str,
                        help="model types available for training. [effectortransformer , esm1bmodel]")

    # Basic Training Control
    parser.add_argument('--batch_size', default=32, type=int,
                        help="bacth size used in training. (default: 32)")
    parser.add_argument('--num_workers', default=4, type=int,
                        help="num. of workers used in dataloader")
    parser.add_argument('--seed', default=42, type=int, 
                        help="random seed used in training. (default: 42)")
    parser.add_argument('--lr', default=1e-4, type=float,
                        help="learning rate. (default: 1e-4)")
    parser.add_argument('--warm_epochs', default=1, type=int,
                        help="num. of epochs under warm start. (default: 1)")
    parser.add_argument('--patience', default=5, type=int,
                        help="patience for early stopping. (default: 5")
    parser.add_argument('--lr_scheduler', choices=['step', 'cosine'], type=str,
                        help="learning rate scheduler. [step, cosine]")
    parser.add_argument('--lr_decay_steps', default=10, type=int, 
                        help="step of learning rate decay. (default: 10)")
    parser.add_argument('--lr_decay_rate', default=0.5, type=float,
                        help="ratio of learning rate decay. (default: 0.5)")
    parser.add_argument('--lr_decay_min_lr', default=5e-6, type=float,
                        help="minimum value of learning rate. (default: 5e-6)")

    # Training Info
    parser.add_argument('--max_epochs', default=30, type=int,
                        help="maximum num. of epochs. (default: 30")
    parser.add_argument('--data_dir', default='./data', type=str,
                        help="path to data. (default: ./data)")
    parser.add_argument('--weight_decay', default=1e-5, type=float,
                        help="weight decay for regularization. (default: 1e-5)")
    parser.add_argument('--log_dir', default='./logs', type=str,
                        help="path to the logging directory. (default: ./logs)")

    # Model Hyperparameters
    parser.add_argument('--hid_dim', default=256, type=int,
                        help="hidden dimension in the model. (default: 256)")
    parser.add_argument('--num_layers', default=1, type=int,
                        help="num. of training transformer layers (default: 1)")
    parser.add_argument('--num_heads', default=4, type=int,
                        help="num. of attention heads (default: 4)")
    parser.add_argument('--dropout_rate', default=0.4, type=float,
                        help="dropout rate (default: 0.4)")

    # KFold Support
    parser.add_argument('--kfold', default=5, type=int,
                        help="num. of CV folds. (default: 5)")
    parser.add_argument('--fold_num', default=0, type=int,
                        help="fold number (default: 0)")

    args = parser.parse_args()

    main(args)
