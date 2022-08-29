import numpy as np
from torch.utils.data import Dataset
from sklearn.model_selection import StratifiedKFold, train_test_split

from esm import FastaBatchedDataset


class TXSESequenceDataSet(Dataset):

    def __init__(self, fasta_path, transform=None, mode='train', kfold=0, fold_num=0, seed=42):
        self.fasta_path = fasta_path
        self.transform = transform
        self.kfold = kfold
        self.mode = mode
        self.fold_num = fold_num
        self.seed = seed

        self.check_dataset()

    def check_dataset(self):
        tempdataset = FastaBatchedDataset.from_file(self.fasta_path)
        sequence_labels = [label[:4] for label in tempdataset.sequence_labels]
        sequence_strs = tempdataset.sequence_strs

        if self.mode == 'test':
            self.sequence_labels = np.array(sequence_labels)
            self.sequence_strs = np.array(sequence_strs)

        else:
            if self.kfold != 0:
                kf = StratifiedKFold(n_splits=self.kfold,
                                     shuffle=True, random_state=self.seed)
                train_idx, valid_idx = list(kf.split(sequence_strs, sequence_labels))[
                    self.fold_num]
            else:
                train_idx, valid_idx = train_test_split(
                    sequence_strs, sequence_labels, test_size=0.2, random_state=self.seed)

            if self.mode == 'train':
                self.sequence_labels = np.array(sequence_labels)[train_idx]
                self.sequence_strs = np.array(sequence_strs)[train_idx]
            else:
                self.sequence_labels = np.array(sequence_labels)[valid_idx]
                self.sequence_strs = np.array(sequence_strs)[valid_idx]

    def __getitem__(self, idx):
        label = self.sequence_labels[idx]
        seq_str = self.sequence_strs[idx]
        if self.transform is not None:
            label = self.transform(label)
        return label, seq_str

    def __len__(self):
        return len(self.sequence_labels)
