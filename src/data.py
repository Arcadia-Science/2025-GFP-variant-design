from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset, DataLoader


def get_seq_and_score_data():
    data = pd.read_csv(
        "/datasets/seq_and_score/seq_and_score.csv"
    )

    num_mutations = []
    for idx, row in data.iterrows():
        if row.mutations is np.nan:
            num_mutations.append(0)
        else:
            num_mutations.append(
                len(row.mutations.split(":"))
            )
    data["num_mutations"] = num_mutations
    return data


class VariantDataset(Dataset):
    def __init__(self, embeddings, labels):
        """
        Args:
            embeddings: ESM-2 embeddings (N x 5120)
            labels: Binary labels (N x 1)
        """
        self.embeddings = torch.FloatTensor(embeddings)
        self.labels = torch.FloatTensor(labels)
        
    def __len__(self):
        return len(self.embeddings)
        
    def __getitem__(self, idx):
        return self.embeddings[idx], self.labels[idx]


class BaseDataHolder(ABC):
    def __init__(self, df):
        self.df = df
        self.set_X_y()

    @abstractmethod
    def _get_X_y(self):
        ...

    def set_X_y(self):
        X, y = self._get_X_y()
        self.X = X.to(torch.float32)
        self.y = y.to(torch.float32)

    def for_per_of_data(self, per):
        return self.__class__(
            self.df.sample(n=int(per*self.df.shape[0]))
        )
        
    def for_cut_offs(self, min_distance=None, max_distance=None):
        if min_distance is None:
            min_distance = self.df.num_mutations.min()
        if max_distance is None:
            max_distance = self.df.num_mutations.max()

        return self.__class__(
            self.df[(self.df.num_mutations >= min_distance) & (self.df.num_mutations <= max_distance)]
        )

    def loader_all_data(self, batch_size=64):
        return DataLoader(
            VariantDataset(self.X, self.y),
            batch_size=batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True
        )
    
    def train_val_split(self, batch_size=64, test_size=0.3, rand_state=42):
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=0.30, random_state=42)
        
        train_loader = DataLoader(
            VariantDataset(X_train, y_train),
            batch_size=batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True
        )

        val_loader = DataLoader(
            VariantDataset(X_test, y_test),
            batch_size=batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )
    
        return train_loader, val_loader


class SmallEmbDataHolder(BaseDataHolder):
    def _get_X_y(self):
        return (
            torch.tensor(np.array(self.df.embedding.tolist())),
            torch.tensor(self.df.score.tolist())
        )


class CombinedDataHolder(BaseDataHolder):
    def __init__(self, df):
        super().__init__(df)
        self.embeddings_indexes = [str(e) for e in range(5120)]

    def _get_X_y(self):
        return torch.tensor(self.df.features.tolist()), torch.tensor(self.df.score.tolist())


class ESMDataHolder(BaseDataHolder):
    def __init__(self, df):
        self.embeddings_indexes = [str(e) for e in range(5120)]
        super().__init__(df)

    def _get_X_y(self):
        return torch.tensor(self.df[self.embeddings_indexes].to_numpy()), torch.tensor(self.df.score.values)