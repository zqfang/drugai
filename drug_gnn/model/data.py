
import os, sys
import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data, Dataset, DataLoader
from torch.utils.data.sampler import Sampler
from features.featurization import MolGraph
from typing import Union, List, Tuple, Dict


class MolData(Data):
    def __init__(self, molgraph: MolGraph, smiles: str, 
                       label: Union[int, List[float]], 
                       atom_messages: bool =False):
        self.x = torch.tensor(molgraph.f_atoms, dtype=torch.float)
        self.edge_index = torch.tensor(molgraph.edge_index, dtype=torch.long).t().contiguous()
        self.edge_attr = torch.tensor(molgraph.f_bonds, dtype=torch.float)
        if isinstance(label, int): 
            label = [label]
            self.y = torch.tensor(label, dtype=torch.long) # need to be torch.long if classification
        else:
            self.y = torch.tensor(label, dtype=torch.float)
        self.smiles = smiles
        if not atom_messages:
            self.edge_index_linegraph = torch.tensor(molgraph.edge_index_linegraph, dtype=torch.long).contiguous()
        self.b2a = molgraph.b2a
        self.a2b = molgraph.a2b

    def __inc__(self, key, value):
        """
        batching increment
        """
        if key == "edge_index_linegraph": # directed line_graph
            return self.edge_attr.size(0) # need to check this !
        elif key == "edge_index":
            return self.x.size(0)
        elif key == "b2a": # incread atom index number
            return self.x.size(0) 

        return super().__inc__(key, value)

    def __cat_dim__(self, key, value):
        """
        Batching concate
        """
        if key == "edge_index_linegraph":
            return 1
        elif key == "edge_index":
            return 1
        elif key == "y_multivalues":
            # Batching Along New Dimensions, return None.
            # That'is a list of attributes of shape [num_features] 
            # should be returned as [num_examples, num_features]
            return None #
        return super().__inc__(key, value)


class MolDataset(Dataset):

    def __init__(self, smiles, labels, args, atom_messages=False, mode='train'):
        super(MolDataset, self).__init__()
        
        if args.split_path:
            self.split_idx = 0 if mode == 'train' else 1 if mode == 'val' else 2
            self.split = np.load(args.split_path, allow_pickle=True)[self.split_idx]
        else:
            self.split = list(range(len(smiles)))  # fix this
        self.smiles = [smiles[i] for i in self.split]
        self.labels = [labels[i] for i in self.split]
        self.data_map = {k: v for k, v in zip(range(len(self.smiles)), self.split)}
        self.task = args.task
        self.atom_messages = atom_messages

        # if mode == 'train':
        #     self.mean = np.mean(self.labels) # FIXME
        #     self.std = np.std(self.labels)

    def molgraph2data(self, molgraph, key):
        data = Data()
        data.x = torch.tensor(molgraph.f_atoms, dtype=torch.float)
        data.edge_index = torch.tensor(molgraph.edge_index, dtype=torch.long).t().contiguous()
        data.edge_attr = torch.tensor(molgraph.f_bonds, dtype=torch.float)
        data.y = torch.tensor([self.labels[key]], dtype= torch.long if self.task == 'classification' else torch.float)
        data.smiles = self.smiles[key]
        return data

    def __len__(self):
        return len(self.smiles)

    def __getitem__(self, key):
        smi = self.smiles[key]
        molgraph = MolGraph(smi, atom_features_extra=None, 
                                 bond_features_extra=None, 
                                 overwrite_default_atom_features=False, 
                                 overwrite_default_bond_features=False)
        mol = self.molgraph2data(molgraph, key)
        #mol = MolData(molgraph, smi, self.labels[key], self.atom_messages)
        return mol



class StereoSampler(Sampler):

    def __init__(self, data_source):
        self.data_source = data_source

    def __iter__(self):
        groups = [[i, i + 1] for i in range(0, len(self.data_source), 2)]
        idx = torch.randperm(len(groups))
        groups = groups[idx]
        # np.random.shuffle(groups)
        indices = list(chain(*groups))
        return iter(indices)

    def __len__(self):
        return len(self.data_source)


def construct_loader(args, modes=('train', 'val')):

    if isinstance(modes, str):
        modes = [modes]

    data_df = pd.read_pickle(args.data_path)
    smiles = data_df.index.to_list()
    labels = data_df.values.astype(np.float32)

    # data_df = pd.read_csv(args.data_path)
    # smiles = data_df.iloc[:, 0].values
    # labels = data_df.iloc[:, 1:].values
    # FIXME: 
    if args.task == 'classification':
        args.output_size = len(np.unique(labels))
        labels = np.squeeze(labels).astype(int)
    else:
        args.output_size = labels.shape[1]
        labels = labels.astype(np.float32)
    loaders = []
    for mode in modes:
        dataset = MolDataset(smiles, labels, args, mode)
        loader = DataLoader(dataset=dataset,
                            batch_size=args.batch_size,
                            shuffle=not args.no_shuffle if mode == 'train' else False,
                            num_workers=args.num_workers,
                            pin_memory=True,
                            sampler=StereoSampler(dataset) if args.shuffle_pairs else None)
        loaders.append(loader)

    if len(loaders) == 1:
        return loaders[0]
    else:
        return loaders