
import os, sys
import numpy as np
import pandas as pd
import pickle
import torch
from torch_geometric.data import Data, Dataset, DataLoader
from torch.utils.data.sampler import Sampler
from features.featurization import MolGraph
from typing import Union, List, Tuple, Dict
from itertools import chain


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

    def __init__(self, smiles: List, labels: List, args):
        super(MolDataset, self).__init__()
        
        self.smiles = smiles
        dtype= torch.long if args.task.lower() == 'classification' else torch.float
        self.labels = torch.tensor(labels, dtype= dtype)
        # self.data_map = {k: v for k, v in zip(range(len(self.smiles)), self.split)}
  
    def molgraph2data(self, molgraph: MolGraph, key):
        data = Data()
        data.x = torch.tensor(molgraph.f_atoms, dtype=torch.float)
        data.edge_index = torch.tensor(molgraph.edge_index, dtype=torch.long).t().contiguous()
        data.edge_attr = torch.tensor(molgraph.f_bonds, dtype=torch.float)
        data.y = self.labels[key]
        data.smiles = self.smiles[key]
        return data

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, key):
        ## FIXME: only one each time, slice ?
        smi = self.smiles[key]
        molgraph = MolGraph(smi, atom_features_extra=None, 
                                 bond_features_extra=None, 
                                 overwrite_default_atom_features=False, 
                                 overwrite_default_bond_features=False)
        mol = self.molgraph2data(molgraph, key)
        #mol = MolData(molgraph, smi, self.labels[key], self.atom_messages)
        return mol



def construct_loader(args, modes=('train', 'val')):

    if isinstance(modes, str):
        modes = [modes]


    # data_df = pd.read_pickle(args.data_path)
    # smiles = data_df.index.to_list()
    # labels = data_df.values.astype(np.float32)

    # first column is SMILES, labels will be the rest columns.
    # multiclass classification are not supported yet !
    data_df = pd.read_csv(args.data_path)
    smiles = data_df.iloc[:, 0].values
    labels = data_df.iloc[:, 1:].values

     # read train, val, test split, or create one and save to log_dir
    if args.split_path:
        with open(args.split_path, 'rb') as handle:
            split = pickle.load(handle)
    else:
        indices = list(range(len(labels)))
        # 0.8, 0.1, 0.1
        lengths = [int(len(indices)*0.8), int(len(indices)*0.1)]
        lengths+=[len(indices) -sum(lengths)]
        # random split
        split = torch.utils.data.random_split(indices, lengths)
        split = [list(s) for s in split]
        with open(os.path.join(args.log_dir,"data.split.pkl"), 'wb') as out:
            pickle.dump(split, out)

    # 
    if args.task.lower() == 'classification':
        args.output_size = len(np.unique(labels))
        labels = np.squeeze(labels).astype(int)
    else:
        args.output_size = labels.shape[1]
        labels = labels.astype(np.float32)

    # construct dataloaders
    loaders = []
    for mode in modes:
        mode_idx = 0 if mode == 'train' else 1 if mode == 'val' else 2
        smiles_subset = [smiles[i] for i in split[mode_idx] ]
        labels_subset = [labels[i] for i in split[mode_idx] ]
        dataset = MolDataset(smiles_subset, labels_subset, args)
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