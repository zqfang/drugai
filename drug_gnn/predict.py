"""
Extract gradient, input, output of layers

"""

import os
import torch
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from model.utils import create_logger, get_loss_func, SaveLayerOutput

from model.gnn import GNN
from model.parsing import parse_train_args
from model.data import MolDataset
from torch_geometric.data import DataLoader


args = parse_train_args()
torch.manual_seed(args.seed)
logger = create_logger('eval', args.log_dir)
loss = get_loss_func(args)

# read data in
data_df = pd.read_csv(args.data_path)
smiles = data_df.iloc[:, 0].values
# construct dataloaders
loaders = []
dataset = MolDataset(smiles, None, args)
test_loader = DataLoader(dataset=dataset,batch_size=args.batch_size,
                    shuffle=False,num_workers=args.num_workers,
                    pin_memory=True,sampler=None)

# set parameter
setattr(args, 'output_size', 978) # FIXME
setattr(args, 'num_edge_features', test_loader.dataset.num_edge_features)
setattr(args, 'num_node_features', test_loader.dataset.num_node_features)

# load best model
model = GNN(args).to(args.device)
state_dict = torch.load(os.path.join(args.log_dir, 'best_model'), map_location=args.device)
model.load_state_dict(state_dict)


## hook
save_output = SaveLayerOutput()
hook_handles = []
# inject
handle = model.ffn.register_forward_hook(save_output)
hook_handles.append(handle)
    

# predict test data
model.eval()
preds, ys = [], []
with torch.no_grad():
    for data in tqdm(test_loader, total=len(test_loader)):
        data = data.to(args.device)
        out = model(data)
        preds.extend(out.detach().cpu().tolist())

logger.info("Save embeddings")
embed = torch.cat(save_output.inputs, dim=0).cpu().numpy()
np.save(os.path.join(args.log_dir, "preds.embedding.npy"), embed)

# remove handle
for h in hook_handles: h.remove()

# save predictions
smiles = test_loader.dataset.smiles
preds_path = os.path.join(args.log_dir, 'preds.csv')
pd.DataFrame(list(zip(smiles, preds)), columns=['smiles', 'prediction']).to_csv(preds_path, index=False)
logger.info('Done')



