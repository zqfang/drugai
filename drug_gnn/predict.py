"""
Extract gradient, input, output of layers

"""

import os
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm.auto import tqdm
from model.utils import create_logger, get_loss_func, SaveLayerOutput

from model.gnn import GNN
from model.parsing import parse_train_args
from model.data import MolDataset
from torch_geometric.data import DataLoader
from rdkit import Chem

global args
args = parse_train_args()
torch.manual_seed(args.seed)
logger = create_logger('predict', args.log_dir)
loss = get_loss_func(args)

# read data in
data_df = pd.read_csv(args.data_path)
mask = data_df.iloc[:, 0].apply(lambda x: True if Chem.MolFromSmiles(x) else False) # valid smiles
smiles = data_df.iloc[:, 0][mask].values
logger.info(f"Parseble compound num: {len(smiles)}")
# construct dataloaders
dataset = MolDataset(args, smiles, None)
test_loader = DataLoader(dataset=dataset,batch_size=args.batch_size,
                    shuffle=False,num_workers=args.num_workers,
                    pin_memory=True,sampler=None)

# set parameter
setattr(args, 'output_size', 978) # FIXME
setattr(args, 'num_edge_features', test_loader.dataset.num_edge_features)
setattr(args, 'num_node_features', test_loader.dataset.num_node_features)

# load best model
model = GNN(args).to(args.device)
checkpoints = torch.load(os.path.join(args.log_dir, 'best_model'), map_location=args.device)
model.load_state_dict(checkpoints['state_dict'])


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

# remove handle
for h in hook_handles: h.remove()
## outputs
output_prefix = Path(args.data_path).stem
np.save(os.path.join(args.log_dir, f"{output_prefix}.embeddings.npy"), embed)

# save predictions
smiles = test_loader.dataset.smiles
logger.info("Save predicted expressions")
preds_path = os.path.join(args.log_dir, f"{output_prefix}.preds.expression.csv")
## FIXME: columns should match to your trained model
pd.DataFrame(preds, index=smiles, columns=checkpoints['targets']).to_csv(preds_path)
logger.info('Done')



