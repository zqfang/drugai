"""
Extract gradient, input, output of layers

"""

import os
import torch
import numpy as np
from tqdm.auto import tqdm
from model.data import construct_loader
from model.utils import create_logger, get_loss_func, SaveLayerOutput

from model.gnn import GNN
from model.parsing import parse_train_args


args = parse_train_args()
torch.manual_seed(args.seed)
logger = create_logger('train', args.log_dir)
loss = get_loss_func(args)


# predict test data
test_loader = construct_loader(args, modes='test')
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
with torch.no_grad():
    for data in tqdm(test_loader, total=len(test_loader)):
        data = data.to(args.device)
        out = model(data)

logger.info("Save molecule embeddings")
embed = torch.cat(save_output.inputs, dim=0).cpu().numpy()
np.save(os.path.join(args.log_dir, "preds.embedding.npy"), embed)

# remove handle
for h in hook_handles: h.remove()
logger.info("Done")



