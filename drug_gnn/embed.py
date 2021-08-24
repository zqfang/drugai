"""
Extract gradient, input, output of layers

"""

import os
import torch
import numpy as np
from tqdm.auto import tqdm
from model.data import construct_loader
from model.utils import create_logger, get_loss_func

from model.gnn import GNN
from model.parsing import parse_train_args


class SaveOutput:
    """Example 1
    """
    def __init__(self):
        self.outputs = []
        self.inputs = []
        
    def __getitem__(self, idx):
        if 0 <= idx < len(self.outputs):
            return self.outputs[idx] 
        
    def __call__(self, module, module_in, module_out):
        self.outputs.append(module_out)
        self.inputs.append(module_in[0])
    
    def __len__(self):
        return len(self.outputs)
    
    def clear(self):
        self.outputs = []
        
    def module_output_to_numpy(self, tensor):
        return tensor.detach().to('cpu').numpy()  


class SaveGrads:
    def __init__(self):
        self.outputs = []
        
    def __getitem__(self, idx):
        if 0 <= idx < len(self.outputs):
            return self.outputs[idx] 
        
    def __call__(self, module, grad_in, grad_out):
        self.outputs.append(grad_out)
    
    def __len__(self):
        return len(self.outputs)
    
    def clear(self):
        self.outputs = []
        
    def module_output_to_numpy(self, tensor):
        return tensor.detach().to('cpu').numpy()  



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
save_output = SaveOutput()
hook_handles = []
# inject
handle = model.ffn.register_forward_hook(save_output)
hook_handles.append(handle)
    

# predict test data

model.eval()
error, correct = 0, 0

preds, ys = [], []
with torch.no_grad():
    for data in tqdm(test_loader, total=len(test_loader)):
        data = data.to(args.device)
        out = model(data)

        # error += loss(out, data.y).item()
        # preds.extend(out.detach().cpu().tolist())
        # if task == 'classification':
        #     predicted = torch.softmax(out.detach(), dim=1).argmax(dim=1)
        #     correct += (predicted == data.y).sum().double()
        #     ys.extend(data.y.detach().cpu().tolist()

embed = torch.cat(save_output.inputs, dim=0).cpu().numpy()
np.save(os.path.join(args.log_dir, "preds.embedding.npy"), embed)

# remove handle
for h in hook_handles: h.remove()
print('done')



