"""
This example script for pytorch hooks. Given layer (name),
extract features(output): register_forward_hook
extract grads: register_backward_hook

"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Iterable, Callable



"""
example one
"""
class FeatureExtractor(nn.Module):
    def __init__(self, model: nn.Module, layers: Iterable[str]):
        super().__init__()
        self.model = model
        self.layers = layers
        self._features = {layer: torch.empty(0) for layer in layers}

        for layer_id in layers:
            layer = dict([*self.model.named_modules()])[layer_id]
            layer.register_forward_hook(self.save_outputs_hook(layer_id))

    def save_outputs_hook(self, layer_id: str) -> Callable:
        def fn(_, __, output):
            self._features[layer_id] = output
        return fn

    def forward(self, x: Tensor) -> Dict[str, Tensor]:
        _ = self.model(x)
        return self._features


resnet_features = FeatureExtractor(resnet50(), layers=["layer4", "avgpool"])
features = resnet_features(dummy_input)
print({name: output.shape for name, output in features.items()})


"""
example 2
"""
class SaveOutput:
    def __init__(self):
        self.outputs = []
        
    def __getitem__(self, idx):
        if 0 <= idx < len(self.outputs):
            return self.outputs[idx] 
        
    def __call__(self, module, module_in, module_out):
        self.outputs.append(module_out)
    
    def __len__(self):
        return len(self.outputs)
    
    def clear(self):
        self.outputs = []
        
    def module_output_to_numpy(self, tensor):
        return tensor.detach().to('cpu').numpy()  



save_output = SaveOutput()
hook_handles = []

for layer in model.ffn.modules():
    handle = layer.register_forward_hook(save_output)
    hook_handles.append(handle)


# get output
embeds = next(iter(train_loader))
inp_mhc, inp_ag = embeds['mhc_embed'], embeds['ag_embed']
outputs = model(inp_mhc, inp_ag)
# remove handle
for h in hook_handles: h.remove()


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