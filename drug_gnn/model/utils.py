import os
import logging
from argparse import Namespace
from torch import nn



def create_logger(name: str, log_dir: str = None) -> logging.Logger:
    """
    Creates a logger with a stream handler and file handler.

    :param name: The name of the logger.
    :param log_dir: The directory in which to save the logs.
    :return: The logger.
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.propagate = False

    # Set logger
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    logger.addHandler(ch)

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    fh = logging.FileHandler(os.path.join(log_dir, name + '.log'))
    fh.setLevel(logging.INFO)
    logger.addHandler(fh)

    return logger


def get_loss_func(args: Namespace) -> nn.Module:
    """
    Gets the loss function corresponding to a given dataset type.

    :param args: Namespace containing the dataset type ("classification" or "regression").
    :return: A PyTorch loss function.
    """
    if args.task == 'classification':
        return nn.CrossEntropyLoss()

    if args.task == 'regression':
        return nn.MSELoss()

    raise ValueError(f'Dataset type "{args.task}" not supported.')


class SaveLayerOutput:
    """Helper function for saving layer input and output
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


class SaveLayerGrads:
    """Helper function for save layer gradients
    """
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