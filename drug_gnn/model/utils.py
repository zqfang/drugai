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
