from argparse import ArgumentParser, Namespace
import torch


def add_train_args(parser: ArgumentParser):
    """
    Adds training arguments to an ArgumentParser.

    :param parser: An ArgumentParser.
    """
    # General arguments
    parser.add_argument('--mode',  type=str, default='train',
                        choices=['train','eval'], help='training or evaluating')
    parser.add_argument('--data_path', type=str,
                        help='Path to data CSV file')
    parser.add_argument('--split_path', type=str,
                        help='Path to .npy file containing train/val/test split indices')
    parser.add_argument('--log_dir', type=str, default=None,
                        help='Directory where model checkpoints will be saved')
    parser.add_argument('--task', type=str, default='regression', choices=('regression', 'classification'),
                        help='Regression or classification task')
    parser.add_argument('--seed', type=int, default=0,
                        help='Random seed to use when splitting data into train/val/test sets.'
                             'When `num_folds` > 1, the first fold uses this seed and all'
                             'subsequent folds add 1 to the seed.')
    # Training arguments
    parser.add_argument('--n_epochs', type=int, default=30,
                        help='Number of epochs to run')
    parser.add_argument('--batch_size', type=int, default=50,
                        help='Batch size')
    parser.add_argument('--warmup_epochs', type=float, default=2.0,
                        help='Number of epochs during which learning rate increases linearly from'
                             'init_lr to max_lr. Afterwards, learning rate decreases exponentially'
                             'from max_lr to final_lr.')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Learning rate')
    parser.add_argument('--num_workers', type=int, default=6,
                        help='Number of workers to use in dataloader')
    parser.add_argument('--no_shuffle', action='store_true', default=False,
                        help='Whether or not to retain default ordering during training')
    parser.add_argument('--shuffle_pairs', action='store_true', default=False,
                        help='Whether or not to shuffle only pairs of stereoisomers')

    # Model arguments
    parser.add_argument('--gnn_type', type=str,
                        choices=['gin', 'gcn', 'dmpnn'],
                        help='Type of gnn to use')
    parser.add_argument('--hidden_size', type=int, default=300,
                        help='Dimensionality of hidden layers in MPN')
    parser.add_argument('--ffn_hidden_size', type=int, default=300,
                        help='Dimensionality of hidden layers in FFN')
    parser.add_argument('--depth', type=int, default=3,
                        help='Number of message passing steps')
    parser.add_argument('--dropout', type=float, default=0.2,
                        help='Dropout probability')
    parser.add_argument('--graph_pool', type=str, default='sum',
                        choices=['sum', 'mean', 'max', 'attn', 'set2set'],
                        help='How to aggregate atom representations to molecule representation')
    parser.add_argument('--atom_messages', action='store_true', default=False,
                        help='atom center messages, instead of edge messages')

def modify_train_args(args: Namespace):
    """
    Modifies and validates training arguments in place.

    :param args: Arguments.
    """
    # shuffle=False for custom sampler
    if args.shuffle_pairs:
        setattr(args, 'no_shuffle', True)
    setattr(args, 'device', torch.device('cuda' if torch.cuda.is_available() else 'cpu'))


def parse_train_args() -> Namespace:
    """
    Parses arguments for training (includes modifying/validating arguments).

    :return: A Namespace containing the parsed, modified, and validated args.
    """
    parser = ArgumentParser()
    add_train_args(parser)
    args = parser.parse_args()
    modify_train_args(args)

    return args
