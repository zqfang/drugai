# Drug GNN
Drug discovery using Graph Neural Network

This is a `Pytorch Geometric` implementation of the paper [Chemprop](https://pubs.acs.org/doi/10.1021/acs.jcim.9b00237).

Three GCNs are included:
- [DMPNN](https://pubs.acs.org/doi/10.1021/acs.jcim.9b00237):
- [GCN](https://arxiv.org/abs/1609.02907)
- [GIN](https://arxiv.org/abs/1905.12265>), see also [GINEConv](https://pytorch-geometric.readthedocs.io/en/latest/_modules/torch_geometric/nn/conv/gin_conv.html)


## Dependency
- numpy
- pandas
- scikit-learn
- optuna: hyperparameter search
- python >= 3.7
- Pytorch >= 1.5
- Pytorch Geometric >= 1.7
- RDkit


## Usage

Train:
```shell
python drug_gnn/train.py --data_path ${data} --split_path ${split} \
                        --task ${regression} \
                        --gnn_type {dmpnn, gcn, gin} --log_dir checkpoints \
                        --n_epochs 50 --batch_size 50
```

Evaluation:
```shell
python drug_gnn/eval.py --data_path ${data} --split_path ${split} \
                        --task ${regression} \
                        --gnn_type {dmpnn, gcn, gin} --log_dir checkpoints \
                        --n_epochs 50 --batch_size 50
```

Hyperparameter tuning:
```shell
python drug_gnn/hyperopt.py --data_path ${data} --task ${regression}  \
                            --gnn_type dmpnn --n_epochs 30 \
                            --hyperopt_dir hyper_dmpnn
```

## Contact

## Misc


## Others

This project is based on [chemprop](https://github.com/chemprop/chemprop), and [chiral_gnn](https://github.com/PattanaikL/chiral_gnn)

