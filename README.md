# DrugAI
Drug discovery using Graph Neural Network

## What

This repo implements [DMPNN](https://pubs.acs.org/doi/10.1021/acs.jcim.9b00237) and more using `Pytorch Geometric`, including

- [DMPNN](https://pubs.acs.org/doi/10.1021/acs.jcim.9b00237):
- [GCN](https://arxiv.org/abs/1609.02907)
- [GINE](https://arxiv.org/abs/1905.12265>)


## Dependency

- python >= 3.7
- Pytorch >= 1.5
- Pytorch Geometric >= 1.7
- RDkit


## Usage

Train
```shell
python drug_gnn train.py --data_path ${data} --task ${regression} \
                         --gnn_type {dmpnn, gcn, gin} --log_dir checkpoints \
                         --n_epochs 50 --batch_size 50
```

Eval
```shell
python drug_gnn eval.py --data_path ${data} --task ${regression} \
                         --gnn_type {dmpnn, gcn, gin} --log_dir checkpoints \
                         --n_epochs 50 --batch_size 50
```
## Contact

## Misc


## Others

This project is based on [chemprop](https://github.com/chemprop/chemprop), and [chiral_gnn](https://github.com/PattanaikL/chiral_gnn)

