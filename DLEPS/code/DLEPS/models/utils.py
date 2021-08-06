import gzip
import pandas
import h5py
import numpy as np

def one_hot_array(i, n):
    return list(map(int, [ix == i for ix in range(n)]))

def many_one_hot(indices, d):
    # (t,) - indices for n documents and t timesteps
    t = indices.shape[0]
    oh = np.zeros((t,d))
    oh[np.arange(t), indices] = 1
    return oh



def one_hot_index(vec, charset):
    return list(map(charset.index, vec))

def from_one_hot_array(vec):
    oh = np.where(vec == 1)
    if oh[0].shape == (0, ):
        return None
    return int(oh[0][0])

def decode_smiles_from_indexes(vec, charset):
    return "".join([charset[x] for x in vec]).strip()

def load_dataset(filename, split = True):
    h5f = h5py.File(filename, 'r')
    if split:
        data_train = h5f['data_train'][:]
    else:
        data_train = None
    data_test = h5f['data_test'][:]
    charset =  h5f['charset'][:]
    h5f.close()
    if split:
        return (data_train, data_test, charset)
    else:
        return (data_test, charset)
