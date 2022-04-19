import numpy as np
import scipy.io


def load_avg(datapath, key, num=100):
    data = scipy.io.loadmat(datapath)[key]
    data_mid = data[0, :, 1500:2500]
    if 1000 % num == 0:
        K = 1000 // num
    else:
        raise ValueError('num must be a factor of 1100')
    
    data_avg = np.mean(data_mid.reshape(data_mid.shape[0], K, num), axis=-1)
    return data_avg