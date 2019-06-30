import numpy as np
import scipy.spatial.distance as ds

def replace_nan(X):
    """ replace nan and inf to 0
    """
    X[np.isnan(X)] = 0
    X[np.isinf(X)] = 0
    return X


def compute_label_sim(sig_y1, sig_y2, sim_scale):
    """ compute class label similarity
    """
    dist = ds.cdist(sig_y1, sig_y2, 'euclidean')
    dist = dist.astype(np.float32)
    Sim = np.exp(-np.square(dist) * sim_scale);
    s = np.sum(Sim, axis=1)
    Sim = replace_nan(Sim / s[:, None])
    return Sim
