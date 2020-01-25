import numpy as np
import scipy.stats as sps


def apost_beta(alpha, beta, sample):
    return alpha + np.sum(sample), beta + len(sample) - np.sum(sample)


def hdr(dist, alpha=0.05):
    return dist.ppf(alpha/2), dist.ppf(1-alpha/2)

def person_coeff(X, Y):
    return ((X - X.mean())*(Y - Y.mean())).sum()/(((X - X.mean())**2).sum()*((Y - Y.mean())**2).sum())**0.5

def arth(x):
    return 0.5 * np.log((1+x)/(1-x))