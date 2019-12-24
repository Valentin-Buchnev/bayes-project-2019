import numpy as np
import scipy.stats as sps


def aposterior_dist(X, params, aprior='Beta'):
    if aprior == 'Beta':
        alpha, beta = params
        return sps.beta(alpha + np.sum(X), beta + len(X) - np.sum(X))


def aposterior_prob(X, params_hyp, params_dist, hypothesis='two-sided complex', aprior='Beta'):
    """
    :param X: выборка из распределения Бернулли
    :param params_dist: параметры априорного распределения
    :param params_hyp: параметры гипотез
    :param hypothesis: тип гипотезы
    :param aprior: априорное распределение
    :return: апостериорная вероятность нулевой гипотезы
    """
    if hypothesis == 'two-sided complex' and aprior=='Beta':
        theta = params_hyp
        alpha, beta = params_dist
        dist = aposterior_dist(X, (alpha, beta))
        return dist.cdf(theta)
