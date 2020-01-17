import numpy as np
import scipy.stats as sps


class BernoulliTesting:

    def __init__(self):
        self.sample = []

    def fit(self, X):
        self.sample = X

    def aposterior_prob(self, params_hyp, params_dist, hypothesis='complex vs complex', alternative='less', aprior='Beta'):
        """
        :param X: выборка из распределения Бернулли
        :param params_dist: параметры априорного распределения
        :param params_hyp: параметры гипотез
        :param hypothesis: тип гипотезы
        :param aprior: априорное распределение
        :return: апостериорная вероятность нулевой гипотезы
        """
        if hypothesis == 'complex vs complex' and aprior == 'Beta':

            alpha, beta = params_dist
            dist = sps.beta(alpha + np.sum(self.sample), beta + len(self.sample) - np.sum(self.sample))
            if alternative == 'less':
                theta = params_hyp
                return 1 - dist.cdf(theta), dist.cdf(theta)
            if alternative == 'less':
                theta = params_hyp
                return dist.cdf(theta), 1 - dist.cdf(theta)
            if alternative == 'two-sided':
                theta1, theta2 = params_hyp
                return dist.cdf(theta2) - dist.cdf(theta1), 1 - dist.cdf(theta2) + dist.cdf(theta1)

        if hypothesis == 'simple vs simple' and aprior == 'Bernoulli':
            theta1, theta2 = params_hyp
            p = params_dist

            prob1 = theta1**(np.sum(self.sample)) * (1 - theta1)**(len(self.sample) - np.sum(self.sample)) * p
            prob2 = theta2**(np.sum(self.sample)) * (1 - theta2)**(len(self.sample) - np.sum(self.sample)) * (1 - p)

            return prob1/(prob1 + prob2), prob2/(prob1 + prob2)

        if hypothesis == 'H_0 modification' and aprior == 'Beta':
            alpha, beta, eps = params_dist
            dist = sps.beta(alpha + np.sum(self.sample), beta + len(self.sample) - np.sum(self.sample))

            if alternative == 'less':
                theta = params_hyp
                return dist.cdf(theta + eps) - dist.cdf(theta), dist.cdf(theta)

            if alternative == 'more':
                theta = params_hyp
                return dist.cdf(theta) - dist.cdf(theta-eps),  1- dist.cdf(theta)

