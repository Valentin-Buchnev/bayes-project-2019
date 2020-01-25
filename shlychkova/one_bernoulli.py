import numpy as np
import scipy.stats as sps
from scipy import optimize
import scipy.special
from bayes_testing_inner import hdr, apost_beta


class OneBernoulliTesting:

    def __init__(self):
        self.sample = []
        self.sample_sum = 0
        self.sample_len = 0

    def fit(self, X):
        """
                :param X: выборка из распределения Бернулли
        """
        self.sample = X
        self.sample_sum = X.sum()
        self.sample_len = len(X)

    def aprior_prob(self, params_hyp, params_dist, hypothesis='complex vs complex', alternative='less', aprior='Beta'):
        """
                :param params_dist: параметры априорного распределения
                :param params_hyp: параметры гипотез
                :param hypothesis: тип гипотезы
                :param aprior: априорное распределение
                :return: априорные вероятности нулевой и первой гипотезы
        """

        if hypothesis == 'complex vs complex' and aprior == 'Beta':
            alpha, beta = params_dist
            dist = sps.beta(alpha, beta)
            if alternative == 'less':
                theta = params_hyp
                return 1 - dist.cdf(theta), dist.cdf(theta)
            if alternative == 'more':
                theta = params_hyp
                return dist.cdf(theta), 1 - dist.cdf(theta)
            if alternative == 'two-sided':
                theta1, theta2 = params_hyp
                return dist.cdf(theta2) - dist.cdf(theta1), 1 - dist.cdf(theta2) + dist.cdf(theta1)

        if hypothesis == 'simple vs simple' and aprior == 'Bernoulli':

            p = params_dist
            return p, 1 - p

        if hypothesis == 'H_0 modification' and aprior == 'Beta':

            alpha, beta, eps = params_dist
            dist = sps.beta(alpha, beta)

            if alternative == 'less':
                theta = params_hyp
                return dist.cdf(theta + eps) - dist.cdf(theta), dist.cdf(theta)

            if alternative == 'more':
                theta = params_hyp
                return dist.cdf(theta) - dist.cdf(theta - eps), 1 - dist.cdf(theta)

        if hypothesis == 'aprior with atom' and aprior == 'Beta':
            alpha, beta, pi0 = params_dist
            return pi0, 1-pi0

    def aposterior_prob(self, params_hyp, params_dist, hypothesis='complex vs complex', alternative='less', aprior='Beta'):
        """
                :param params_dist: параметры априорного распределения
                :param params_hyp: параметры гипотез
                :param hypothesis: тип гипотезы
                :param aprior: априорное распределение
                :return: апостериорные вероятности нулевой и первой гипотезы
        """
        if hypothesis == 'complex vs complex' and aprior == 'Beta':

            alpha, beta = params_dist
            dist = sps.beta(alpha + np.sum(self.sample), beta + len(self.sample) - np.sum(self.sample))
            if alternative == 'less':
                theta = params_hyp
                return 1 - dist.cdf(theta), dist.cdf(theta)
            if alternative == 'more':
                theta = params_hyp
                return dist.cdf(theta), 1 - dist.cdf(theta)
            if alternative == 'two-sided':
                theta1, theta2 = params_hyp
                return dist.cdf(theta2) - dist.cdf(theta1), 1 - dist.cdf(theta2) + dist.cdf(theta1)

        if hypothesis == 'simple vs simple' and aprior == 'Bernoulli':
            theta1, theta2 = params_hyp
            p = params_dist
            prob1 = theta1**(self.sample_sum) * (1 - theta1)**(self.sample_len - self.sample_sum) * p
            prob2 = theta2**(self.sample_sum) * (1 - theta2)**(self.sample_len - self.sample_sum) * (1 - p)

            return prob1/(prob1 + prob2), prob2/(prob1 + prob2)

        if hypothesis == 'H_0 modification' and aprior == 'Beta':
            alpha, beta, eps = params_dist
            dist = sps.beta(alpha + self.sample_sum, beta + self.sample_len - self.sample_sum)

            if alternative == 'less':
                theta = params_hyp
                return dist.cdf(theta + eps) - dist.cdf(theta), dist.cdf(theta)

            if alternative == 'more':
                theta = params_hyp
                return dist.cdf(theta) - dist.cdf(theta-eps),  1 - dist.cdf(theta)

        if hypothesis == 'aprior with atom' and aprior == 'Beta':
            alpha, beta, pi0 = params_dist
            pi1 = 1 - pi0
            theta = params_hyp
            p1 = 0

            if alternative == 'two-sided':

                p1 = 1 / scipy.special.beta(alpha, beta) * scipy.special.beta(
                    alpha+self.sample_sum, beta+self.sample_len-self.sample_sum)

            elif alternative == 'less':

                p1 = 1/(scipy.special.beta(alpha, beta) / scipy.special.beta(
                    alpha + self.sample_sum, beta + self.sample_len - self.sample_sum)*sps.beta(
                    alpha + self.sample_sum, beta + self.sample_len - self.sample_sum).cdf(theta)/sps.beta(
                    alpha, beta).cdf(theta))

            elif alternative == 'more':
                p1 = 1/(scipy.special.beta(alpha, beta) / scipy.special.beta(
                    alpha + self.sample_sum, beta + self.sample_len - self.sample_sum) * (1 - sps.beta(
                    alpha + self.sample_sum, beta + self.sample_len - self.sample_sum).cdf(theta))/(1 - sps.beta(
                    alpha, beta).cdf(theta)))

            p0 = theta ** self.sample_sum * (1 - theta) ** (self.sample_len - self.sample_sum)

            p = pi0 * p0 + pi1 * p1

            p1_apost = pi1 * p1 / p
            p0_apost = pi0 * p0 / p

            return p0_apost, p1_apost


    def bayes_factor(self, params_hyp, params_dist, hypothesis='complex vs complex', alternative='less', aprior='Beta'):
        """
                :param params_dist: параметры априорного распределения
                :param params_hyp: параметры гипотез
                :param hypothesis: тип гипотезы
                :param aprior: априорное распределение
                :return: баесовский фактор
        """

        p1_apost, p2_apost = self.aposterior_prob(params_hyp, params_dist, hypothesis, alternative, aprior)
        p1_apr, p2_apr = self.aprior_prob(params_hyp, params_dist, hypothesis, alternative, aprior)
        return p1_apost * p1_apr / p2_apost / p1_apr

    def hdr(self, params_dist):
        """
                :param params_dist: параметры априорного распределения
        """
        alpha, beta = params_dist
        return hdr(sps.beta(apost_beta(alpha, beta, self.sample)))

    def lindi_method(self, params_hyp, params_dist):
        """
                :param params_dist: параметры априорного распределения
                :param params_hyp: параметры гипотез
                :return: область наибольшой плотности
        """

        alpha, beta = params_dist
        theta = params_hyp
        dist = sps.beta(alpha + self.sample_sum, beta + self.sample_len - self.sample_sum)
        p = dist.pdf(theta)
        def g(y):
            return dist.pdf(y) - p
        x_r = optimize.brentq(g, 0.9, 1)
        return dist.cdf(x_r) - dist.cdf(theta) - 0.95