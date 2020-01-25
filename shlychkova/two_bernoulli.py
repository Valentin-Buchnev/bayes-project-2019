import numpy as np
import scipy.stats as sps
from scipy import optimize
from bayes_testing_inner import hdr, person_coeff, arth


class TwoBernoulliTesting:

    def __init__(self):
        self.second_samplefirst_sample = []
        self.second_sample = []

    def fit(self, X, Y):
        r"""
                :param X: выборка из распределения Бернулли
                :param Y: выборка из распределения Бернулли
        """
        self.first_sample = X
        self.second_sample = Y

    def dispersive_analysis(self, params_dist, alpha=0.05, hypothesis='simple vs complex', alternative='two-sided', aprior='Beta'):
        """

                :param params_dist:
                    [alpha_1, beta_1, alpha_2, beta_2] - параметры априорного распределения
                :param alpha: alpha для доверительного интервала
                :param hypothesis: тип гипотезы
                :param alternative: тип альтернативы
                :param aprior: априорное растпределние
                :return: { 'HDR': alpha% интервал, 'aposterior params': параметры распределения статистики}
        """

        if aprior == 'Beta' and alternative == 'two-sided' and hypothesis =='simple vs complex':
            # параметры априорного распределения
            alpha1, beta1, alpha2, beta2 = params_dist

            # параметры апостериорного распределения
            sum_positive = np.sum(self.first_sample)
            alpha1_apost, beta1_apost = alpha1 + sum_positive, beta1 + len(self.first_sample) - sum_positive
            sum_positive = np.sum(self.second_sample)
            alpha2_apost, beta2_apost = alpha2 + sum_positive, beta2 + len(self.second_sample) - sum_positive

            # параметры итогового арспределения
            a = np.log((alpha1_apost - 0.5)*(beta2_apost-0.5)/(alpha2_apost - 0.5)/(beta1_apost-0.5))
            sigma = 1/alpha1_apost + 1/beta1_apost + 1/alpha2_apost + 1/beta2_apost

            return {'HDR': hdr(sps.norm(a, sigma), alpha), 'aposterior params': [a, sigma]}

    def correlation_analysis(self, params_dist=0, alpha=0.05):
        """
                :param params_dist: c - степень в апрорном распредлении
                :param alpha: alpha для доверительного интервала
                :return: { 'HDR': alpha% интервал, 'aposterior params': параметры распределения статистики}
        """
        c = params_dist
        n = len(self.first_sample)
        theta = person_coeff(self.first_sample, self.second_sample)
        return {'HDR': hdr(sps.norm(arth(theta), 1/n), alpha), 'aposterior params': [arth(theta), 1/n]}





