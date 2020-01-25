import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as sps
import seaborn as sns

class Hypotheses1UncnownSigmaUniformPrior:
# Нужно объединить с другим классом, для случая 
  def __init__(self):
    pass

  def test(self, sample, test_params):
    """
    :param sample: - выборка из нормального распределения с неизвестной
     дисперсией
    :param  test_params - словарь с параметрами гипотезы
    обязательно содержит:
    "type_of_test" - одно из "atom", "modification", "lindley", "atom",
    "leq", "geq" - тип гипотезы 
    "t0": собственно, точка, относительно которой проверяется гипотеза
    для метода с атомом: pi_0: - априорная вероятность, k: относится к дисперсии 
    априорного распределения, см. выкладки
    для метода модификации гипотезы: "epsilon" - радиус "окрестности" t0
    для метода Линдли - "conf" - уроветь зачимости
    возвращает метод также словарь, в зависимости от гипотезы может 
    содержать "HDR" в виде концов отрезка, "rejected" - для метода Линдли,
    "p0", "p1", "B" - апостериорные вероятности и байесовский фактор,
    "post_params" - ппараметры постериорного распределения, см. выкладки и примеры
    применения
    """
    t0 = test_params["t0"]
    n = len(sample)

    test_type = test_params["type_of_test"]

    assert test_type in ["modification", "lindley", "atom",
                                  "leq", "geq"], "unknown test type"

    

    if test_type == "atom":
      try:
        self.pi_0 = test_params["pi_0"]
        self.pi_1 = 1 - self.pi_0
        self.k = test_params["k"]
      except Exception:
        print("not enough params")


      v = n - 1
      t = (t0 - np.mean(sample)) * np.sqrt(n - 1) / (np.std(sample) ** 2) 

      self.p0 = (1 + t ** 2 / v) ** (-(v + 1) / 2) * self.pi_0 # почему не - n  / 2 ?
      self.p1 = 1 / np.sqrt(1 + n * self.k) * (1 + t ** 2 / (1 + n * self.k) / v) ** (-(v + 1) / 2) * self.pi_1
      normalizator = self.p0 + self.p1
      self.p0 /= normalizator
      self.p1 /= normalizator
      self.B = self.p0 / self.p1

      self.t0 = t0
      self.n = n
      return {
          "p0":self.p0,
          "p1":self.p1,
          "B": self.B,

      }

    m = np.mean(sample)
    S_sq = np.mean((sample - m) ** 2) 
    poster_distr = lambda x: 1 / ((m - x)**2 + S_sq) ** (n/2)
    
    big_num = 10**4
    step = 1000
    big_space = np.linspace(m-big_num, m+big_num, 2 * big_num * step)
    divisor = np.sum(poster_distr(big_space)) / step
    poster_distr = lambda x: 1 / ((m - x)**2 + S_sq) ** (n/2) / divisor
    pdf_with_step = poster_distr(big_space) / step
    cdf = np.cumsum(pdf_with_step)
    if test_type == "lindley":
        try:
          conf = 1 - test_params["conf"]
        except Exception:
          print("not enough params")
        l, r = big_space[np.argmin((cdf - (1 - conf)/2)**2)], big_space[np.argmin((cdf - (1 + conf)/2)**2)]
        reject = False
        if t0 < l or t0 > r:
          reject = True
        return {
            "reject":reject,
            "HDR": (l, r),
            "post_params": {
                "n": n, 
                "mean": m,
                "S_sq": S_sq
            } 
        }

    if test_type == "modification":
          try:
            epsilon = test_params["epsilon"]
          except Exception:
            print("not enough params")
          p0 = cdf[np.argmin((t0 + epsilon)**2)] - cdf[np.argmin((t0 - epsilon)**2)]

          # под HDR в данном случае подразумеваем такую симметричную окрестность  
          # среднего, что t0 в неё попадает "а границе"

          #Возможно, есть смысл сделать по другому?
          delta = np.abs(m - t0)
          l, r = (m- delta, m + delta)
          return {
              "p0": p0,
              "p1": 1 - p0,
              "HDR": (l, r),
              "post_params": {
                  "n": n, 
                  "mean": m,
                  "S_sq": S_sq
              } 
          }
    
    if test_type in ["leq", "geq"]:
      p0 = cdf[np.argmin((t0 - big_space)**2)]
      p1 = 1 - p0
      if test_type == "geq":
        p0, p1 = p1, p0
      B = p0 / p1
      # При чем тут HDR, не очень понял
      return {
          "p0": p0,
          "p1": p1, 
          "B": B
      }
