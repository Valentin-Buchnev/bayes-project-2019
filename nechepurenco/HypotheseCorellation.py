import numpy as np
import scipy.stats as sps

arth = lambda x: 0.5 * np.log(1 + x) / np.log(1 - x)

class HypotheseCorellation:
# Нужно объединить с другим классом, для случая 
  def __init__(self):
    pass

  def test(self, X, Y, test_params):
    """
    :param X: - первая выборка
    :param Y: - вторая выборка
    :param  test_params - словарь с параметрами гипотезы
    обязательно содержит:
    "type_of_test" - одно из "modification", "lindley", "leq", "geq" - тип гипотезы 
    "t0": собственно, точка, относительно которой проверяется гипотеза
    для метода модификации гипотезы: "epsilon" - радиус "окрестности" t0
    для метода Линдли - "conf" - уроветь зачимости
    возвращает метод также словарь, в зависимости от гипотезы может 
    содержать "HDR" в виде концов отрезка - важно: именно для
    arth theta, а не theta (за исключением метода модификации критерия), "rejected" - для метода Линдли,
    "p0", "p1", "B" - апостериорные вероятности и байесовский фактор,
    "post_params" - ппараметры постериорного распределения, см. выкладки и примеры
    применения
    """
    t0 = test_params["t0"]
    n = len(X)
    test_type = test_params["type_of_test"]

    assert test_type in ["modification", "lindley", "leq", "geq"], "unknown test type"

    corr_coef = np.corrcoef(X, Y)[1][0]
    t0_arth = arth(t0)
    distr = sps.norm(arth(corr_coef), np.sqrt(1/n)) # апостериорное распредение корелляции
     
    if test_type == "lindley":
        try:
          conf = 1 - test_params["conf"]
        except Exception:
          print("not enough params")
        l, r = distr.ppf((1 - conf)/2), distr.ppf((1 + conf)/2)
        reject = False
        if t0_arth < l or t0_arth > r:
          reject = True
        return {
            "reject":reject,
            "HDR": (l, r),
            "post_params": {
                "mean": arth(corr_coef),
                "sigma^2": 1/n
            } 
        }

    if test_type == "modification":
          try:
            epsilon = test_params["epsilon"]
          except Exception:
            print("not enough params")
          p0 = distr.cdf(arth(t0 + epsilon)) - distr.cdf(arth(t0 - epsilon))

          # под HDR в данном случае подразумеваем такую симметричную окрестность  
          # среднего, что t0 в неё попадает "а границе"

          #Возможно, есть смысл сделать по другому?
          delta = np.abs(corr_coef - t0)
          l, r = (corr_coef- delta, corr_coef + delta)
          return {
              "p0": p0,
              "p1": 1 - p0,
              "HDR": (l, r),
              "post_params": {
                  "mean": arth(corr_coef),
                  "sigma^2": 1/n
              } 
          }
    
    if test_type in ["leq", "geq"]:
      p0 = distr.cdf(t0_arth)
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
