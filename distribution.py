import numpy as np


class MultivariateGaussian(object):
    def __init__(self, mean, cov):
        self.mean = mean
        self.cov = cov
    def sample(self, n=1):
        return np.random.multivariate_normal(self.mean, self.cov, n)
    def info(self):
        return "mv_gaussian_mean_{}_cov_{}".format(self.mean, self.cov)

class Exponential(object):
    def __init__(self, scale):
        self.scale = scale
    def sample(self, n=1):
        return np.random.exponential(self.scale, n)
    def info(self):
        return "exponential_scale_{}".format(self.scale)

class Uniform(object):
    def __init__(self, low, high):
        self.low = low
        self.high = high
    def sample(self, n=1):
        return np.random.uniform(self.low, self.high, n)
    def info(self):
        return "uniform_low_{}_high_{}".format(self.low, self.high)

class Geometric(object):
    def __init__(self, p):
        self.p = p
    def sample(self, n=1):
        return np.random.geometric(self.p, n)
    def info(self):
        return "geometric_p_{}".format(self.p)


