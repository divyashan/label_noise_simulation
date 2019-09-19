import numpy as np


class MultivariateGaussian(object):
    def __init__(self, mean, cov):
        self.mean = mean
        self.cov = cov
    def sample(n=1):
        return np.random.multivariate_gaussian(self.mean, self.cov, n)
