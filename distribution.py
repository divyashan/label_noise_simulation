import numpy as np
import scipy.stats


class MultivariateGaussian(object):
    def __init__(self, mean, cov):
        self.mean = mean
        self.cov = cov
        self.dist = scipy.stats.multivariate_normal(self.mean, self.cov)
    def sample(self, n=1):
        return self.dist.rvs(size=n)
    def pdf(self, x):
        return self.dist.pdf(x)
    def info(self):
        return "mv_gaussian_mean_{}_cov_{}".format(self.mean, self.cov)

class Exponential(object):
    def __init__(self, scale, mean=0):
        self.scale = scale
        self.mean = mean
    def sample(self, n=2):
        return np.random.exponential(self.scale, n) + self.mean
    def info(self):
        return "exponential_mean_{}_scale_{}".format(self.mean, self.scale)
    def pdf(self, x):
        return scipy.stats.multivariate_normal.pdf(x, self.mean, self.scale)

class Uniform(object):
    def __init__(self, low, high):
        self.low = low
        self.high = high
    def sample(self, n=2):
        return np.random.uniform(self.low, self.high, n)
    def info(self):
        return "uniform_low_{}_high_{}".format(self.low, self.high)
    def pdf(self, x):
        return scipy.stats.multivariate_normal.pdf(x, self.mean, self.cov)

class Geometric(object):
    def __init__(self, p):
        self.p = p
    def sample(self, n=2):
        return np.random.geometric(self.p, n)
    def info(self):
        return "geometric_p_{}".format(self.p)
    def pdf(self, x):
        return scipy.stats.multivariate_normal.pdf(x, self.mean, self.cov)


class Blob(object):
    def __init__(self):
        self.name = "blob"
    def sample(self, n=2):
        raise NotImplementedError
    def info(self):
        return self.name
    def pdf(self, x):
        return scipy.stats.multivariate_normal.pdf(x, self.mean, self.cov)


