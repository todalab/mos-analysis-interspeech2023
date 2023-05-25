import numpy as np
from scipy.optimize import newton
from scipy.stats import norm, t


class Hoeffding:
    def __init__(self, error_rate):
        self.error_rate = error_rate

    def p(self, n, interval):
        return np.exp(-2 * n * interval ** 2)

    def confidence_interval(self, n):
        return np.sqrt(np.log(2.0/self.error_rate) / (2 * n))

    def n_samples(self, interval):
        return np.log(2.0/self.error_rate) / (2.0 * np.square(interval))


class BernoulliChernoff:
    def __init__(self, error_rate):
        self.error_rate = error_rate

    def p(self, n, x, mu):
        return np.exp(-n * self.kl(x, mu))

    def kl(self, p, q):
        p = np.clip(p, 1e-7, 1 - 1e-7)
        q = np.clip(q, 1e-7, 1 - 1e-7)
        return p * np.log(p / q) + (1.0 - p) * np.log((1.0 - p) / (1.0 - q))

    def confidence_interval(self, n, mu):
        a = - np.log(2.0 / self.error_rate)
        def shift(delta):
            if mu - delta > 0:
                return mu - delta
            else:
                return mu + delta
        f = lambda delta: n * self.kl(shift(delta), mu) + a
        delta = newton(f, 0.05)
        return delta

    def n_samples(self, interval, mu):
        return np.log(2.0/self.error_rate) / self.kl(mu - interval, mu)


class BernoulliExactAsymptotics:
    def __init__(self, error_rate):
        self.error_rate = error_rate

    def p(self, n, x, mu):
        a = np.sqrt((1 - x) / (2 * np.pi * x * n))
        b = mu / (mu - x)
        c = np.exp(-n * self.kl(x, mu))
        return a * b * c

    def kl(self, p, q):
        return p * np.log(p / q) + (1.0 - p) * np.log((1.0 - p) / (1.0 - q))

    def confidence_interval(self, n, mu):
        a = -np.log(2.0 / self.error_rate)
        def f(delta):
            x = mu - delta
            b = -0.5 * np.log(1 - x)
            c = 0.5 * np.log(2 * np.pi * x * n)
            d = -np.log(mu) + np.log(mu - x)
            e = n * self.kl(x, mu)
            return a + b + c + d + e
        delta = newton(f, 0.05)
        return delta

    def n_samples(self, interval, mu):
        x = mu - interval
        kl = self.kl(x, mu)
        a = -np.log(2.0/self.error_rate)
        b = -0.5 * np.log(1 - x)
        c = - np.log(mu / (mu - x))
        d = 0.5 * np.log(2 * np.pi * x)
        f = lambda n: a + b + c + + d + 0.5 * np.log(n) + n * kl
        n = newton(f, 1000)
        return n


class CentralLimitTheorem:
    def __init__(self, error_rate):
        self.error_rate = error_rate

    def p(self, n, sample_mean, mean, stddev):
        y = np.sqrt(n) * (sample_mean - mean) / stddev
        return norm.cdf(y)

    def confidence_interval(self, n, stddev):
        s = stddev / np.sqrt(n)
        delta = norm.ppf(1.0 - self.error_rate / 2) * s
        return delta

    def n_samples(self, interval, stddev):
        a = stddev / interval
        b = norm.ppf(1.0 - self.error_rate / 2)
        n = np.square(a * b)
        return n


class CentralLimitTheoremT:
    def __init__(self, error_rate):
        self.error_rate = error_rate

    def p(self, n, sample_mean, mean, stddev):
        y = np.sqrt(n) * (sample_mean - mean) / stddev
        return t.cdf(y, n-1)

    def confidence_interval(self, n, stddev):
        s = stddev / np.sqrt(n)
        delta = t.ppf(1.0 - self.error_rate / 2, n-1) * s
        return delta

    def n_samples(self, interval, stddev):
        assert False, "Do not use this method. Use confidence_interval instead."
        st = CentralLimitTheorem(self.error_rate)
        ng = st.n_samples(interval, stddev)
        a = stddev / interval
        b = t.ppf(1.0 - self.error_rate / 2, int(ng)-1)
        n = np.square(a * b)
        return n
