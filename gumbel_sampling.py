import argparse
import random
import math
import sys
from matplotlib import pyplot
import numpy


class Distribution(object):
    def gen(self):
        return 0

    def likeli(self, x):
        return 1 if x == 0 else 0

    def loglikeli(self, x):
        return 0


class Gaussian(Distribution):
    def gen(self):
        return random.gauss(0, 1)

    def likeli(self, x):
        return numpy.exp(- x * x / 2) / numpy.sqrt(2 * math.pi)

    def loglikeli(self, x):
        return - numpy.log(2 * math.pi) / 2 - x ** 2 / 2


class Gumbel(Distribution):
    def gen(self):
        assert False, "not implemented"
        return 0

    def likeli(self, x):
        return numpy.exp(x - numpy.exp(x))

    def loglikeli(self, x):
        return x - numpy.exp(x)


def sample_is_naive(n, target, p, q=Gaussian()):
    """
    using important sampling to compute the  expectation of the target function

    :param n: number of samples
    :param target: target function to compute
    :param p: the target density function
    :param q: the proposal density function
    :return: Expectation[target(x)]_p
    """
    s = 0
    m = 0
    samples = []
    weights = []
    for i in xrange(n):
        x = q.gen()
        qx = q.likeli(x)
        if qx > 0:
            w = p.likeli(x) / qx
            s += target(x) * w
            m += 1
            samples.append(x)
            weights.append(w)
    s /= m
    return s, numpy.array(samples), numpy.array(weights)


def sample_is(n, target, p, q=Gaussian()):
    """
    using important sampling to compute the  expectation of the target function

    :param n: number of samples
    :param target: target function to compute
    :param p: the target density function
    :param q: the proposal density function
    :return: Expectation[target(x)]_p
    """
    samples = numpy.array([q.gen() for _ in xrange(n)])
    weights = numpy.exp(p.loglikeli(samples) - q.loglikeli(samples))
    s = numpy.average(weights * target(samples))
    return s, samples, weights


def sample_mh(n, target, p, q=Gaussian(), using_random_walk=False):
    """
    using Metropolis-Hastings algorithm to sample

    :param n: number of samples
    :param target: target function to compute
    :param p: the target density function
    :param q: the proposal density function
    :param using_random_walk: if using random-walk style MH, x' = x + q(.)
    :return: Expectation[target(x)]_p
    """
    s = 0
    x = random.random()
    samples = []
    for i in xrange(n):
        x2y = q.gen()
        y = (x + x2y) if using_random_walk else x2y
        y2x = (- x2y) if using_random_walk else x
        accept_ratio = math.exp(p.loglikeli(y) + q.loglikeli(y2x)
                                - p.loglikeli(x) - q.loglikeli(x2y))
        r = random.random()
        if r <= accept_ratio:
            x = y
        s += target(x)
        samples.append(x)
    s /= n
    return s, numpy.array(samples)


def plot_samples(f, samples, weights=None):
    """
    plot histogram of weighted samples and sample path
    :param f: the target distribution to compare with
    :param samples:
    :param weights:
    :return:
    """
    fig = pyplot.figure()
    grid = pyplot.GridSpec(1, 5)
    main_ax = fig.add_subplot(grid[0, 1:])
    if weights is None:
        main_ax.plot(samples, '.', markersize=0.2)
    else:
        maxw = max(weights)
        main_ax.scatter(range(len(samples)), samples,
                        s=numpy.maximum((weights / maxw) ** 2, 0.01))
    pyplot.xlim(0, len(samples))
    hist_ax = fig.add_subplot(grid[0, 0], sharey=main_ax)
    mn = min(samples)
    mx = max(samples)
    delta = mx - mn
    mn -= delta * 0.05
    mx += delta * 0.05
    tt = numpy.linspace(mn, mx, 100)
    hist_ax.hist(samples, 50, weights=weights, density=True,
                 orientation='horizontal', lw=0.5, edgecolor='b')
    hist_ax.plot(f(tt), tt)
    hist_ax.invert_xaxis()
    pyplot.draw()


def main(argv):
    parser = argparse.ArgumentParser(description='Monte Carlo Sampling Demo for '
                                                 'Gumbel distribution')
    parser.add_argument('num', metavar='N', type=int,
                        help='number of samples')
    option = parser.parse_args(argv)
    n = option.num
    normal_dist = Gaussian()
    gumbel_dist = Gumbel()
    meanx, samples, weights = sample_is_naive(n, lambda x: x, gumbel_dist,
                                              normal_dist)
    squarex = numpy.average(samples ** 2 * weights)
    print("=== Importance sampling ===")
    print("E[X] = {}".format(meanx))
    print("E[X^2] = {}".format(squarex))
    meanx1, samples1, weights1 = sample_is(n, lambda x: x, gumbel_dist, normal_dist)
    squarex1 = numpy.average(samples1 ** 2 * weights1)
    plot_samples(gumbel_dist.likeli, samples1, weights1)
    print("Using log to calculate")
    print("E[X] = {}".format(meanx1))
    print("E[X^2] = {}".format(squarex1))
    meanx2, samples2 = sample_mh(n, lambda x: x, gumbel_dist, normal_dist)
    squarex2 = numpy.average(samples2 ** 2)
    plot_samples(gumbel_dist.likeli, samples2)
    print("=== Metropolis Hastings sampling ===")
    print("E[X] = {}".format(meanx2))
    print("E[X^2] = {}".format(squarex2))
    meanx3, samples3 = sample_mh(n, lambda x: x, gumbel_dist, normal_dist,
                                 using_random_walk=True)
    squarex3 = numpy.average(samples3 ** 2)
    plot_samples(gumbel_dist.likeli, samples3)
    print("=== Metropolis Hastings sampling using random walk ===")
    print("E[X] = {}".format(meanx3))
    print("E[X^2] = {}".format(squarex3))
    pyplot.show()


if __name__ == '__main__':
    main(sys.argv[1:])
