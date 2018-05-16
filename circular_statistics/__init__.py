from __future__ import division
import numpy as np
from scipy.stats import chi2, circmean, binom
import warnings

"""
All functions are imitations from the Circular Statistics Toolbox for Matlab by Philipp Berens, 2009 
(berens@tuebingen.mpg.de - www.kyb.mpg.de/~berens/circStat.html).
"""

def circ_r(phases):
    """
    Computes mean resultant vector length for circular data.

    :param phases: Sample of angles in radians.
    :return Mean resultant vector length.
    """
    return np.abs(np.sum(np.exp(1j*phases), 0)) / np.size(phases, 0)


def circ_dist(x, y):
    """
    Pairwise difference x_i-y_i around the circle computed efficiently.

    :param x: Sample of linear random variable.
    :param y: Sample of linear random variable or one single angle.
    :return: Array with pairwise differences.
    :rtype:

    References:
    Biostatistical Analysis, J. H. Zar, p. 651
    """
    return np.angle(np.exp(1j*x) / np.exp(1j*y))


def circ_dist2(x, y=None):
    """
    All pairwise difference x_i-y_j around the circle computed efficiently.

    :param x: Sample of linear random variable.
    :param y: Sample of linear random variable or None for auto-distances.
    :return: Matrix with pairwise differences (M_ij = dist(x_i, y_j)).

    References:
    Biostatistical Analysis, J. H. Zar, p. 651
    """
    if y is None:
        y = x
    x = np.array([x]).T

    return np.angle(np.tile(np.exp(1j*x), (1, len(y))) / np.tile(np.exp(1j*y), (len(x), 1)))


def circ_median(phases):
    """
    Computes the median for circular data.

    :param phases: Sample of angles in radians
    :type phases: array-like
    :return: md: Median.
    :rtype: float

    References:
    Biostatistical Analysis, J. H. Zar (26.6)
    """

    phases = np.mod(phases, 2 * np.pi)

    dist_mat = circ_dist2(phases, phases)
    n_pos_dist = np.sum(dist_mat >= 0, 0)
    n_neg_dist = np.sum(dist_mat <= 0, 0)

    diff_pos_neg_dists = np.abs(n_pos_dist-n_neg_dist)
    m = np.min(diff_pos_neg_dists)  # find the one who has the same number of pos. and neg. distances
    # (i.e. lies in the middle)
    idx_med = np.where(diff_pos_neg_dists == m)[0]

    if m > 1:
        warnings.warn('Ties detected.')

    md = circmean(phases[idx_med])  # in case of an even number of phases take the mean between the minima

    if abs(circ_dist(circmean(phases), md)) > abs(circ_dist(circmean(phases), md + np.pi)):
        md = np.mod(md+np.pi, 2*np.pi)
    return md


def circ_medtest(phases, md):
    """
    Tests for significance of the median.
    H0: The population has median angle md.

    References:
    Fisher NI, 1995

    :param phases: Sample of angles in radians.
    :param md: Median.
    :return pval  p-value
    """

    if np.size(md) > 1:
        raise ValueError('Median can only be a single value.')

    n = np.size(phases)

    # compute deviations from median
    d = np.angle(np.exp(1j * phases) / np.exp(1j * md))

    n1 = np.sum(d < 0)
    n2 = np.sum(d > 0)

    # compute p-value with binomial test
    x = np.concatenate([np.arange(0, np.min([n1, n2]) + 1), np.arange(np.max([n1, n2]), n + 1)])
    pval = np.sum(binom.pmf(x, n, 0.5))

    return pval


def circ_cmtest(phase_samples):
    """
    Non parametric multi-sample test for equal medians. Similar to a Kruskal-Wallis test for linear data.
    H0: The populations have equal medians.

    References:
    Fisher NI, 1995

    :param phase_samples: List of samples, each containing phases in radians.
    :return pval: p-value (discard H0 if p-value is small);
            med: best estimate of the shared population median if H0 is not discarded at the 0.05 level, nan otherwise;
            P: test statistic of the common median test.
    """
    phases_all = [phase for sample in phase_samples for phase in sample]
    n_samples = len(phase_samples)
    n_phases_all = len(phases_all)
    med_all = circ_median(phases_all)

    n_phases = np.zeros((n_samples, 1))
    m = np.zeros((n_samples, 1))
    for sample_idx in range(n_samples):
        n_phases[sample_idx] = len(phase_samples[sample_idx])

        d = circ_dist(phase_samples[sample_idx], med_all)

        m[sample_idx] = np.sum(d < 0)

    if np.any(n_phases < 10):
        warnings.warn('Test not applicable. Sample size in at least one sample is too small.')

    M = np.sum(m)
    P = (n_phases_all**2 / (M * (n_phases_all - M))) * np.sum(m**2. / n_phases) - n_phases_all * M / (n_phases_all - M)

    pval = 1 - chi2.cdf(P, n_samples - 1)

    if pval < 0.05:
        med_all = np.nan

    return pval, med_all, P


if __name__ == '__main__':
    alpha1 = np.random.uniform(0, np.pi, 50)
    alpha2 = np.random.uniform(np.pi, 2 * np.pi, 50)
    alphas = [alpha1, alpha2]

    pval, med, P = circ_cmtest(alphas)
    if pval < 0.05:
        print 'H0 (Populations have equal medians) rejected'
    else:
        print 'H0 (Populations have equal medians) accepted'
        print 'Median: %.2f' % med

    alpha1 = np.random.uniform(0, np.pi, 50)
    alpha2 = np.random.uniform(0, 2 * np.pi, 50)
    alphas = [alpha1, alpha2]

    pval, med, P = circ_cmtest(alphas)
    if pval < 0.05:
        print 'H0 (Populations have equal medians) rejected'
    else:
        print 'H0 (Populations have equal medians) accepted'
        print 'Median: %.2f' % med