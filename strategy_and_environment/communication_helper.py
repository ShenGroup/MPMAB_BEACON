import numpy as np
from scipy.optimize import linear_sum_assignment
import itertools


def truncate(x, Q):
    """This function truncates x in [0,1] to Q bits."""
    bit_representation = d2b(x, Q)
    y = 0
    for q in range(Q):
        if bit_representation[q] == '1':
            y += 2 ** (-q - 1)
    return y


def d2b(decimal, Q):
    """
    This function returns the bit sequence of decimal which is truncated by Q bits
    decimal: [0,1],
    Q: sequence length
    """
    i = 0
    if decimal == 1:
        return str(1).rjust(Q, '1')
    decimal_convert = ""
    while decimal != 0 and i < Q:
        result = int(decimal * 2)
        decimal = decimal * 2 - result
        decimal_convert = decimal_convert + str(result)
        i = i + 1
    while i < Q:
        decimal_convert = decimal_convert + "0"
        i = i + 1
    return decimal_convert


def Oracle(mean, type='linear'):
    """This function returns an optimal matching of the combinatorial optimization problem."""
    if type == 'linear' or type == 'proportional fairness':
        temp = -mean
        rind, cind = linear_sum_assignment(temp)
        return cind
    elif type == 'max_min fairness':
        M, K = mean.shape
        L = list(range(K))
        all_permutations = list(itertools.permutations(L, M))
        tmp = []
        optimal_value = -333
        for i in range(len(all_permutations)):
            if np.prod(mean[list(range(M)), all_permutations[i]]) > optimal_value:
                optimal_value = np.prod(mean[list(range(M)), all_permutations[i]])
                tmp = all_permutations[i]
        return tmp
