import math

rad = 0.85
doc = 0.8
stu = 0.6

# a

# 0.512
three_doc = doc ** 3

# 0.384
two_doc = (3 * doc * doc * (1-doc))

# 0.896
at_least_two_doc = two_doc + three_doc

doc_correct = at_least_two_doc


# b
def binom(n, k, p):
    return math.comb(n, k) * (p ** k) * ((1-p) ** (n-k))


def binom_gt(n, k, p):
    prob = 0
    for i in range(k, n+1):
        prob += binom(n, i, p)
    return prob


# we assume ties are solved with a random coinflip
def majority_correct(n, p):
    if n % 2 == 0:
        return (
            binom(n, n // 2, p) * 0.5
            + binom_gt(n, n // 2 + 1, p)
        )
    else:
        return binom_gt(n, n // 2 + 1, p)


# 0.814
stu_correct = majority_correct(19, 0.6)

# TODO: c

# d


def min_size_group(desired_prob, p):
    i = 1
    prob = p
    while prob <= desired_prob:
        i += 1
        prob = majority_correct(i, p)
    return i


# 39
min_size_students = min_size_group(doc_correct, stu)
