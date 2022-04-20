import math

import pandas as pd
import numpy as np
import seaborn as sns

sns.set_theme()

rad = 0.85
doc = 0.8
stu = 0.6

# a

three_doc = doc ** 3
print(f"2a: Three doctors correct = {three_doc:.3f}")

two_doc = (3 * doc * doc * (1-doc))

at_least_two_doc = two_doc + three_doc
print(f"2a: At least 2 doctors correct = {at_least_two_doc:.3f}")


# b
def binom(n, k, p):
    """
    Calculate P(X=k) for binomial distribution B(n, p)
    """
    return math.comb(n, k) * (p ** k) * ((1-p) ** (n-k))


def binom_ge(n, k, p):
    """
    Calaculate P(X>=k) for binomial distribution B(n, p)
    """
    prob = 0
    for i in range(k, n+1):
        prob += binom(n, i, p)
    return prob


def majority_correct(n, p):
    """
    Calculate probability that the majority has the correct answer for
    B(n, p).

    If n is even we assume ties are split with a random
    coinflip.
    """
    # n is even
    if n % 2 == 0:
        return (
            binom(n, n // 2, p) * 0.5
            + binom_ge(n, n // 2 + 1, p)
        )
    # n is odd
    else:
        return binom_ge(n, n // 2 + 1, p)


stu_correct = majority_correct(19, 0.6)
print(f"2b: Majority students correct = {stu_correct:.3f}")

# c
df = pd.DataFrame(
    {
        'jury': jury_size,
        'competence': competence,
        'probability': majority_correct(jury_size, competence)
    }
    for competence in np.arange(0.55, 1.05, 0.1)
    for jury_size in np.arange(1, 22, 2)
)

g = sns.relplot(
    data=df, x="jury", y="probability", hue="competence", kind="line"
)

g.savefig("2c.pdf")


# d
def min_size_jury(desired_prob, p):
    """
    The minimum size of a jury with competency p to have a desired
    probability.
    """
    i = 1
    prob = p
    while prob <= desired_prob:
        i += 2
        prob = majority_correct(i, p)
    return i


min_size_students = min_size_jury(at_least_two_doc, stu)
print(f"2d: Minimum student jury size = {min_size_students}")
print("2d: Probability student jury is correct = "
      + f"{majority_correct(min_size_students, stu):.3f}")
