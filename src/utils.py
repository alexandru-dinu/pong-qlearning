import numpy as np


def random_choice(lst):
    r = np.random.randint(len(lst))
    return lst[r]
