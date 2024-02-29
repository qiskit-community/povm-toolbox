import numpy as np


# Gram-Schmidt
def gs(X: np.ndarray) -> np.ndarray:
    """Returns the orthonormal basis resulting from Gram-Schmidt process of X"""

    Q, _ = np.linalg.qr(X)
    return Q


def n_sphere(param: np.ndarray) -> np.ndarray:
    """Returns a unit vector on the n-sphere"""

    n = len(param)
    x = np.ones(n + 1)
    for i in range(n - 1):
        x[i] *= np.cos(np.pi * param[i])
        x[i + 1:] *= np.sin(np.pi * param[i])
    x[-2] *= np.cos(2 * np.pi * param[-1])
    x[-1] *= np.sin(2 * np.pi * param[-1])

    return x


def povms_union():
    return None
