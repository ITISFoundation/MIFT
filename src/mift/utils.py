import numpy as np
from typing import Tuple

try:
    import jax.numpy as jnp
except:
    print(
        "JAX is not available - GPU might not be available. This will make computation slower."
    )


def compute_svd_jax_numpy(AA: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """compute SVD using jax numpy if available, otherwise use numpy"""
    try:
        u, s, vh = jnp.linalg.svd(AA)
        u = np.float64(u)
        assert (
            np.sum(np.isnan(u)) == 0
        ), "JAX produced NaN output. Defaulting to numpy SVD"
        s = np.float64(s)
        vh = np.float64(vh)
    except:
        u, s, vh = np.linalg.svd(AA)

    return u, s, vh


def regularize(s, threshold=1e-1):
    t = np.where(s < threshold)
    if len(t[0]):
        sval_nums = t[0][0]
    else:
        sval_nums = len(s)

    return sval_nums


def check_regularization(max_reg_iterations, AA, bb, u, s, vh, sval_nums):
    cnt = 0
    while cnt < max_reg_iterations:  # to limit number of iterations
        cnt += 1
        sval_nums = int(sval_nums)
        coef = np.matmul(
            vh[0:sval_nums, :].transpose(),
            np.matmul(
                np.diag(1 / s[0:sval_nums]),
                np.matmul(u[:, 0:sval_nums].transpose(), bb),
            ),
        )
        if np.max(np.abs(np.matmul(AA, coef) - bb)) < 3e-3:
            break
        else:
            sval_nums *= 1.02

    return coef


def check_regularization_3D(max_reg_iterations, AA, bb, u, s, vh, sval_nums):
    cnt = 0
    while cnt < max_reg_iterations:
        cnt += 1
        sval_nums = int(sval_nums)
        coef = np.matmul(
            vh[0:sval_nums, :].transpose(),
            np.matmul(
                np.diag(1 / s[0:sval_nums]),
                np.matmul(u[:, 0:sval_nums].transpose(), bb),
            ),
        )
        if (np.max(np.abs(np.matmul(AA, coef) - bb)) < 3e-3) or (
            (
                np.mean(np.abs(np.matmul(AA, coef) - bb)) < 0.1
            )  ## only diff wrt normal check_regularization
        ):
            break
        else:
            sval_nums *= 1.02

    return coef
