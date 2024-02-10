# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
ProMises algorithm
~~~~~~~~~~~~~~~~~~
A version of the Procrustes-von Mises-Fisher (ProMises) algorithm for
functional alignment. Our modification can't really make any
promises, so we call it the empty ProMises algorithm.
"""
from typing import Optional, Tuple

import jax.numpy as jnp
from jax.experimental import sparse

from hypercoil.engine import Tensor


def empty_promises(
    X: Tensor,
    M: Tensor,
    spatial_prior_loc: Tensor,
    spatial_prior_data: Tensor,
    spatial_prior_weight: float,
    new_M: Optional[Tensor] = None,
    update_weight: int = 0,
) -> Tuple[Tensor, Tuple[Tensor, float]]:
    """
    A single step of the empty ProMises algorithm.

    The spatial prior loc and data are the index and weight of a sparse
    matrix in ELL format. The matrix is implicitly symmetrised when the
    required matrix-vector products are computed.

    Details of the original ProMises algorithm can be found in the paper:
    https://onlinelibrary.wiley.com/doi/full/10.1002/hbm.26170
    """
    X, M = X.T, M.T
    _, _, Q = jnp.linalg.svd(X, full_matrices=False)
    _, _, P = jnp.linalg.svd(M, full_matrices=False)
    Z = X @ Q.T
    N = M @ P.T
    spatial_prior_data = jnp.broadcast_to(
        spatial_prior_data,
        spatial_prior_loc.shape,
    )
    spatial_prior_data = jnp.where(
        spatial_prior_loc == -1,
        0.,
        spatial_prior_data,
    )
    spatial_prior_loc = jnp.where(
        spatial_prior_loc == -1,
        0,
        spatial_prior_loc,
    )[..., None]
    spatial_prior = sparse.BCOO(
        (
            spatial_prior_data,
            spatial_prior_loc,
        ),
        shape=2 * (spatial_prior_loc.shape[0],),
    )
    spatial_prior = (
        Q @ spatial_prior @ P.T + (P @ spatial_prior @ Q.T).T
    ) / 2
    U, _, V = jnp.linalg.svd(
        N.T @ Z + spatial_prior_weight * spatial_prior,
    )
    R = U @ V
    Z = Z @ R
    X = Z @ P
    new_update_weight = update_weight + 1
    if new_M is None:
        new_M = jnp.zeros_like(M)
    new_M = (
        update_weight / new_update_weight * new_M + 
       1 / new_update_weight * X
    )
    return X.T, (new_M.T, new_update_weight)
