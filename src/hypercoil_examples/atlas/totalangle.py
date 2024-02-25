# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Inner subspace angles
~~~~~~~~~~~~~~~~~~~~~
Principal and total angles between subspaces defined by matrix slices of a
tensor.
"""
from typing import Optional

from hypercoil.engine import Tensor

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from scipy.linalg import orth, subspace_angles

from hypercoil.functional import recondition_eigenspaces


def image_basis(
    X: Tensor,
    rcond: Optional[Tensor] = None,
    rank: Optional[int] = None,
) -> Tensor:
    """
    Compute the basis of the image of the matrices in ``X``.
    """
    U, S, V = jnp.linalg.svd(X, full_matrices=False)
    m, n = U.shape[-2], V.shape[-1]
    if rcond is None:
        rcond = jnp.finfo(S.dtype).eps * jnp.maximum(m, n)
    tol = jnp.max(S) * rcond
    count = jnp.sum(S > tol, dtype=int)
    key = jnp.broadcast_to(jnp.arange(U.shape[-1]), U.shape)
    Q = jnp.where(key < count, U, 0.)
    if rank is not None:
        Q = Q[..., :rank]
    return Q


def inner_angles(
    X: Tensor,
    Q: Optional[Tensor] = None,
    rcond: Optional[Tensor] = None,
    rank: Optional[int] = None,
    psi: float = 0.0,
    xi: float = 0.0,
    *,
    key: Optional['jax.random.PRNGKey'] = None,
) -> Tensor:
    """
    Compute the principal angles between the subspaces defined by the columns
    of the matrices in ``X``.
    """
    if Q is None:
        Q = image_basis(X, rcond=rcond, rank=rank)
    QHQ = jnp.einsum('...cnj,...dnk->...cdjk', Q.conj(), Q)
    B = Q - jnp.einsum('...cnj,...cdjk->...cdnk', Q, QHQ)
    if key is not None:
        key_sin, key_cos = jax.random.split(key, 2)
        # The gradient of SVD is undefined when the singular values are
        # degenerate. We can perturb the eigenspaces to avoid this issue.
        svd_cos_arg = recondition_eigenspaces(
            QHQ, psi=psi, xi=xi, key=key_cos
        )
        svd_sin_arg = B + psi - xi + jax.random.uniform(
            key=key_sin, shape=B.shape, maxval=xi
        )
    else:
        svd_cos_arg = QHQ
        svd_sin_arg = B
    sigma = jnp.linalg.svd(svd_cos_arg, compute_uv=False)
    # Our implementation of the principal angles differs from scipy's in that
    # we require the operation to be differentiable. Accordingly, we must
    # handle the case of infinite derivatives at the boundaries of the domain
    # of the arccos and arcsin functions.
    arcsin_arg = jnp.clip(
        jnp.linalg.svd(svd_sin_arg, compute_uv=False), -1., 1.
    )
    arccos_arg = jnp.clip(sigma[..., ::-1], -1., 1.)
    arcsin_mask = ~jnp.isclose(jnp.abs(arcsin_arg), 1.)
    arccos_mask = ~jnp.isclose(jnp.abs(arccos_arg), 1.)
    arcsin_sign = jnp.sign(arcsin_arg)
    arccos_sign = jnp.sign(arccos_arg)
    arcsin_arg = jnp.where(arcsin_mask, arcsin_arg, 0.)
    arccos_arg = jnp.where(arccos_mask, arccos_arg, 0.)
    return jnp.where(
        sigma ** 2 >= 0.5,
        jnp.where(
            arcsin_mask,
            jnp.arcsin(arcsin_arg),
            arcsin_sign * jnp.pi / 2,
        ),
        jnp.where(
            arccos_mask,
            jnp.arccos(arccos_arg),
            jnp.pi / 2 - arccos_sign * jnp.pi / 2,
        ),
        #jnp.arcsin(jnp.clip(
        #    jnp.linalg.svd(svd_sin_arg, compute_uv=False), -1., 1.
        #)),
        #jnp.arccos(jnp.clip(sigma[..., ::-1], -1., 1.)),
    )


def main():
    AAA = jax.random.normal(key=jax.random.PRNGKey(0), shape=(5, 10, 3))
    inp = (AAA @ AAA.swapaxes(-2, -1))
    assert inp.shape == (5, 10, 10)
    assert jnp.allclose(
        jax.jit(image_basis, static_argnames=('rank',))(inp, rank=3),
        jnp.stack([orth(e) for e in inp]),
    )
    assert jnp.allclose(
        jnp.asarray([[subspace_angles(e, f) for e in inp] for f in inp]),
        inner_angles(inp, rank=3),
        atol=1e-6,
    )
    inp = jnp.asarray((
        ((1, 0, 0, 0), (0, 1, 0, 0)),
        ((0, 1, 0, 0), (1, 0, 0, 0)),
        ((-1, 0, 0, 0), (0, 1, 0, 0)),
        ((0, 1, 0, 0), (0, 0, 1, 0)),
        ((0, 0, 1, 0), (0, 0, 0, 1)),
        ((0.9, 0, 0.1, 0), (0, 0.9, 0.1, 0)),
        ((0.8, 0, 0.2, 0), (0, 0.8, 0.2, 0)),
        ((0.7, 0, 0.3, 0), (0, 0.7, 0.3, 0)),
        ((0.6, 0, 0.4, 0), (0, 0.6, 0.4, 0)),
        ((0.5, 0, 0.5, 0), (0, 0.5, 0.5, 0)),
        ((0.4, 0, 0.6, 0), (0, 0.4, 0.6, 0)),
        ((0.3, 0, 0.7, 0), (0, 0.3, 0.7, 0)),
        ((0.2, 0, 0.8, 0), (0, 0.2, 0.8, 0)),
        ((0.1, 0, 0.9, 0), (0, 0.1, 0.9, 0)),
    )).swapaxes(-2, -1)
    subspace_angles(inp[0], inp[3]), inner_angles(inp)
    total_angle_matrix = jnp.prod(jnp.cos(inner_angles(inp)), axis=-1)
    print(total_angle_matrix)
    assert jnp.linalg.det(total_angle_matrix) == 0
    # Technically we should be able to use a larger submatrix, but the
    # floating point precision is not enough to guarantee that the determinant
    # is positive
    assert jnp.linalg.det(total_angle_matrix[3:7, 3:7]) > 0
    plt.imshow(total_angle_matrix, cmap='bone', vmin=0, vmax=1)
    plt.savefig('/tmp/total_angles_example.png')


if __name__ == '__main__':
    main()
