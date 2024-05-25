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


N_POINTS = 1000
CLUSTER_KEY = 0
KEY = 10


def empty_promises(
    X: Tensor,
    M: Tensor,
    spatial_prior_loc: Tensor,
    spatial_prior_data: Tensor,
    spatial_prior_weight: float,
    new_M: Optional[Tensor] = None,
    update_weight: int = 0,
    cotransport: Optional[Tensor] = None,
    return_loading: bool = False,
) -> Tuple[Tensor, Tuple[Tensor, float]]:
    """
    A single step of the empty ProMises algorithm.

    The spatial prior loc and data are the index and weight of a sparse
    matrix in ELL format. The matrix is implicitly symmetrised when the
    required matrix-vector products are computed.

    Details of the original ProMises algorithm can be found in the paper:
    https://onlinelibrary.wiley.com/doi/full/10.1002/hbm.26170
    """
    if M is not None:
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
            #N.T @ Z +
            Z.T @ N +
            spatial_prior_weight * spatial_prior,
        )
        R = U @ V
        #assert 0
        Z = Z @ R
        X = Z @ P
        if cotransport is not None:
            cotransport = [P.T @ R.T @ (Q @ C) for C in cotransport]
            # TODO: We must correct for the Jacobian of the transformation,
            #       since we're mostly cotransporting probabilities.
    else:
        M = X = X.T
    new_update_weight = update_weight + 1
    if new_M is None:
        new_M = jnp.zeros_like(M).T
    new_M = (
        update_weight / new_update_weight * new_M + 
        1 / new_update_weight * X.T
    )
    if return_loading:
        state = (new_M, new_update_weight, R, Q, P) # P.T @ R.T @ Q)
    else:
        state = (new_M, new_update_weight)
    if cotransport is not None:
        return X.T, state, cotransport
    return X.T, state


def main():
    import jax
    import numpyro
    from scipy.linalg import orthogonal_procrustes
    import matplotlib.pyplot as plt

    def sample_cluster_assignments(
        cluster_outer: numpyro.distributions.Categorical,
        cluster_inner: numpyro.distributions.Categorical,
        n_points: int = N_POINTS,
        key: int = CLUSTER_KEY,
    ) -> Tensor:
        key = jax.random.PRNGKey(key)

        key_o, key_i = jax.random.split(key, 2)
        outer_labels = cluster_outer.sample(key=key_o, sample_shape=(n_points,))
        inner_labels = cluster_inner.sample(key=key_i, sample_shape=(n_points,))
        return outer_labels, inner_labels

    def sample_data(
        coords: numpyro.distributions.Normal,
        outer_labels: Tensor,
        inner_labels: Tensor,
        n_points: int = N_POINTS,
        *,
        key: 'jax.random.PRNGKey',
    ) -> Tensor:
        sample = coords.sample(
            key=key,
            sample_shape=(1000,),
        ).transpose(0, 2, 1, 3)
        sample = sample[(jnp.arange(n_points), outer_labels, inner_labels)]
        sample = sample / jnp.linalg.norm(sample, axis=-1, keepdims=True)
        return sample

    def solve_and_plot(
        coords: numpyro.distributions.Normal,
        outer_labels: Tensor,
        inner_labels: Tensor,
        size: int,
        key: int,
    ):
        key_src, key_ref, key_rot = jax.random.split(jax.random.PRNGKey(key), 3)
        A = jax.random.normal(key=key_rot, shape=(size, size))
        Q, _ = jnp.linalg.qr(A)
        sample_ref = sample_data(coords, outer_labels, inner_labels, key=key_ref)
        sample_src = sample_data(coords, outer_labels, inner_labels, key=key_src)
        try:
            sample_src = sample_src @ Q
            sample_src = sample_src.T
            sample_ref = sample_ref.T
            transpose = True
        except TypeError:
            sample_src = Q @ sample_src
            transpose = False
        # sample3 = sample_data(outerc, innerc, key=100)
        # sample3r = QQ @ sample3
        #sample3r = sample3 @ Q

        # sample3to1refR, _ = orthogonal_procrustes(
        #     A=sample3r,
        #     B=sample,
        # )
        # sample3to1ref = sample3r @ sample3to1refR

        R, _ = orthogonal_procrustes(
            A=sample_src.T,
            B=sample_ref.T,
        )
        sample_src2ref_ref = (sample_src.T @ R).T

        sample_src2ref, (_, _) = empty_promises(
            X=sample_src,
            M=sample_ref,
            spatial_prior_loc=jnp.arange(size)[..., None],
            spatial_prior_data=jnp.ones((size, 1)),
            spatial_prior_weight=0.0,
        )

        if transpose:
            sample_src = sample_src.T
            sample_ref = sample_ref.T
            sample_src2ref = sample_src2ref.T
            sample_src2ref_ref = sample_src2ref_ref.T
        sample_src2ref_ref = sample_src2ref_ref / jnp.linalg.norm(
            sample_src2ref_ref, axis=-1, keepdims=True
        )
        sample_src2ref = sample_src2ref / jnp.linalg.norm(
            sample_src2ref, axis=-1, keepdims=True
        )

        def create_figure(
            name: str,
            data: Tensor,
        ):
            view = (None,) # (40, -30, 'z')

            fig = plt.figure(figsize=(12, 6))
            ax = fig.add_subplot(121, projection='3d')
            ax.view_init(*view)
            ax.scatter(*data.T, c=outer_labels, cmap='nipy_spectral')
            ax = fig.add_subplot(122, projection='3d')
            ax.view_init(*view)
            ax.scatter(*data.T, c=inner_labels, cmap='nipy_spectral')
            fig.savefig(f'/tmp/{size}{name}.png')

        create_figure('sample_ref', sample_ref)
        create_figure('sample_src', sample_src)
        create_figure('sample_src2ref_procrustes', sample_src2ref_ref)
        create_figure('sample_src2ref_promises', sample_src2ref)

    # generative model
    cluster_outer = numpyro.distributions.Categorical(
        jnp.asarray([0.4, 0.21, 0.16, 0.13, 0.1]),
    )
    cluster_inner = numpyro.distributions.Categorical(
        jnp.asarray([0.3, 0.3, 0.2, 0.2])
    )
    mu_inner = numpyro.distributions.Normal(
        loc=jnp.asarray((
            (1, -1, 0),
            (0, 0, 1),
            (0, 0, -1),
            (0, 1, 0),
            (-1, 0, 0),
        )),
        scale=jnp.asarray((0.25, 0.2, 0.2, 0.15, 0.15))[..., None]
    )
    coords = numpyro.distributions.Normal(
        loc=mu_inner.sample(key=jax.random.PRNGKey(0), sample_shape=(4,)),
        scale=0.1,
    )

    outer_labels, inner_labels = sample_cluster_assignments(
        cluster_outer,
        cluster_inner,
    )
    solve_and_plot(coords, outer_labels, inner_labels, size=N_POINTS, key=KEY)
    solve_and_plot(coords, outer_labels, inner_labels, size=3, key=KEY)
    assert 0


if __name__ == '__main__':
    main()
