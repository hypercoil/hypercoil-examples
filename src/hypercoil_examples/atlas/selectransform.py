# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Selectivity space transformation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Transformations of selectivity spaces -- the hyperspherical space of
selectivity profiles -- for instance, degree normalisation, thresholding,
Mahalanobis transformation, etc.
"""
from itertools import permutations
from typing import Optional, Union

import jax
import jax.numpy as jnp
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt

from hypercoil.engine import Tensor
from hypercoil.nn import AtlasLinear
from hypercoil_examples.atlas.vmf import (
    get_data, ENCODER_FACTORY, ENCODE_SELF
)
from hypercoil_examples.atlas.vmf import (
    whiten_data as incomplete_mahalanobis_transform
)
from hyve import (
    Cell, plotdef, surf_from_archive, surf_scalars_from_array,
    plot_to_image, save_figure
)


def degree_norm(X: Tensor) -> Tensor:
    normaliser = (
        (1 / jnp.sqrt(X.sum(axis=-1, keepdims=True))) @
        (1 / jnp.sqrt(X.sum(axis=-2, keepdims=True)))
    )
    return normaliser * X


def threshold(
    X: Tensor,
    threshold: float,
    binarise: bool = False,
) -> Tensor:
    repl = 1. if binarise else X
    X = jnp.where(
        X < threshold,
        0.,
        repl,
    )
    return X


def topk_threshold(
    X: Tensor,
    k: Union[float, int],
    binarise: bool = False,
    axis: int = -1,
) -> Tensor:
    if axis != -1:
        X = X.swapaxes(axis, -1)
    repl = 1. if binarise else X
    if isinstance(k, float):
        k = int(X.shape[-1] * k)
    X = jnp.where(
        X < jnp.partition(X, k)[..., k, None],
        0.,
        repl,
    )
    if axis != -1:
        X = X.swapaxes(axis, -1)
        X = jnp.where(
            X.sum(-1, keepdims=True) == 0,
            -1 / X.shape[-1],
            X,
        )
    return X


def logistic_mixture_threshold(
    loc: Tensor,
    scale: Tensor,
    k: Union[float, int],
    axis: int = -1,
) -> Tensor:
    if axis != -1:
        loc = loc.swapaxes(axis, -1)
        scale = scale.swapaxes(axis, -1)
    if isinstance(k, float):
        k = int(loc.shape[-1] * k)
    est = jnp.partition(loc, k)[..., k, None]
    X = 1 - jax.nn.sigmoid((est - loc) / scale)
    # Really, we should run a bisection search, but for
    # now this is good enough
    X = X * (X.shape[-1] - k) / (X.sum(-1, keepdims=True))
    X = X - (loc.shape[-1] - k) / loc.shape[-1]
    if axis != -1:
        X = X.swapaxes(axis, -1)
        X = jnp.where(
            X.sum(-1, keepdims=True) == 0,
            -1 / X.shape[-1],
            X,
        )
    return X


def create_plot(
    X: Tensor,
    vmin: float = -0.05,
    vmax: float = 0.05,
    selected: slice = slice(-2, None),
):
    fig = plt.figure(figsize=(18, 9), tight_layout=True)
    gs = gridspec.GridSpec(2, 4)
    normalised = (X / jnp.linalg.norm(X, axis=-1, keepdims=True)).T

    ax = fig.add_subplot(gs[:, :2])
    ax.imshow(
        normalised,
        aspect='auto',
        vmin=vmin,
        vmax=vmax,
        cmap='inferno',
    )

    L, Q = jnp.linalg.eigh(jnp.cov(X.T))
    pcs = (Q.T @ X.T)[selected]
    lim = jnp.abs(pcs).max()
    ax = fig.add_subplot(gs[0, 2])
    ax.scatter(*pcs, s=0.01)
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    L, Q = jnp.linalg.eigh(jnp.cov(normalised))
    pcs = (Q.T @ normalised)[selected]
    ax = fig.add_subplot(gs[1, 2])
    ax.scatter(*pcs, s=0.01)
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)

    coors = X.T[selected]
    lim = jnp.abs(coors).max()
    ax = fig.add_subplot(gs[0, 3])
    ax.scatter(*coors, s=0.01)
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax = fig.add_subplot(gs[1, 3])
    ax.scatter(*normalised[selected], s=0.01)
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)

    return fig, (L, pcs, coors)


def brain_visualisation(
    name: str,
    data: Optional[jnp.ndarray] = None,
):
    layout = Cell() / Cell() << (1 / 2)
    layout = layout | layout << (1 / 2)
    annotations = {
        0: dict(
            hemisphere='left',
            view='lateral',
        ),
        1: dict(
            hemisphere='left',
            view='medial',
        ),
        2: dict(
            hemisphere='right',
            view='lateral',
        ),
        3: dict(
            hemisphere='right',
            view='medial',
        ),
    }
    layout = layout.annotate(annotations)
    cmap = 'RdYlBu_r'
    plot_f = plotdef(
        surf_from_archive(),
        surf_scalars_from_array(
            'surf_scalars',
            is_masked=True,
        ),
        plot_to_image(),
        save_figure(
            layout_kernel=layout,
            padding=10,
            canvas_size=(1000, 800),
            canvas_color=(1, 1, 1),
            fname_spec=f'scalars-{name}',
            scalar_bar_action='collect',
        ),
    )
    plot_f(
        template='fsLR',
        surf_projection='veryinflated',
        surf_scalars_array=data,
        surf_scalars_cmap=cmap,
        surf_scalars_clim='robust',
        window_size=(500, 400),
        hemisphere=['left', 'right'],
        views={
            'left': ('medial', 'lateral'),
            'right': ('medial', 'lateral'),
        },
        # theme=pv.themes.DarkTheme(),
        output_dir='/tmp',
        load_mask=True,
    )


def save_plot(
    X: Tensor,
    vmin: float = -0.05,
    vmax: float = 0.05,
    selected: slice = slice(-2, None),
    fname: str = 'selectivity_space.png',
):
    fig, (L, pcs, _) = create_plot(X, vmin, vmax, selected)
    fig.savefig(f'/tmp/{fname}.png')
    brain_visualisation(f'{fname}pc1', data=pcs[0])
    brain_visualisation(f'{fname}pc2', data=pcs[1])
    plt.figure(figsize=(6, 6))
    plt.plot(range(len(L)), L[::-1], 'o-')
    plt.savefig(f'/tmp/{fname}eig.png')
    plt.close('all')


def main():
    atlas = ENCODER_FACTORY['icosphere']()
    model = AtlasLinear.from_atlas(atlas, encode=True, key=jax.random.PRNGKey(0))
    data = get_data('MSC')
    # coors, parcels_enc, atlas_coors = ENCODE_SELF['icosphere'](
    #     model=model, data=data, atlas=atlas
    # )
    enc = model(data)
    enc = jnp.concatenate((enc['cortex_L'], enc['cortex_R']))

    save_plot(enc, fname='enc')
    enc_deg = degree_norm(jnp.where(enc > 0, enc, 0))
    save_plot(enc_deg, vmin=0., fname='enc_deg')
    enc_thr1 = topk_threshold(enc, k=0.9, binarise=True, axis=-1)
    save_plot(enc_thr1, vmin=0., fname='enc_thr1')
    enc_thr0 = topk_threshold(enc, k=0.9, binarise=True, axis=-2)
    save_plot(enc_thr0, vmin=0., fname='enc_thr0')
    # enc_thr_deg = degree_norm(jnp.where(enc_thr0 > 0, enc_thr0, 0))
    # save_plot(enc_thr_deg, vmin=0., fname='enc_thr_deg')
    # loc = jnp.arctanh(jnp.clip(enc, -0.75, 0.75))
    loc = enc
    scale = jnp.minimum(-2e-2 * jnp.log(jnp.abs(loc)), 5)
    enc_sig = logistic_mixture_threshold(enc, scale, k=0.9, axis=0)
    save_plot(enc_sig, vmin=0., fname='enc_sig')
    enc_sph_thr0 = topk_threshold(
        incomplete_mahalanobis_transform(enc)[0], k=0.9, axis=0
    )
    save_plot(enc_sph_thr0, vmin=0., fname='enc_sph_thr0')
    enc_sph_sig = logistic_mixture_threshold(
        incomplete_mahalanobis_transform(loc)[0], scale, k=0.9, axis=-1
    )
    save_plot(enc_sph_sig, vmin=0., fname='enc_sph_sig')
    enc_sph = incomplete_mahalanobis_transform(enc)[0]
    save_plot(enc_sph, vmin=0., fname='enc_sph')
    enc_sph_deg = degree_norm(jnp.where(enc_sph > 0, enc_sph, 0))
    save_plot(enc_sph_deg, vmin=0., fname='enc_sph_deg')


if __name__ == '__main__':
    main()
