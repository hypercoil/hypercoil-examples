# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
DCCC
~~~~
Distance-controlled coaffiliation coefficient (DCCC) is a relaxation of the
DCBC (distance-controlled boundary coefficient, Zhi et al.) that allows for
the evaluation of continuous-valued parcellations. It's also extremely slow.
"""
from typing import Any, Optional, Literal, Mapping, Sequence, Tuple

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import nibabel as nb
import numpy as np
import pandas as pd
import seaborn as sns
import templateflow.api as tflow

from hypercoil.engine import Tensor, _to_jax_array
from hypercoil.init.atlas import CortexSubcortexGIfTIAtlas
from hypercoil_examples.atlas.cross2subj import ATLAS_PATH
from hypercoil_examples.atlas.vmf import get_data, ENCODER_FACTORY


def get_atlas_path(name: str) -> Mapping:
    return {
        'L': (
            '/Users/rastkociric/Downloads/sparque/src/sparque/'
            f'DCBC/parcellations/{name}.32k.L.label.gii'
        ),
        'R': (
            '/Users/rastkociric/Downloads/sparque/src/sparque/'
            f'DCBC/parcellations/{name}.32k.R.label.gii'
        ),
    }

NUM_NODES = 4000
START = 0
KEY = 71
NODE_SLICE = slice(START, START + NUM_NODES)
ATLAS_PATH = get_atlas_path('Glasser_2016')


def spherical_distance(coors: Tensor) -> Tensor:
    radius = jnp.linalg.norm(coors, axis=-1, keepdims=True)
    coors = coors / radius
    return radius * jnp.arccos(
        jnp.clip(coors @ coors.T + jnp.eye(coors.shape[0]), -1., 1.)
    )


def gauss_kernel(scale: float) -> callable:
    def _call(loc: float, distances: Tensor) -> Tensor:
        return jnp.exp(-((distances - loc) ** 2 / scale))
    return _call


def bin_kernel(scale: float) -> callable:
    def _call(loc: float, distances: Tensor) -> Tensor:
        return (jnp.abs(distances - loc) < scale).astype(float)
    return _call


def dccc(
    coors: Tensor,
    features: Tensor,
    assignment: Tensor,
    distance_matrix: Optional[Tensor] = None,
    distance: callable = spherical_distance,
    distance_kernel: callable = gauss_kernel(2.),
    integral_samples: Tensor = jnp.arange(40) * 0.9,
    normalise: bool = True,
    coaffiliation: Literal['probability', 'similarity'] = 'probability',
):
    if distance_matrix is None:
        distance_matrix = distance(coors)
    self_loop = (distance_matrix == 0)
    if normalise:
        features = (
            features - features.mean(axis=-1, keepdims=True)
        ) / jnp.linalg.norm(features, axis=-1, keepdims=True)
    match coaffiliation:
        case 'probability':
            assignment = assignment / (
                assignment.sum(axis=-1, keepdims=True)
                + jnp.finfo(assignment.dtype).eps
            )
        case 'similarity':
            assignment = assignment / (
                jnp.linalg.norm(assignment, axis=-1, keepdims=True)
                + jnp.finfo(assignment.dtype).eps
            )
    width = jnp.diff(integral_samples)
    #height = jnp.zeros_like(width)
    wweight = jnp.zeros_like(width)
    bweight = jnp.zeros_like(width)
    within = jnp.zeros_like(width)
    between = jnp.zeros_like(width)
    # height = jnp.zeros((width.size, assignment.shape[-1]))
    # weight = jnp.zeros((width.size, assignment.shape[-1]))
    for i, sample_point in enumerate(integral_samples[1:]):
        print(f'Bin {i}')
        kernel_matrix = jnp.where(
            self_loop,
            0.,
            distance_kernel(sample_point, distance_matrix)
        )
        wweight = wweight.at[i].set(
            jnp.einsum(
                '...ap,...bp,...ab->', assignment, assignment, kernel_matrix
            ) + jnp.finfo(width.dtype).eps
        )
        bweight = bweight.at[i].set(
            jnp.einsum(
                '...ap,...bp,...ab->', assignment, 1 - assignment, kernel_matrix
            ) + jnp.finfo(width.dtype).eps
        )
        within = within.at[i].set(
            jnp.einsum(
                '...ap,...bp,...at,...bt,...ab->',
                assignment, assignment, features, features, kernel_matrix
            ) / wweight[i]
        )
        between = between.at[i].set(
            jnp.einsum(
                '...ap,...bp,...at,...bt,...ab->',
                assignment, 1 - assignment, features, features, kernel_matrix
            ) / bweight[i]
        )
    #height = (within - between)
    weight = wweight * bweight / (wweight + bweight)

    result = width * (within - between) * (weight / weight.sum())
    return {
        'integral_samples': integral_samples[1:],
        'w_within': wweight,
        'w_between': bweight,
        'corr_within': within,
        'corr_between': between,
        'weight': weight,
        'dccc': result,
    }


def plot_dccc(
    result: Mapping[str, Any],
    output_path: str,
):
    fig, ax = plt.subplots(2, 2, figsize=(8, 8))
    fig.suptitle(f'DCCC {result["dccc"].sum()}', fontsize=16)
    ax[0][0].plot(
        result['integral_samples'],
        (
            result['corr_within'] * result['w_within'] +
            result['corr_between'] * result['w_between']
        ) / (result['w_within'] + result['w_between']),
        c='black',
    )
    ax[0][1].bar(
        result['integral_samples'],
        result['w_between'],
        color='red',
        alpha=0.5,
        linewidth=1,
    )
    ax[0][1].bar(
        result['integral_samples'],
        result['w_within'],
        color='black',
        alpha=0.5,
        linewidth=1,
    )
    ax[0][1].axvline(
        (result['integral_samples'] * result['w_within']).sum() / result['w_within'].sum(),
        c='black',
        linestyle='--',
    )
    ax[0][1].axvline(
        (result['integral_samples'] * result['w_between']).sum() / result['w_between'].sum(),
        c='red',
        linestyle='--',
    )
    ax[1][0].plot(
        result['integral_samples'],
        result['weight'] / result['weight'].sum(),
        c='black',
    )
    ax[1][1].plot(
        result['integral_samples'],
        result['corr_within'],
        c='black',
    )
    ax[1][1].plot(
        result['integral_samples'],
        result['corr_between'],
        c='red',
    )
    sns.despine(fig)
    plt.savefig(output_path)


def main():
    #ATLAS_PATH = get_atlas_path('Yeo_JNeurophysiol11_7Networks')
    atlas_null = ENCODER_FACTORY['icosphere'](n_subdivisions=4, disc_scale=1.5)
    atlas = CortexSubcortexGIfTIAtlas(
        data_L=ATLAS_PATH['L'],
        data_R=ATLAS_PATH['R'],
        name='parcellation',
    )
    # data = get_data('HCP')
    # data = data[atlas.compartments.compartments['cortex_L'].mask_array]
    data = np.stack([
        e.data for e in
        nb.load('/Users/rastkociric/Downloads/DCBC/data/s02/s02.L.wbeta.32k.func.gii').darrays
    ]).T
    mask = tflow.get(
        'fsLR',
        density='32k',
        hemi='L',
        space=None,
        suffix='dparc',
        desc='nomedialwall',
    )
    mask = nb.load(mask).darrays[0].data.astype(bool)
    data = data[mask]
    data = jnp.where(jnp.isnan(data), jnp.arange(data.shape[-1])[None, :], data)


    coors_L = atlas.coors[atlas.compartments.compartments['cortex_L'].mask_array]
    coors_R = atlas.coors[atlas.compartments.compartments['cortex_R'].mask_array]
    order = jax.random.permutation(
        jax.random.PRNGKey(KEY),
        len(data),
    )[NODE_SLICE]

    result = dccc(
        coors=coors_L[order],
        features=data[order],
        assignment=atlas.maps['cortex_L'].T[order],
        integral_samples=jnp.arange(40) * 0.9,
    )
    dccc_result = result.get('dccc')
    print(dccc_result.sum())
    plot_dccc(result, '/tmp/dccc.png')

    result_null = dccc(
        coors=coors_L[order],
        features=data[order],
        assignment=atlas_null.maps['cortex_L'].T[order],
        integral_samples=jnp.arange(40) * 0.9,
    )
    dccc_result_null = result_null.get('dccc')
    print(dccc_result_null.sum())
    plot_dccc(result_null, '/tmp/dccc_null.png')


    fig, ax = plt.subplots(7, 6, figsize=(24, 28), tight_layout=True)
    slc = slice(100, 200)

    distance_matrix = spherical_distance(coors_L[:1000, :])
    distance_submatrix = distance_matrix[slc, slc]
    distance_kernel = gauss_kernel(2.)
    #distance_kernel = bin_kernel(0.5)

    ax[0][0].imshow(distance_submatrix)
    for i, sample in enumerate(result['integral_samples']):
        kernel_submatrix = distance_kernel(sample, distance_submatrix)
        r = (i + 1) // 6
        c = (i + 1) % 6
        ax[r][c].imshow(kernel_submatrix)
        ax[r][c].set_xticks([])
        ax[r][c].set_yticks([])

    fig.savefig('/tmp/dccc_kernels.png')


if __name__ == '__main__':
    main()
