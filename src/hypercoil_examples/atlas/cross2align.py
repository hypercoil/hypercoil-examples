# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Two subjects, with functional alignment
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Projection of parcellation fit to one subject onto a second subject's
functional space. This time, we use the Empty ProMises algorithm to
align the functional data of the two subjects to a common space first.

Usage note: This won't work unless meaninit.py has been run first.
"""
from typing import Optional

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from hypercoil.init import (
    CortexSubcortexGIfTIAtlas,
    VonMisesFisher,
)
from hypercoil.engine import Tensor
from hypercoil.nn.atlas import AtlasLinear

from hypercoil_examples.atlas.cross2subj import ATLAS_PATH, visualise
from hypercoil_examples.atlas.encoders import (
    create_icosphere_encoder,
)
from hypercoil_examples.atlas.promises import empty_promises
from hypercoil_examples.atlas.spatialinit import init_spatial_priors
from hypercoil_examples.atlas.vmf import get_data, whiten_data


SPATIAL_PRIOR_WEIGHT = (
    0.0, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0, 200.0, 500.0, 1000.0, 2000.0, 5000.0
    #0.0, 1.0, 40, 500.0, 1e6,
)


def estimate_distance_distribution(
    name: str,
    R: Tensor,
    spatial_prior_loc: Tensor,
    spatial_prior_data: Tensor,
    spatial_prior_weight: float,
):
    import pandas as pd
    import seaborn as sns
    loadings = jax.vmap(
        lambda x, i: x[i],
        in_axes=(0, 0),
    )(R, spatial_prior_loc.squeeze())
    loadings = loadings / jnp.linalg.norm(R, axis=-1, keepdims=True)
    distance = -jnp.log(spatial_prior_data)
    distance = jnp.broadcast_to(distance, loadings.shape)
    distance = jnp.where(spatial_prior_loc == -1, 4, distance)
    df = pd.DataFrame({
        'distance': distance.ravel(),
        'loadings': loadings.ravel(),
    })
    sns.violinplot(x='distance', y='loadings', data=df, density_norm='width')
    plt.savefig(
        f'/tmp/desc-distloadings_{name}_priorweight-{spatial_prior_weight}.png'
    )
    plt.close('all')


def parcel_attention(
    atlas: CortexSubcortexGIfTIAtlas,
    spatial_loc_left: Tensor,
    spatial_loc_right: Tensor,
):
    parcel_left = atlas.maps['cortex_L'].argmax(0)
    attention_key = parcel_left[spatial_loc_left]
    attention_left = jnp.where(
        spatial_loc_left == -1,
        0,
        attention_key == parcel_left[..., None],
    ).astype(float)
    parcel_right = atlas.maps['cortex_R'].argmax(0)
    attention_key = parcel_right[spatial_loc_right]
    attention_right = jnp.where(
        spatial_loc_right == -1,
        0,
        attention_key == parcel_right[..., None],
    ).astype(float)
    return attention_left, attention_right


def main(
    spatial_prior_weight: Optional[float] = 0.0,
    use_parcel_attention: bool = False,
    rotate_back: bool = False,
    transpose: bool = False,
    plot_orig_and_template: bool = False,
):
    template = np.load('/tmp/mean_init.npy')
    spatial_loc_left, spatial_loc_right, spatial_data = init_spatial_priors()
    template = template / jnp.linalg.norm(template, axis=-1, keepdims=True)
    template_left = template[:spatial_loc_left.shape[0]]
    template_right = template[spatial_loc_left.shape[0]:]

    atlas = CortexSubcortexGIfTIAtlas(
        data_L=ATLAS_PATH['L'],
        data_R=ATLAS_PATH['R'],
        name='Glasser360',
    )
    if plot_orig_and_template:
        visualise(
            log_prob_L=atlas.maps['cortex_L'].T,
            log_prob_R=atlas.maps['cortex_R'].T,
            name='original',
        )
    encoder = create_icosphere_encoder()
    encoder = AtlasLinear.from_atlas(
        encoder,
        encode=True,
        key=jax.random.PRNGKey(0),
    )
    model = AtlasLinear.from_atlas(
        atlas,
        encode=True,
        forward_mode='project',
        key=jax.random.PRNGKey(0),
    )

    msc = get_data('MSC')
    enc = encoder(msc, encode=True, decode_labels=False)

    spatial_mu = model(
        atlas.coors, forward_mode='map', encode=False, decode_labels=False
    )
    temporal_mu = model(
        template,
        forward_mode='map',
        encode=False,
        decode_labels=False,
    )
    temporal_mu_L, temporal_mu_R = temporal_mu[:180], temporal_mu[180:]
    temporal_L = VonMisesFisher(mu=temporal_mu_L, kappa=10)
    temporal_R = VonMisesFisher(mu=temporal_mu_R, kappa=10)
    spatial_L = VonMisesFisher(mu=spatial_mu[:180], kappa=10)
    spatial_R = VonMisesFisher(mu=spatial_mu[180:], kappa=10)
    if plot_orig_and_template:
        log_prob_L = (
            temporal_L.log_prob(template_left) +
            spatial_L.log_prob(atlas.coors[:enc['cortex_L'].shape[0]])
        )
        log_prob_R = (
            temporal_R.log_prob(template_right) +
            spatial_R.log_prob(atlas.coors[enc['cortex_L'].shape[0]:])
        )
        visualise(
            log_prob_L=log_prob_L,
            log_prob_R=log_prob_R,
            name='template',
        )

    enc['cortex_L'], _ = whiten_data(enc['cortex_L'])
    enc['cortex_R'], _ = whiten_data(enc['cortex_R'])
    enc['cortex_L'] = enc['cortex_L'] / jnp.linalg.norm(
        enc['cortex_L'], axis=-1, keepdims=True
    )
    enc['cortex_R'] = enc['cortex_R'] / jnp.linalg.norm(
        enc['cortex_R'], axis=-1, keepdims=True
    )
    #enc_orig = {**enc}
    if transpose:
        size = template_left.shape[-1]
        template_left, template_right = template_left.T, template_right.T
        enc['cortex_L'], enc['cortex_R'] = enc['cortex_L'].T, enc['cortex_R'].T
        spatial_loc_left = spatial_loc_right = jnp.arange(size)[..., None]
        spatial_data =jnp.ones((size, 1))
    if use_parcel_attention:
        print('Using parcel attention...')
        attention_left, attention_right = parcel_attention(
            atlas, spatial_loc_left, spatial_loc_right
        )
        spatial_data_left = spatial_data[None, :] * attention_left
        spatial_data_right = spatial_data[None, :] * attention_right
    else:
        spatial_data_left = spatial_data
        spatial_data_right = spatial_data

    print('Aligning left hemisphere to mean template...')
    enc['cortex_L'], (_, _, Rl, Ql, Pl) = empty_promises(
        X=enc['cortex_L'],
        M=template_left,
        spatial_prior_loc=spatial_loc_left,
        spatial_prior_data=spatial_data_left,
        spatial_prior_weight=spatial_prior_weight,
        return_loading=True,
    )
    print(jnp.linalg.norm(template_left - enc['cortex_L']))
    #assert 0
    R = Pl.T @ Rl.T @ Ql
    estimate_distance_distribution(
        'hemi-left_proj-back',
        R,
        spatial_prior_loc=spatial_loc_left,
        spatial_prior_data=spatial_data_left,
        spatial_prior_weight=spatial_prior_weight,
    )
    print('Aligning right hemisphere to mean template...')
    enc['cortex_R'], (_, _, Rr, Qr, Pr) = empty_promises(
        X=enc['cortex_R'],
        M=template_right,
        spatial_prior_loc=spatial_loc_right,
        spatial_prior_data=spatial_data_right,
        spatial_prior_weight=spatial_prior_weight,
        return_loading=True,
    )
    R = Pr.T @ Rr.T @ Qr
    estimate_distance_distribution(
        'hemi-right_proj-back',
        R,
        spatial_prior_loc=spatial_loc_right,
        spatial_prior_data=spatial_data_right,
        spatial_prior_weight=spatial_prior_weight,
    )
    if transpose:
        enc['cortex_L'], enc['cortex_R'] = enc['cortex_L'].T, enc['cortex_R'].T
    enc['cortex_L'] = enc['cortex_L'] / jnp.linalg.norm(
        enc['cortex_L'], axis=-1, keepdims=True
    )
    enc['cortex_R'] = enc['cortex_R'] / jnp.linalg.norm(
        enc['cortex_R'], axis=-1, keepdims=True
    )
    # print('Aligning left hemisphere to mean template...')
    # enc['cortex_L'], _ = empty_promises(
    #     X=enc['cortex_L'].T,
    #     M=template_left.T,
    #     spatial_prior_loc=jnp.ones(template_left.shape[-1], dtype=int)[..., None], #spatial_loc_left,
    #     spatial_prior_data=jnp.ones((template_left.shape[-1], 1), dtype=float), #spatial_data,
    #     spatial_prior_weight=0.0,
    # )
    # print('Aligning right hemisphere to mean template...')
    # enc['cortex_R'], _ = empty_promises(
    #     X=enc['cortex_R'].T,
    #     M=template_right.T,
    #     spatial_prior_loc=jnp.ones(template_right.shape[-1], dtype=int)[..., None], #spatial_loc_right,
    #     spatial_prior_data=jnp.ones((template_right.shape[-1], 1), dtype=float), #spatial_data,
    #     spatial_prior_weight=0.0,
    # )
    # enc['cortex_L'], enc['cortex_R'] = enc['cortex_L'].T, enc['cortex_R'].T
    log_prob_L = (
        temporal_L.log_prob(enc['cortex_L']) +
        spatial_L.log_prob(atlas.coors[:enc['cortex_L'].shape[0]])
    )
    log_prob_R = (
        temporal_R.log_prob(enc['cortex_R']) +
        spatial_R.log_prob(atlas.coors[enc['cortex_L'].shape[0]:])
    )
    # _, _, (log_prob_L,) = empty_promises(
    #     X=template_left,
    #     M=enc_orig['cortex_L'],
    #     spatial_prior_loc=spatial_loc_left,
    #     spatial_prior_data=spatial_data,
    #     spatial_prior_weight=spatial_prior_weight,
    #     cotransport=(log_prob_L,),
    # )
    # _, _, (log_prob_R,) = empty_promises(
    #     X=template_right,
    #     M=enc_orig['cortex_R'],
    #     spatial_prior_loc=spatial_loc_right,
    #     spatial_prior_data=spatial_data,
    #     spatial_prior_weight=spatial_prior_weight,
    #     cotransport=(log_prob_R,),
    # )
    if rotate_back:
        log_prob_L = Ql.T @ Rl @ (Pl @ log_prob_L)
        log_prob_R = Qr.T @ Rr @ (Pr @ log_prob_R)
    prob_L = jax.nn.softmax(-log_prob_L, axis=-1)
    prob_R = jax.nn.softmax(-log_prob_R, axis=-1)
    expected_energy_L = jnp.sum(prob_L * log_prob_L, axis=-1)
    expected_energy_R = jnp.sum(prob_R * log_prob_R, axis=-1)
    expected_energy_L = jnp.where(
        jnp.isnan(expected_energy_L),
        0,
        expected_energy_L,
    )
    expected_energy_R = jnp.where(
        jnp.isnan(expected_energy_R),
        0,
        expected_energy_R,
    )

    visualise(
        log_prob_L=log_prob_L,
        log_prob_R=log_prob_R,
        name=f'MSC_{spatial_prior_weight}',
    )
    visualise(
        energy_L=expected_energy_L,
        energy_R=expected_energy_R,
        name=f'MSC_energy_{spatial_prior_weight}',
        parcellation=False,
    )
    print(
        'Total parcels detected (left hemisphere):',
        len(jnp.unique(log_prob_L.argmax(-1)))
    )
    print(
        'Total parcels detected (right hemisphere):',
        len(jnp.unique(log_prob_R.argmax(-1)))
    )


    hcp = get_data('HCP')
    enc = encoder(hcp, encode=True, decode_labels=False)
    enc['cortex_L'], _ = whiten_data(enc['cortex_L'])
    enc['cortex_R'], _ = whiten_data(enc['cortex_R'])
    #enc_orig = {**enc}
    if transpose:
        enc['cortex_L'], enc['cortex_R'] = enc['cortex_L'].T, enc['cortex_R'].T

    print('Aligning left hemisphere to mean template...')
    enc['cortex_L'], (_, _, Rl, Ql, Pl) = empty_promises(
        X=enc['cortex_L'],
        M=template_left,
        spatial_prior_loc=spatial_loc_left,
        spatial_prior_data=spatial_data_left,
        spatial_prior_weight=spatial_prior_weight,
        return_loading=True,
    )
    R = Pl.T @ Rl.T @ Ql
    estimate_distance_distribution(
        'hemi-left_proj-cross',
        R,
        spatial_prior_loc=spatial_loc_left,
        spatial_prior_data=spatial_data_left,
        spatial_prior_weight=spatial_prior_weight,
    )
    print('Aligning right hemisphere to mean template...')
    enc['cortex_R'], (_, _, Rr, Qr, Pr) = empty_promises(
        X=enc['cortex_R'],
        M=template_right,
        spatial_prior_loc=spatial_loc_right,
        spatial_prior_data=spatial_data_right,
        spatial_prior_weight=spatial_prior_weight,
        return_loading=True,
    )
    R = Pr.T @ Rr.T @ Qr
    estimate_distance_distribution(
        'hemi-right_proj-cross',
        R,
        spatial_prior_loc=spatial_loc_right,
        spatial_prior_data=spatial_data_right,
        spatial_prior_weight=spatial_prior_weight,
    )
    if transpose:
        enc['cortex_L'], enc['cortex_R'] = enc['cortex_L'].T, enc['cortex_R'].T
    enc['cortex_L'] = enc['cortex_L'] / jnp.linalg.norm(
        enc['cortex_L'], axis=-1, keepdims=True
    )
    enc['cortex_R'] = enc['cortex_R'] / jnp.linalg.norm(
        enc['cortex_R'], axis=-1, keepdims=True
    )
    log_prob_L = (
        temporal_L.log_prob(enc['cortex_L']) +
        spatial_L.log_prob(atlas.coors[:enc['cortex_L'].shape[0]])
    )
    log_prob_R = (
        temporal_R.log_prob(enc['cortex_R']) +
        spatial_R.log_prob(atlas.coors[enc['cortex_L'].shape[0]:])
    )
    # _, _, (log_prob_L,) = empty_promises(
    #     X=template_left,
    #     M=enc_orig['cortex_L'],
    #     spatial_prior_loc=spatial_loc_left,
    #     spatial_prior_data=spatial_data,
    #     spatial_prior_weight=spatial_prior_weight,
    #     cotransport=(log_prob_L,),
    # )
    # _, _, (log_prob_R,) = empty_promises(
    #     X=template_left,
    #     M=enc_orig['cortex_R'],
    #     spatial_prior_loc=spatial_loc_right,
    #     spatial_prior_data=spatial_data,
    #     spatial_prior_weight=spatial_prior_weight,
    #     cotransport=(log_prob_R,),
    # )
    if rotate_back:
        log_prob_L = Ql.T @ Rl @ (Pl @ log_prob_L)
        log_prob_R = Qr.T @ Rr @ (Pr @ log_prob_R)
    prob_L = jax.nn.softmax(-log_prob_L, axis=-1)
    prob_R = jax.nn.softmax(-log_prob_R, axis=-1)
    expected_energy_L = jnp.sum(prob_L * log_prob_L, axis=-1)
    expected_energy_R = jnp.sum(prob_R * log_prob_R, axis=-1)
    expected_energy_L = jnp.where(
        jnp.isnan(expected_energy_L),
        0,
        expected_energy_L,
    )
    expected_energy_R = jnp.where(
        jnp.isnan(expected_energy_R),
        0,
        expected_energy_R,
    )

    visualise(
        log_prob_L=log_prob_L,
        log_prob_R=log_prob_R,
        name=f'HCP_{spatial_prior_weight}',
    )
    visualise(
        energy_L=expected_energy_L,
        energy_R=expected_energy_R,
        name=f'HCP_energy_{spatial_prior_weight}',
        parcellation=False,
    )
    print(
        'Total parcels detected (left hemisphere):',
        len(jnp.unique(log_prob_L.argmax(-1)))
    )
    print(
        'Total parcels detected (right hemisphere):',
        len(jnp.unique(log_prob_R.argmax(-1)))
    )


if __name__ == "__main__":
    for weight in SPATIAL_PRIOR_WEIGHT:
        main(spatial_prior_weight=weight)
