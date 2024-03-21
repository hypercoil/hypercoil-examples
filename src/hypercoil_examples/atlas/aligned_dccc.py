# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Aligned DCCC
~~~~~~~~~~~~
Compute the aligned DCCC for a group-template parcellation and for aligned
individual parcellations using the MSC data.

Usage note: This won't work unless meaninit.py has been run first.
"""
import pickle
from typing import Literal, Optional, Tuple

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from numpyro.distributions import Beta

from hypercoil.init import (
    CortexSubcortexGIfTIAtlas,
    VonMisesFisher,
)
from hypercoil.engine import Tensor
from hypercoil.nn.atlas import AtlasLinear

from hypercoil_examples.atlas.cross2subj import visualise
from hypercoil_examples.atlas.dccc import (
    dccc, get_atlas_path, plot_dccc
)
from hypercoil_examples.atlas.encoders import (
    create_icosphere_encoder,
)
from hypercoil_examples.atlas.promises import empty_promises
from hypercoil_examples.atlas.spatialinit import init_spatial_priors
from hypercoil_examples.atlas.vmf import _get_data, whiten_data

MSC_DATA_ROOT = '/Users/rastkociric/Downloads/ds000224/'
SPATIAL_PRIOR_WEIGHT = 5e2
DCCC_SAMPLE_KEY = 71
DCCC_NUM_NODES = 4000
SUBJECTS = ('01', '02', '03', '04', '05', '06', '07', '08', '09', '10')
SESSIONS = ('01', '02', '03', '04', '05', '06', '07', '08', '09', '10')
ATLAS_NAME = 'Glasser_2016'
MLE_TEMPERATURE = 0.5
MAX_ITER_ENERGY_EQUILIBRIUM = 20
SPATIAL_KAPPA_LIM = (1e1, 3e1)
TEMPORAL_KAPPA_LIM = (1e1, None)


def get_msc_dataset(
    subject: str,
    session: str,
) -> str:
    return (
        f'{MSC_DATA_ROOT}/sub-MSC{subject}_ses-func{session}_'
        'task-rest_space-fsLR_den-91k_bold.dtseries.nii'
    )


def map_into(X: Tensor, lim: Tuple[float, float]) -> Tensor:
    if lim[0] is None:
        return jnp.minimum(X, lim[1])
    if lim[1] is None:
        return jnp.maximum(X, lim[0])
    x_min, x_max = X.min(), X.max()
    if lim[0] == 'auto':
        lim = (x_min, lim[1])
    if lim[1] == 'auto':
        lim = (lim[0], x_max)
    return lim[0] + (X - x_min) * (lim[1] - lim[0]) / (x_max - x_min)


def kappa_iteration(
    coor: Tensor,
    asgt: Tensor,
    max_iter: int = 100,
):
    from hypercoil.init.vmf import log_bessel
    d = coor.shape[-1]
    def a_p(kappa: Tensor) -> Tensor:
        return jnp.exp(
            log_bessel(order=d / 2, kappa=kappa) -
            log_bessel(order=d / 2 - 1, kappa=kappa)
        )
    def init(rbar: Tensor) -> Tensor:
        return (rbar * (d - rbar ** 2)) / (1 - rbar ** 2)
    def step(kappa: Tensor, rbar: Tensor) -> Tensor:
        a_p_kappa = a_p(kappa)
        #return rbar * kappa * a_p_kappa
        return (
            kappa - (a_p_kappa - rbar) /
            (1 - a_p_kappa ** 2 - (d - 1) * a_p_kappa / kappa)
        )
    # if d == 3:
    #     assert 0
    coor = coor / jnp.linalg.norm(coor, axis=-1, keepdims=True)
    mu_unnorm = asgt.T @ coor
    asgt_sum = asgt.sum(-2, keepdims=True)
    rbar = jnp.linalg.norm(mu_unnorm, -2) / asgt_sum
    kappa = init(rbar)
    for _ in range(max_iter):
        kappa = step(kappa, rbar)
    return kappa.squeeze(), mu_unnorm / asgt_sum.T


def doublet_energy(
    doublet,
    D: Tensor,
    Q: Tensor,
) -> Tensor:
    """
    Compute energy of a distribution Q for doublets in D.
    """
    coassignment = jnp.einsum('...snd,...sd->...sn', Q[..., D, :], Q)
    result = -doublet.log_prob(coassignment)
    return jnp.where(D >= 0, result, 0)


def param_bimodal_beta(n_classes: int):
    """Bimodal beta distribution with a minimum at the maximum entropy"""
    alpha = jnp.log(n_classes) / (
        jnp.log(n_classes) - jnp.log(1 - 1 / n_classes)
    )
    return Beta(alpha, 1 - alpha)


def compute_dccc(
    atlas: CortexSubcortexGIfTIAtlas,
    data: jnp.ndarray,
    log_prob: jnp.ndarray,
    name: str,
    fname: str,
    method: Literal['argmax', 'softmax'] = 'argmax',
):
    match method:
        case 'argmax':
            dccc_assignment = jnp.eye(log_prob.shape[-1])[log_prob.argmax(-1)]
        case 'softmax':
            dccc_assignment = jax.nn.softmax(log_prob, -1)
    coors_L = atlas.coors[atlas.compartments.compartments['cortex_L'].mask_array]
    order = jax.random.permutation(
        jax.random.PRNGKey(DCCC_SAMPLE_KEY),
        len(dccc_assignment),
    )[slice(0, DCCC_NUM_NODES)]
    result = dccc(
        coors=coors_L[order],
        features=data[:len(dccc_assignment)][order],
        assignment=dccc_assignment[order],
        integral_samples=jnp.arange(40) * 0.9,
    )
    dccc_result = result.get('dccc')
    print(f'DCCC {name}: {dccc_result.sum()}')
    plot_dccc(result, f'/tmp/dccc_{fname}.png')
    return result


def create_template():
    template = np.load('/tmp/mean_init.npy')
    spatial_loc_left, spatial_loc_right, spatial_data = init_spatial_priors()
    template = template / jnp.linalg.norm(template, axis=-1, keepdims=True)
    template_left = template[:spatial_loc_left.shape[0]]
    template_right = template[spatial_loc_left.shape[0]:]

    atlas_path = get_atlas_path(ATLAS_NAME)
    atlas = CortexSubcortexGIfTIAtlas(
        data_L=atlas_path['L'],
        data_R=atlas_path['R'],
        name='parcellation',
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

    log_prob_L = (
        temporal_L.log_prob(template_left) +
        spatial_L.log_prob(
            atlas.coors[atlas.compartments.compartments['cortex_L'].mask_array]
        )
    )
    log_prob_R = (
        temporal_R.log_prob(template_right) +
        spatial_R.log_prob(
            atlas.coors[atlas.compartments.compartments['cortex_R'].mask_array]
        )
    )
    return (
        atlas,
        encoder,
        template_left,
        template_right,
        spatial_loc_left,
        spatial_loc_right,
        spatial_data,
        temporal_L,
        temporal_R,
        spatial_L,
        spatial_R,
        log_prob_L,
        log_prob_R,
    )


def single_subject(
    subject: str,
    session: str,
    atlas: CortexSubcortexGIfTIAtlas,
    encoder: AtlasLinear,
    template_left: jnp.ndarray,
    template_right: jnp.ndarray,
    spatial_loc_left: jnp.ndarray,
    spatial_loc_right: jnp.ndarray,
    spatial_data: jnp.ndarray,
    temporal_L: VonMisesFisher,
    temporal_R: VonMisesFisher,
    spatial_L: VonMisesFisher,
    spatial_R: VonMisesFisher,
    log_prob_L: jnp.ndarray,
    log_prob_R: jnp.ndarray,
    spatial_prior_weight: Optional[float] = SPATIAL_PRIOR_WEIGHT,
):
    dccc_result = {}
    msc = _get_data(get_msc_dataset(subject, session))
    enc = encoder(msc, encode=True, decode_labels=False)

    dccc_result['template'] = compute_dccc(
        atlas,
        msc,
        log_prob_L,
        f'template (MSC {subject} {session})',
        f'{subject}_{session}_template_L',
    )

    enc['cortex_L'], _ = whiten_data(enc['cortex_L'])
    enc['cortex_R'], _ = whiten_data(enc['cortex_R'])
    enc['cortex_L'] = enc['cortex_L'] / jnp.linalg.norm(
        enc['cortex_L'], axis=-1, keepdims=True
    )
    enc['cortex_R'] = enc['cortex_R'] / jnp.linalg.norm(
        enc['cortex_R'], axis=-1, keepdims=True
    )
    cortex_L_orig, cortex_R_orig = enc['cortex_L'], enc['cortex_R']
    spatial_data_left = spatial_data
    spatial_data_right = spatial_data

    print('Aligning left hemisphere to mean template...')
    enc['cortex_L'], _ = empty_promises(
        X=enc['cortex_L'],
        M=template_left,
        spatial_prior_loc=spatial_loc_left,
        spatial_prior_data=spatial_data_left,
        spatial_prior_weight=spatial_prior_weight,
        return_loading=True,
    )
    print(jnp.linalg.norm(template_left - enc['cortex_L']))
    print('Aligning right hemisphere to mean template...')
    enc['cortex_R'], _ = empty_promises(
        X=enc['cortex_R'],
        M=template_right,
        spatial_prior_loc=spatial_loc_right,
        spatial_prior_data=spatial_data_right,
        spatial_prior_weight=spatial_prior_weight,
        return_loading=True,
    )
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
    visualise(
        log_prob_L=log_prob_L,
        log_prob_R=log_prob_R,
        name=f'MSC_{subject}_{session}_aligned',
    )
    print(
        'Total parcels detected (left hemisphere):',
        len(jnp.unique(log_prob_L.argmax(-1)))
    )
    print(
        'Total parcels detected (right hemisphere):',
        len(jnp.unique(log_prob_R.argmax(-1)))
    )
    dccc_result['individual'] = compute_dccc(
        atlas,
        msc,
        log_prob_L,
        f'subject-specific (MSC {subject} {session})',
        f'{subject}_{session}_individual_L',
    )
    atlas.to_gifti(
        f'/tmp/atlas_MSC_sub-MSC{subject}_ses-{session}_individual',
        maps={'cortex_L': log_prob_L, 'cortex_R': log_prob_R},
        discretise=False,
    )

    #asgt_L = jnp.eye(log_prob_L.shape[-1])[log_prob_L.argmax(-1)]
    asgt_L = jax.nn.softmax(log_prob_L / MLE_TEMPERATURE, -1)
    #asgt_R = jnp.eye(log_prob_R.shape[-1])[log_prob_R.argmax(-1)]
    asgt_R = jax.nn.softmax(log_prob_R / MLE_TEMPERATURE, -1)

    # temporal_kappa_mle_L, temporal_mu_mle_L = kappa_iteration(
    #     cortex_L_orig, asgt_L
    # )
    # # Don't iterate for spatial because the log Bessel approximation is not
    # # stable for low dimensions. That means our normalisation constant is
    # # also wrong, so we'll have to fix that.
    # spatial_kappa_mle_L, spatial_mu_mle_L = kappa_iteration(
    #     atlas.coors[:enc['cortex_L'].shape[0]], asgt_L, max_iter=0
    # )
    # temporal_kappa_mle_R, temporal_mu_mle_R = kappa_iteration(
    #     cortex_R_orig, asgt_R
    # )
    # spatial_kappa_mle_R, spatial_mu_mle_R = kappa_iteration(
    #     atlas.coors[enc['cortex_L'].shape[0]:], asgt_R, max_iter=0
    # )
    # temporal_kappa_mle_L = map_into(temporal_kappa_mle_L, TEMPORAL_KAPPA_LIM)
    # temporal_kappa_mle_R = map_into(temporal_kappa_mle_R, TEMPORAL_KAPPA_LIM)
    # spatial_kappa_mle_L = map_into(spatial_kappa_mle_L, SPATIAL_KAPPA_LIM)
    # spatial_kappa_mle_R = map_into(spatial_kappa_mle_R, SPATIAL_KAPPA_LIM)
    # temporal_mle_L = VonMisesFisher(mu=temporal_mu_mle_L, kappa=temporal_kappa_mle_L)
    # temporal_mle_R = VonMisesFisher(mu=temporal_mu_mle_R, kappa=temporal_kappa_mle_R)
    # spatial_mle_L = VonMisesFisher(mu=spatial_mu_mle_L, kappa=spatial_kappa_mle_L)
    # spatial_mle_R = VonMisesFisher(mu=spatial_mu_mle_R, kappa=spatial_kappa_mle_R)

    temporal_mu_mle_L = asgt_L.T @ cortex_L_orig
    temporal_mu_mle_L = temporal_mu_mle_L / jnp.linalg.norm(
        temporal_mu_mle_L, axis=-1, keepdims=True
    )
    spatial_mu_mle_L = asgt_L.T @ atlas.coors[:enc['cortex_L'].shape[0]]
    spatial_mu_mle_L = spatial_mu_mle_L / jnp.linalg.norm(
        spatial_mu_mle_L, axis=-1, keepdims=True
    )
    temporal_mu_mle_R = asgt_R.T @ cortex_R_orig
    temporal_mu_mle_R = temporal_mu_mle_R / jnp.linalg.norm(
        temporal_mu_mle_R, axis=-1, keepdims=True
    )
    spatial_mu_mle_R = asgt_R.T @ atlas.coors[enc['cortex_L'].shape[0]:]
    spatial_mu_mle_R = spatial_mu_mle_R / jnp.linalg.norm(
        spatial_mu_mle_R, axis=-1, keepdims=True
    )
    temporal_mle_L = VonMisesFisher(mu=temporal_mu_mle_L, kappa=10)
    temporal_mle_R = VonMisesFisher(mu=temporal_mu_mle_R, kappa=10)
    spatial_mle_L = VonMisesFisher(mu=spatial_mu_mle_L, kappa=10)
    spatial_mle_R = VonMisesFisher(mu=spatial_mu_mle_R, kappa=10)

    log_prob_mle_L = (
        temporal_mle_L.log_prob(cortex_L_orig) +
        spatial_mle_L.log_prob(atlas.coors[:enc['cortex_L'].shape[0]])
    )
    log_prob_mle_R = (
        temporal_mle_R.log_prob(cortex_R_orig) +
        spatial_mle_R.log_prob(atlas.coors[enc['cortex_L'].shape[0]:])
    )
    visualise(
        log_prob_L=log_prob_mle_L,
        log_prob_R=log_prob_mle_R,
        name=f'MSC_{subject}_{session}_MLE',
    )
    print(
        'Total parcels detected (left hemisphere):',
        len(jnp.unique(log_prob_mle_L.argmax(-1)))
    )
    print(
        'Total parcels detected (right hemisphere):',
        len(jnp.unique(log_prob_mle_R.argmax(-1)))
    )
    dccc_result['mle'] = compute_dccc(
        atlas,
        msc,
        log_prob_mle_L,
        f'subject-specific MLE (MSC {subject} {session})',
        f'{subject}_{session}_mle_L',
    )
    atlas.to_gifti(
        f'/tmp/atlas_MSC_sub-MSC{subject}_ses-{session}_mle',
        maps={'cortex_L': log_prob_mle_L, 'cortex_R': log_prob_mle_R},
        discretise=False,
    )

    parcel_count = log_prob_L.shape[-1]
    doublet = param_bimodal_beta(jnp.sqrt(parcel_count))
    energy_aligned_L = doublet_energy(
        doublet, spatial_loc_left, jax.nn.softmax(log_prob_L, -1)
    ).sum(-1) + (
        temporal_L.log_prob(cortex_L_orig) + spatial_L.log_prob(
            atlas.coors[:enc['cortex_L'].shape[0]]
        )
    ).mean(-1)
    energy_aligned_R = doublet_energy(
        doublet, spatial_loc_right, jax.nn.softmax(log_prob_R, -1)
    ).sum(-1) + (
        temporal_R.log_prob(cortex_R_orig) + spatial_R.log_prob(
            atlas.coors[enc['cortex_L'].shape[0]:]
        )
    ).mean(-1)
    log_prob_energy_L = log_prob_mle_L
    log_prob_energy_R = log_prob_mle_R
    for iter in range(MAX_ITER_ENERGY_EQUILIBRIUM):
        print(f'Energy relaxation iteration {iter + 1}...')
        energy_energy_L = doublet_energy(
            doublet, spatial_loc_left, jax.nn.softmax(log_prob_energy_L, -1)
        ).sum(-1) + (
            temporal_mle_L.log_prob(cortex_L_orig) + spatial_mle_L.log_prob(
                atlas.coors[:enc['cortex_L'].shape[0]]
            )
        ).mean(-1)
        energy_energy_R = doublet_energy(
            doublet, spatial_loc_right, jax.nn.softmax(log_prob_energy_R, -1)
        ).sum(-1) + (
            temporal_mle_R.log_prob(cortex_R_orig) + spatial_mle_R.log_prob(
                atlas.coors[enc['cortex_L'].shape[0]:]
            )
        ).mean(-1)
        log_prob_energy_L = jnp.where(
            (energy_aligned_L < energy_energy_L)[..., None], log_prob_L, log_prob_energy_L
        )
        log_prob_energy_R = jnp.where(
            (energy_aligned_R < energy_energy_R)[..., None], log_prob_R, log_prob_energy_R
        )
    visualise(
        log_prob_L=log_prob_energy_L,
        log_prob_R=log_prob_energy_R,
        name=f'MSC_{subject}_{session}_energy',
    )
    print(
        'Total parcels detected (left hemisphere):',
        len(jnp.unique(log_prob_energy_L.argmax(-1)))
    )
    print(
        'Total parcels detected (right hemisphere):',
        len(jnp.unique(log_prob_energy_R.argmax(-1)))
    )
    dccc_result['energy'] = compute_dccc(
        atlas,
        msc,
        log_prob_energy_L,
        f'subject-specific energy min (MSC {subject} {session})',
        f'{subject}_{session}_energy_L',
    )
    atlas.to_gifti(
        f'/tmp/atlas_MSC_sub-MSC{subject}_ses-{session}_energy',
        maps={'cortex_L': log_prob_energy_L, 'cortex_R': log_prob_energy_R},
        discretise=False,
    )
    return dccc_result


def main():
    (
        atlas,
        encoder,
        template_left,
        template_right,
        spatial_loc_left,
        spatial_loc_right,
        spatial_data,
        temporal_L,
        temporal_R,
        spatial_L,
        spatial_R,
        log_prob_L,
        log_prob_R,
    ) = create_template()
    dccc_result = {}
    for subject in SUBJECTS:
        dccc_result[subject] = {}
        for session in SESSIONS:
            dccc_result[subject][session] = single_subject(
                subject,
                session,
                atlas=atlas,
                encoder=encoder,
                template_left=template_left,
                template_right=template_right,
                spatial_loc_left=spatial_loc_left,
                spatial_loc_right=spatial_loc_right,
                spatial_data=spatial_data,
                temporal_L=temporal_L,
                temporal_R=temporal_R,
                spatial_L=spatial_L,
                spatial_R=spatial_R,
                log_prob_L=log_prob_L,
                log_prob_R=log_prob_R,
                spatial_prior_weight=SPATIAL_PRIOR_WEIGHT,
            )
        with open(f'/tmp/msc_dccc_{subject}.pkl', 'wb') as handle:
            checkpoint = {**dccc_result[subject]}
            pickle.dump(checkpoint, handle, protocol=pickle.HIGHEST_PROTOCOL)
    assert 0


if __name__ == '__main__':
    main()
