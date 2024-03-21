# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Annular decomposition: subspace
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Backproject the annular decomposition to individual subjects.
"""
import pickle

import equinox as eqx
import jax
import jax.numpy as jnp
import optax
from jax import config

from hypercoil.engine import _to_jax_array, Tensor
from hypercoil.functional import complex_decompose, pairedcorr
from hypercoil.nn import AtlasLinear
from hypercoil_examples.atlas.aligned_dccc import (
    _get_data,
    get_msc_dataset,
)
from hypercoil_examples.atlas.annularproj import (
    annular_projection,
    align_projection_phase,
    cfg_forward_pass,
    configure_plot,
    gauss_kernel,
    plot_annular_projection,
    reconstruction_error,
    total_amplitude,
    AnnularProjection,
    OrthogonalParameter,
)
from hypercoil_examples.atlas.selectransform import (
    incomplete_mahalanobis_transform,
    logistic_mixture_threshold,
)
from hypercoil_examples.atlas.vmf import (
    ENCODER_FACTORY
)


CONCAT_PATH = (
    '/Users/rastkociric/Downloads/concatannular-10x5/concatannular.pkl'
)
SELECT_DIM = 64
NUM_MAPS = 32
REFERENCE_ORIGIN = 2
REFERENCE_POLARITY = 12
SUBJECTS = ('01', '02', '03', '04', '05', '06', '07', '08', '09', '10')
SESSIONS = ('01', '02', '03', '04', '05') #, '06', '07', '08', '09', '10')
THRESH = 0.025


def align_projection_phase_to_reference(
    projection: Tensor,
    projector: Tensor,
    ref_ori: Tensor,
    ref_proj: Tensor,
):
    ori_planar = (
        (ref_ori / jnp.linalg.norm(ref_ori, axis=-1)) @ projector
    )
    ori_proj = (
        ori_planar[..., 0] + ori_planar[..., 1] * 1j
    )
    ori_proj = ori_proj / jnp.abs(ori_proj)
    projection = projection / ori_proj[..., None]
    projection = jnp.where(
        (
            pairedcorr(
                jnp.angle(ref_proj)[:, None, :],
                jnp.angle(projection)[:, None, :]
            ) < 0
        ).squeeze()[:, None],
        projection.conj(),
        projection,
    )
    # projection = jnp.where(
    #     pairedcorr(jnp.angle(ref_proj)[:, None, :], projection[:, None, :].conj()) >
    #     pairedcorr(jnp.angle(ref_proj)[:, None, :], projection[:, None, :]),
    #     projection,
    #     projection.conj(),
    # )
    return projection


def main():
    with open(CONCAT_PATH, 'rb') as f:
        data = pickle.load(f)

    atlas = ENCODER_FACTORY['consensus']()
    model = AtlasLinear.from_atlas(
        atlas, encode=True, key=jax.random.PRNGKey(0)
    )
    cortex_size = (
        model.weight['cortex_L'].shape[0] +
        model.weight['cortex_R'].shape[0]
    )
    plot_f = configure_plot()

    projections = {}
    for i, subject in enumerate(SUBJECTS):
        projections[subject] = {}
        for j, session in enumerate(SESSIONS):
            print(f'Processing subject {subject} session {session}')
            index = (i * len(SESSIONS) + j) * SELECT_DIM
            proj = data['projector'][:, index:(index + SELECT_DIM), :]
            sub_data = _get_data(get_msc_dataset(subject, session))
            enc = model(sub_data)
            enc = jnp.concatenate((enc['cortex_L'], enc['cortex_R']))
            loc = enc[..., :cortex_size]
            scale = jnp.minimum(-2e-2 * jnp.log(jnp.abs(loc)), 5)
            X = logistic_mixture_threshold(
                incomplete_mahalanobis_transform(loc)[0],
                scale,
                k=0.9,
                axis=-1,
            )
            parcels = model(X, encode=False)[:cortex_size]
            ref_ori = parcels[REFERENCE_ORIGIN]
            (
                projection,
                distr,
                phase_entropy,
                mean_abs,
                total_angle_matrix,
            ) = annular_projection(X, proj)
            projection = align_projection_phase_to_reference(
                projection=projection,
                projector=proj,
                ref_ori=ref_ori,
                ref_proj=data['projection'],
            )
            if session == '01':
                ampl, phase = complex_decompose(projection)
                plot_annular_projection(
                    ampl, phase, projection, distr,
                    f'/tmp/sub-{subject}_ses-{session}_annular_projection.png',
                )
                plot_f(
                    template='fsLR',
                    load_mask=True,
                    projection_array=jnp.where(ampl < THRESH, jnp.nan, phase),
                    surf_scalars_cmap='twilight',
                    surf_projection=('veryinflated',),
                    hemisphere=['left', 'right', None],
                    views={
                        'left': ('medial', 'lateral'),
                        'right': ('medial', 'lateral'),
                        'both': ('dorsal', 'ventral', 'anterior', 'posterior'),
                    },
                    output_dir='/tmp',
                    fname_spec=f'sub-{subject}_ses-{session}_scalars-phase_proj-annular_count-{NUM_MAPS:02d}',
                    window_size=(800, 600),
                )
            projections[subject][session] = projection
    with open('/tmp/subjectspecificprojections.pkl', 'wb') as f:
        pickle.dump(projections, f)
    assert 0


if __name__ == '__main__':
    main()
