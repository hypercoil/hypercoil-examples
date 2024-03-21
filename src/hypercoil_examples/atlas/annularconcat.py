# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Concatenated annular projections
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Group-level annular projections of concatenated data
"""
import pickle

import equinox as eqx
import jax
import jax.numpy as jnp
import optax
from jax import config

from hypercoil.engine import _to_jax_array
from hypercoil.functional import complex_decompose
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

NUM_MAPS = 32
REFERENCE_ORIGIN = 2
REFERENCE_POLARITY = 12
SUBJECTS = ('01', '02', '03', '04', '05', '06', '07', '08', '09', '10')
SESSIONS = ('01', '02', '03', '04', '05') #, '06', '07', '08', '09', '10')

MAX_EPOCH = 200
LEARNING_RATE = 1e-3
PHASE_ENTROPY_NU = 1e0
MEAN_ABS_NU = 1e0
TOTAL_DETERMINANT_NU = 1e-2
RECONSTRUCTION_ERROR_NU = 5e-1
KEY = 47


def main():
    atlas = ENCODER_FACTORY['consensus']()
    model = AtlasLinear.from_atlas(
        atlas, encode=True, key=jax.random.PRNGKey(0)
    )
    cortex_size = (
        model.weight['cortex_L'].shape[0] +
        model.weight['cortex_R'].shape[0]
    )

    X = []
    for subject in SUBJECTS:
        for session in SESSIONS:
            print(f'Processing subject {subject} session {session}')
            data = _get_data(get_msc_dataset(subject, session))
            enc = model(data)
            enc = jnp.concatenate((enc['cortex_L'], enc['cortex_R']))
            loc = enc[..., :cortex_size]
            scale = jnp.minimum(-2e-2 * jnp.log(jnp.abs(loc)), 5)
            X += [
                logistic_mixture_threshold(
                    incomplete_mahalanobis_transform(loc)[0],
                    scale,
                    k=0.9,
                    axis=-1,
                )
            ]
    X = jnp.concatenate(X, -1)
    parcels = model(X, encode=False)[:cortex_size]
    ref_ori = parcels[REFERENCE_ORIGIN]
    ref_pol = parcels[REFERENCE_POLARITY]

    _, Q = jnp.linalg.eigh(jnp.cov(X.T))
    proj = Q[..., -(2 * NUM_MAPS):].reshape(
        (Q.shape[0], NUM_MAPS, 2)
    ).swapaxes(0, 1)
    proj_vec = jnp.reshape(proj.swapaxes(0, 1), (proj.shape[1], -1)).T
    # Using PCA, we should get the best possible reconstruction error
    best_reconstruction_error = (reconstruction_error(
        X / jnp.linalg.norm(X, axis=-1, keepdims=True),
        proj_vec,
    ) ** 2).mean()
    (
        projection,
        distr,
        phase_entropy,
        mean_abs,
        total_angle_matrix,
    ) = annular_projection(
        X / jnp.linalg.norm(X, axis=-1, keepdims=True),
        proj,
    )
    projection = align_projection_phase(
        projection=projection,
        projector=proj,
        ref_ori=ref_ori,
        ref_pol=ref_pol,
    )

    ampl, phase = complex_decompose(projection)
    assert jnp.allclose(
        total_amplitude(
            X / jnp.linalg.norm(X, axis=-1, keepdims=True), proj_vec[-2:]
        ),
        ampl[-1],
        atol=5e-5,
    )
    plot_annular_projection(
        ampl, phase, projection, distr, '/tmp/concatpca_projection.png'
    )
    print(
        phase_entropy, mean_abs, total_angle_matrix,
        jnp.linalg.slogdet(total_angle_matrix),
        best_reconstruction_error,
    )
    plot_f = configure_plot()
    plot_total_ampl = configure_plot(num_maps=1)
    plot_f(
        template='fsLR',
        load_mask=True,
        projection_array=phase,
        surf_scalars_cmap='twilight',
        surf_projection=('veryinflated',),
        hemisphere=['left', 'right', None],
        views={
            'left': ('medial', 'lateral'),
            'right': ('medial', 'lateral'),
            'both': ('dorsal', 'ventral', 'anterior', 'posterior'),
        },
        output_dir='/tmp',
        fname_spec=f'scalars-phase_proj-concatpca_count-{NUM_MAPS:02d}',
        window_size=(800, 600),
    )
    plot_f(
        template='fsLR',
        load_mask=True,
        projection_array=ampl,
        surf_scalars_cmap='inferno',
        surf_scalars_clim=(0.15, 0.3),
        surf_projection=('veryinflated',),
        hemisphere=['left', 'right', None],
        views={
            'left': ('medial', 'lateral'),
            'right': ('medial', 'lateral'),
            'both': ('dorsal', 'ventral', 'anterior', 'posterior'),
        },
        output_dir='/tmp',
        fname_spec=f'scalars-ampl_proj-concatpca_count-{NUM_MAPS:02d}',
        window_size=(800, 600),
    )
    plot_total_ampl(
        template='fsLR',
        load_mask=True,
        projection_array=total_amplitude(
            X / jnp.linalg.norm(X, axis=-1, keepdims=True), proj_vec
        ),
        surf_scalars_cmap='inferno',
        surf_scalars_clim=(0., .7),
        surf_projection=('veryinflated',),
        hemisphere=['left', 'right', None],
        views={
            'left': ('medial', 'lateral'),
            'right': ('medial', 'lateral'),
            'both': ('dorsal', 'ventral', 'anterior', 'posterior'),
        },
        output_dir='/tmp',
        fname_spec=f'scalars-totalampl_proj-concatpca_count-{NUM_MAPS:02d}',
        window_size=(800, 600),
    )

    config.update("jax_debug_nans", True)
    forward = cfg_forward_pass(
        phase_entropy_nu=PHASE_ENTROPY_NU,
        mean_abs_nu=MEAN_ABS_NU,
        total_determinant_nu=TOTAL_DETERMINANT_NU,
        reconstruction_error_nu=RECONSTRUCTION_ERROR_NU,
    )
    # proj = jax.random.normal(
    #     key=jax.random.PRNGKey(47),
    #     shape=(X.shape[-1], 2)
    # )
    proj = Q[..., -(2 * NUM_MAPS):].reshape((Q.shape[0], NUM_MAPS, 2)).swapaxes(0, 1)
    model = AnnularProjection(
        proj=proj,
        kernel=gauss_kernel(jnp.pi / 16),
        n_samples=100,
    )
    model = OrthogonalParameter.map(
        model=model,
        where='proj',
    )
    X = X / jnp.linalg.norm(X, axis=-1, keepdims=True)
    key = jax.random.PRNGKey(KEY)
    opt = optax.adam(learning_rate=LEARNING_RATE)
    opt_state = opt.init(eqx.filter(model, eqx.is_inexact_array))
    losses = []

    for epoch in range(MAX_EPOCH):
        loss, grad = eqx.filter_jit(eqx.filter_value_and_grad(forward))(
            model, X, psi=1e-5, xi=8e-6,
            key=jax.random.fold_in(key=key, data=epoch)
        )
        losses += [loss]
        print(loss)
        updates, opt_state = opt.update(
            eqx.filter(grad, eqx.is_inexact_array),
            opt_state,
            eqx.filter(model, eqx.is_inexact_array),
        )
        model = eqx.apply_updates(model, updates)

    proj = _to_jax_array(model.proj)
    proj_vec = jnp.reshape(proj.swapaxes(0, 1), (proj.shape[1], -1)).T
    annular_reconstruction_error = (reconstruction_error(X, proj_vec) ** 2).mean()
    (
        projection,
        distr,
        phase_entropy,
        mean_abs,
        total_angle_matrix,
    ) = annular_projection(X, proj)
    projection = align_projection_phase(
        projection=projection,
        projector=proj,
        ref_ori=ref_ori,
        ref_pol=ref_pol,
    )
    ampl, phase = complex_decompose(projection)
    plot_annular_projection(
        ampl, phase, projection, distr, '/tmp/concatannular_projection.png'
    )
    print(
        phase_entropy, mean_abs, total_angle_matrix,
        jnp.linalg.slogdet(total_angle_matrix),
        annular_reconstruction_error,
    )
    plot_f(
        template='fsLR',
        load_mask=True,
        projection_array=phase,
        surf_scalars_cmap='twilight',
        surf_projection=('veryinflated',),
        hemisphere=['left', 'right', None],
        views={
            'left': ('medial', 'lateral'),
            'right': ('medial', 'lateral'),
            'both': ('dorsal', 'ventral', 'anterior', 'posterior'),
        },
        output_dir='/tmp',
        fname_spec=f'scalars-phase_proj-concatannular_count-{NUM_MAPS:02d}',
        window_size=(800, 600),
    )
    plot_f(
        template='fsLR',
        load_mask=True,
        projection_array=ampl,
        surf_scalars_cmap='inferno',
        surf_scalars_clim=(0.15, 0.3),
        surf_projection=('veryinflated',),
        hemisphere=['left', 'right', None],
        views={
            'left': ('medial', 'lateral'),
            'right': ('medial', 'lateral'),
            'both': ('dorsal', 'ventral', 'anterior', 'posterior'),
        },
        output_dir='/tmp',
        fname_spec=f'scalars-ampl_proj-concatannular_count-{NUM_MAPS:02d}',
        window_size=(800, 600),
    )
    plot_total_ampl(
        template='fsLR',
        load_mask=True,
        projection_array=total_amplitude(
            X / jnp.linalg.norm(X, axis=-1, keepdims=True), proj_vec
        ),
        surf_scalars_cmap='inferno',
        surf_scalars_clim=(0., .7),
        surf_projection=('veryinflated',),
        hemisphere=['left', 'right', None],
        views={
            'left': ('medial', 'lateral'),
            'right': ('medial', 'lateral'),
            'both': ('dorsal', 'ventral', 'anterior', 'posterior'),
        },
        output_dir='/tmp',
        fname_spec=f'scalars-totalampl_proj-concatannular_count-{NUM_MAPS:02d}',
        window_size=(800, 600),
    )
    with open(f'/tmp/concatannular.pkl', 'wb') as handle:
        pickle.dump({
            'projection': projection,
            'projector': proj,
            'distr': distr,
            'phase_entropy': phase_entropy,
            'mean_abs': mean_abs,
            'total_angle_matrix': total_angle_matrix,
            'annular_reconstruction_error': annular_reconstruction_error,
            'ref_ori': ref_ori,
            'ref_pol': ref_pol,
            'pca_projector': (
                Q[..., -(2 * NUM_MAPS):].reshape((Q.shape[0], NUM_MAPS, 2)).swapaxes(0, 1)
            ),
        }, handle, protocol=pickle.HIGHEST_PROTOCOL)
    assert 0


if __name__ == '__main__':
    main()
