# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Annular projections
~~~~~~~~~~~~~~~~~~~
The annular projection pursuit algorithm is a method to find the best
projection of a hyperspherical dataset onto a 2D plane or set of 2D planes.
"""
from typing import Optional, Tuple

import equinox as eqx
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import optax
from jax import config

from hypercoil.engine import Tensor, _to_jax_array
from hypercoil.functional import complex_decompose, recondition_eigenspaces
from hypercoil.init import OrthogonalParameter
from hypercoil.nn import AtlasLinear
from hypercoil_examples.atlas.vmf import (
    get_data, ENCODER_FACTORY
)
from hypercoil_examples.atlas.selectransform import (
    incomplete_mahalanobis_transform,
    logistic_mixture_threshold,
)
from hypercoil_examples.atlas.totalangle import inner_angles
from hyve import (
    plotdef,
    surf_from_archive,
    surf_scalars_from_array,
    plot_to_image,
    save_grid,
)

MAX_EPOCH = 200
LEARNING_RATE = 1e-3
PHASE_ENTROPY_NU = 1e0
MEAN_ABS_NU = 1e0
TOTAL_DETERMINANT_NU = 1e-2
RECONSTRUCTION_ERROR_NU = 5e-1
KEY = 47
NUM_MAPS = 32
REFERENCE_ORIGIN = 9 # visual cortex
REFERENCE_POLARITY = 0 # somatomotor cortex


def gauss_kernel(scale: float) -> callable:
    def _call(loc: Tensor, distances: Tensor) -> Tensor:
        angle = jnp.exp(jnp.angle(distances) * 1j)
        diff = (angle - loc)
        return jnp.exp(
            -jnp.real(
                jnp.abs(distances) * diff * jnp.conj(diff)
                / scale
            )
        )
    return _call


def annular_projection(
    X: Tensor,
    proj: Tensor,
    kernel: callable = gauss_kernel(jnp.pi / 64),
    n_samples: int = 100,
    psi: float = 0.,
    xi: float = 0.,
    *,
    key: Optional['jax.random.PRNGKey'] = None,
) -> Tuple[Tensor, Tensor, float, float]:
    """
    Annular projection pursuit call.
    """
    planar = X @ proj
    projection = planar[..., 0] + planar[..., 1] * 1j
    samples = jnp.exp(jnp.linspace(
        -jnp.pi,
        (n_samples - 2) * jnp.pi / n_samples,
        n_samples,
    ) * 1j)
    distr = kernel(samples, projection[..., None])
    distr = distr.sum(-2) / distr.sum((-2, -1))[..., None]
    phase_entropy = (-distr * jnp.log(distr)).sum(-1)
    mean_abs = jnp.mean(jnp.abs(projection), axis=-1)
    total_angle_matrix = jnp.prod(jnp.cos(
        inner_angles(None, Q=proj, psi=psi, xi=xi, key=key)
    ), axis=-1)
    return (
        projection,
        distr,
        phase_entropy,
        mean_abs,
        total_angle_matrix
    )


def align_projection_phase(
    projection: Tensor,
    projector: Tensor,
    ref_ori: Tensor,
    ref_pol: Tensor,
):
    ori_planar, pol_planar = (
        (ref_ori / jnp.linalg.norm(ref_ori, axis=-1)) @ projector,
        (ref_pol / jnp.linalg.norm(ref_pol, axis=-1)) @ projector,
    )
    ori_proj, pol_proj = (
        ori_planar[..., 0] + ori_planar[..., 1] * 1j,
        pol_planar[..., 0] + pol_planar[..., 1] * 1j,
    )
    ori_proj = ori_proj / jnp.abs(ori_proj)
    pol_proj = pol_proj / jnp.abs(pol_proj)
    projection = projection / ori_proj[..., None]
    projection = jnp.where(
        (jnp.angle(pol_proj / ori_proj) >= 0)[..., None],
        projection,
        projection.conj(),
    )
    return projection


def configure_plot(num_maps: int = NUM_MAPS) -> callable:
    plot_f = plotdef(
        surf_from_archive(),
        surf_scalars_from_array('projection', is_masked=True),
        plot_to_image(),
        save_grid(
            n_cols=8,
            n_rows=num_maps,
            padding=10,
            canvas_size=(3200, 300 * num_maps),
            canvas_color=(0, 0, 0),
            fname_spec=(
                f'scalars-projection_count-{num_maps:02d}'
            ),
            sort_by=['surfscalars'],
            scalar_bar_action='collect',
        ),
    )
    return plot_f


class AnnularProjection(eqx.Module):
    proj: Tensor
    kernel: callable = gauss_kernel(jnp.pi / 64)
    n_samples: int = 100

    def __call__(
        self,
        X: Tensor,
        psi: float = 0.,
        xi: float = 0.,
        *,
        key: Optional['jax.random.PRNGKey'] = None,
    ) -> Tuple[Tensor, Tensor, float, float]:
        return annular_projection(
            X=X,
            proj=_to_jax_array(self.proj),
            kernel=self.kernel,
            n_samples=self.n_samples,
            psi=psi,
            xi=xi,
            key=key,
        )


def cfg_forward_pass(
    phase_entropy_nu: float,
    mean_abs_nu: float,
    total_determinant_nu: float,
    reconstruction_error_nu: float = 0.,
) -> callable:
    def forward(
        model: AnnularProjection,
        X: Tensor,
        psi: float = 0.,
        xi: float = 0.,
        *,
        key: 'jax.random.PRNGKey',
    ) -> float:
        key_m, key_s = jax.random.split(key, 2)
        _, _, phase_entropy, mean_abs, total_angle_matrix = model(
            X,
            psi=psi,
            xi=xi,
            key=key_m,
        )
        loss = -(
            phase_entropy_nu * phase_entropy +
            mean_abs_nu * mean_abs +
            total_determinant_nu * jnp.linalg.slogdet(
                recondition_eigenspaces(
                    total_angle_matrix,
                    psi=psi,
                    xi=xi,
                    key=key_s,
                )
            )[1]
        ).mean()
        if reconstruction_error_nu > 0:
            proj = _to_jax_array(model.proj)
            proj_vec = jnp.reshape(proj.swapaxes(0, 1), (proj.shape[1], -1)).T
            loss += reconstruction_error_nu * (
                (reconstruction_error(X, proj_vec) ** 2).mean()
            )
        return loss
    return forward


def plot_annular_projection(
    ampl: Tensor,
    phase: Tensor,
    projection: Tensor,
    distr: Tensor,
    output_path: str,
):
    fig, ax = plt.subplots(NUM_MAPS, 3, figsize=(15, NUM_MAPS * 5), tight_layout=True)
    if NUM_MAPS == 1:
        ax = [ax]
    for i, (am, ph, pr, di) in enumerate(zip(ampl, phase, projection, distr)):
        ax[i][0].scatter(
            ph,
            am,
            s=0.01,
            c=ph,
            cmap='twilight',
        )
        ax[i][1].bar(range(len(di)), di)
        ax[i][2].scatter(
            jnp.real(pr),
            jnp.imag(pr),
            s=0.01,
            c=ph,
            cmap='twilight',
        )
    fig.savefig(output_path)


def reconstruction_error(
    X: Tensor,
    proj_vec: Tensor,
) -> Tensor:
    return jnp.linalg.norm(
        X - X @ jnp.linalg.pinv(proj_vec) @ proj_vec,
        axis=-1,
    )


def total_amplitude(
    X: Tensor,
    proj_vec: Tensor,
) -> Tensor:
    return jnp.linalg.norm(
        X @ jnp.linalg.pinv(proj_vec) @ proj_vec,
        axis=-1,
    )


def main():
    atlas = ENCODER_FACTORY['icosphere']()
    model = AtlasLinear.from_atlas(atlas, encode=True, key=jax.random.PRNGKey(0))
    data = get_data('MSC')
    plot_f = configure_plot()
    plot_total_ampl = configure_plot(num_maps=1)
    # coors, parcels_enc, atlas_coors = ENCODE_SELF['icosphere'](
    #     model=model, data=data, atlas=atlas
    # )
    enc = model(data)
    enc = jnp.concatenate((enc['cortex_L'], enc['cortex_R']))
    # parcels_enc = model(enc, encode=False)
    # ref_ori = parcels_enc[REFERENCE_ORIGIN]
    # ref_pol = parcels_enc[REFERENCE_POLARITY]
    # enc_aug = jnp.concatenate((enc, ref_ori[..., None, :], ref_pol[..., None, :]))
    # loc = enc_aug
    # loc = jnp.arctanh(jnp.clip(enc, -0.75, 0.75))
    loc = enc
    scale = jnp.minimum(-2e-2 * jnp.log(jnp.abs(loc)), 5)
    X = logistic_mixture_threshold(
        incomplete_mahalanobis_transform(loc)[0], scale, k=0.9, axis=-1
    )
    # ref_ori, ref_pol = X[-2], X[-1]
    # ref_ori = ref_ori / jnp.linalg.norm(ref_ori)
    # ref_pol = ref_pol / jnp.linalg.norm(ref_pol)
    # X = X[:-2]
    parcels_enc = model(X, encode=False)
    ref_ori = parcels_enc[REFERENCE_ORIGIN]
    ref_pol = parcels_enc[REFERENCE_POLARITY]
    # X = incomplete_mahalanobis_transform(enc)[0]

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
        ampl, phase, projection, distr, '/tmp/pca_projection.png'
    )
    print(
        phase_entropy, mean_abs, total_angle_matrix,
        jnp.linalg.slogdet(total_angle_matrix),
        best_reconstruction_error,
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
        fname_spec=f'scalars-phase_proj-pca_count-{NUM_MAPS:02d}',
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
        fname_spec=f'scalars-ampl_proj-pca_count-{NUM_MAPS:02d}',
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
        fname_spec=f'scalars-totalampl_proj-pca_count-{NUM_MAPS:02d}',
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
        ampl, phase, projection, distr, '/tmp/annular_projection.png'
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
        fname_spec=f'scalars-phase_proj-annular_count-{NUM_MAPS:02d}',
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
        fname_spec=f'scalars-ampl_proj-annular_count-{NUM_MAPS:02d}',
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
        fname_spec=f'scalars-totalampl_proj-annular_count-{NUM_MAPS:02d}',
        window_size=(800, 600),
    )


if __name__ == '__main__':
    main()
