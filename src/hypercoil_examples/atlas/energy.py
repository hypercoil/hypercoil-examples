# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Energy visualisations
~~~~~~~~~~~~~~~~~~~~~
Visualisations of energy functions and their gradients on the brain surface.
"""
from functools import partial
from typing import Literal, Optional

import jax
import jax.numpy as jnp
import equinox as eqx
import lytemaps as nmaps
import matplotlib.pyplot as plt
import nibabel as nb
import pyvista as pv
import templateflow.api as tflow
from numpyro.distributions import Dirichlet

from hypercoil.init import (
    CortexSubcortexGIfTIAtlas,
    VonMisesFisher,
)
from hypercoil.engine import Tensor
from hypercoil.nn.atlas import AtlasLinear
from hyve import (
    Cell,
    ColGroupSpec,
    add_surface_overlay,
    draw_surface_boundary,
    plotdef,
    plot_to_display,
    plot_to_image,
    save_figure,
    surf_from_archive,
    surf_scalars_from_array,
    surf_scalars_from_gifti,
    text_element,
    vertex_to_face,
)

from hypercoil_examples.atlas.beta import BetaCompat
from hypercoil_examples.atlas.cross2subj import ATLAS_PATH
from hypercoil_examples.atlas.data import _get_data, get_msc_dataset
from hypercoil_examples.atlas.encoders import (
    visualise_surface_encoder,
    create_icosphere_encoder,
)
from hypercoil_examples.atlas.model import (
    logit_normal_divergence,
    logit_normal_maxent_divergence,
    marginal_entropy,
)
from hypercoil_examples.atlas.selectransform import (
    incomplete_mahalanobis_transform,
)
from hypercoil_examples.atlas.train import visdef
from hypercoil_examples.atlas.unet import get_meshes


def make_parcellation(
    subject: str = '01',
    session: str = '01',
):
    atlas = CortexSubcortexGIfTIAtlas(
        data_L=ATLAS_PATH['L'],
        data_R=ATLAS_PATH['R'],
        name='Glasser360',
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

    data = _get_data(*get_msc_dataset(subject, session, get_confounds=True))
    enc = encoder(data, encode=True, decode_labels=False)
    parcels = model(data, encode=False, decode_labels=False)

    selectivity_mu = encoder.enc(parcels, ref=data, decode_labels=False)
    spatial_mu = model(
        atlas.coors, forward_mode='map', encode=False, decode_labels=False
    )

    enc['cortex_L'], selectivity_mu_L = incomplete_mahalanobis_transform(
        enc['cortex_L'], selectivity_mu[:180]
    )
    enc['cortex_R'], selectivity_mu_R = incomplete_mahalanobis_transform(
        enc['cortex_R'], selectivity_mu[180:]
    )
    selectivity_L = VonMisesFisher(mu=selectivity_mu_L, kappa=10)
    selectivity_R = VonMisesFisher(mu=selectivity_mu_R, kappa=10)
    spatial_L = VonMisesFisher(mu=spatial_mu[:180], kappa=10)
    spatial_R = VonMisesFisher(mu=spatial_mu[180:], kappa=10)
    energy_L = (
        -selectivity_L.log_prob(enc['cortex_L']) -
        spatial_L.log_prob(atlas.coors[:enc['cortex_L'].shape[0]])
    )
    energy_R = (
        -selectivity_R.log_prob(enc['cortex_R']) -
        spatial_R.log_prob(atlas.coors[enc['cortex_L'].shape[0]:])
    )
    prob_L = jax.nn.softmax(-energy_L, axis=-1)
    prob_R = jax.nn.softmax(-energy_R, axis=-1)
    mesh_L, mesh_R = get_meshes()
    return (
        enc['cortex_L'], enc['cortex_R'],
        atlas.coors[:enc['cortex_L'].shape[0]],
        atlas.coors[enc['cortex_L'].shape[0]:],
        mesh_L.icospheres[0], mesh_R.icospheres[0],
        selectivity_L, selectivity_R, spatial_L, spatial_R, prob_L, prob_R
    )


def point_energy(
    Z: Tensor,
    S: Tensor,
    selectivity_distribution: VonMisesFisher,
    spatial_distribution: VonMisesFisher,
) -> Tensor:
    return -(
        selectivity_distribution.log_prob(Z) +
        spatial_distribution.log_prob(S)
    )


def doublet_energy(
    D: Tensor,
    Q: Tensor,
    doublet_potential: callable,
    doublet_temperature: float = 1.,
) -> Tensor:
    """
    Compute energy of a distribution Q for doublets in D.
    """
    if doublet_temperature != 1.:
        Q = jax.nn.softmax(jnp.log(Q) / doublet_temperature, axis=-1)
    coassignment = jnp.einsum(
        '...snd,...sd->...sn',
        Q[..., D, :],
        Q,
    )
    result = doublet_potential(coassignment)
    return jnp.where(D < 0, 0., result)


def expected_energy(
    Q: Tensor,
    U: Optional[Tensor] = None,
    D: Optional[Tensor] = None,
    *,
    doublet_potential: Optional[callable] = None,
    doublet_temperature: float = 1.,
    mass_potential: Optional[callable] = None,
    div_potential: Optional[callable] = None,
    calc_point_energy: bool = False,
) -> Tensor:
    energy = jnp.asarray(0)
    # point energies
    if calc_point_energy:
        energy += (Q * U).sum(-1)
    # doublet energies
    if doublet_potential is not None:
        energy += doublet_energy(
            D=D,
            Q=jnp.clip(Q, 1e-6, 1 - 1e-6),
            doublet_potential=doublet_potential,
            doublet_temperature=doublet_temperature,
        ).sum(-1)
    if mass_potential is not None:
        energy += mass_potential(Q.sum(-2))
    if div_potential is not None:
        energy += div_potential(Q)
    return energy

def energy(
    Z: Optional[Tensor] = None,
    S: Optional[Tensor] = None,
    D: Optional[Tensor] = None,
    *,
    selectivity_distribution: Optional[VonMisesFisher] = None,
    spatial_distribution: Optional[VonMisesFisher] = None,
    doublet_potential: callable,
    mass_potential: Optional[callable] = None,
    div_potential: Optional[callable] = None,
    calc_point_energy: bool = False,
    temperature: float = 1.,
    doublet_temperature: float = 1.,
) -> Tensor:
    U = None
    if (
        selectivity_distribution is not None and
        spatial_distribution is not None
    ):
        U = point_energy(
            Z=Z,
            S=S,
            spatial_distribution=spatial_distribution,
            selectivity_distribution=selectivity_distribution,
        )
        Q = jax.nn.softmax(-U / temperature, axis=-1)
    return expected_energy(
        Q=Q,
        U=U,
        D=D,
        doublet_potential=doublet_potential,
        doublet_temperature=doublet_temperature,
        mass_potential=mass_potential,
        div_potential=div_potential,
        calc_point_energy=calc_point_energy,
    )


class Energies(eqx.Module):
    U: Optional[Tensor] = None
    D: Optional[Tensor] = None

    @classmethod
    def from_coords(
        cls,
        Z: Optional[Tensor] = None,
        S: Optional[Tensor] = None,
        D: Optional[Tensor] = None,
        selectivity_distribution: Optional[VonMisesFisher] = None,
        spatial_distribution: Optional[VonMisesFisher] = None,
    ):
        U = point_energy(
            Z=Z,
            S=S,
            spatial_distribution=spatial_distribution,
            selectivity_distribution=selectivity_distribution,
        )
        return cls(U=U, D=D)
    
    @property
    def Q(self) -> Tensor:
        return jax.nn.softmax(-self.U, axis=-1)

    def __call__(
        self,
        *,
        doublet_potential: callable,
        mass_potential: Optional[callable] = None,
        div_potential: Optional[callable] = None,
        calc_point_energy: bool = False,
        temperature: float = 1.,
        doublet_temperature: float = 1.,
    ) -> Tensor:
        Q = jax.nn.softmax(-self.U / temperature, axis=-1)
        return expected_energy(
            U=self.U,
            Q=Q,
            D=self.D,
            doublet_potential=doublet_potential,
            mass_potential=mass_potential,
            div_potential=div_potential,
            calc_point_energy=calc_point_energy,
            doublet_temperature=doublet_temperature,
        )


def total_energy(
    energies: Energies,
    *,
    doublet_potential: callable,
    mass_potential: Optional[callable] = None,
    div_potential: Optional[callable] = None,
    calc_point_energy: bool = False,
    doublet_temperature: float = 1.,
) -> Tensor:
    return energies(
        doublet_potential=doublet_potential,
        mass_potential=mass_potential,
        div_potential=div_potential,
        calc_point_energy=calc_point_energy,
        doublet_temperature=doublet_temperature,
    ).sum()


def medial_wall_arrays():
    mwl = nb.load(str(
        tflow.get('fsLR', density='32k', hemi='L', desc='nomedialwall')
    )).darrays[0].data.astype(bool)
    mwr = nb.load(str(
        tflow.get('fsLR', density='32k', hemi='R', desc='nomedialwall')
    )).darrays[0].data.astype(bool)
    return ~mwl, ~mwr


def curv_arrays():
    curv = nmaps.datasets.fetch_fslr()['sulc']
    curvl = nb.load(str(curv.L))
    curvr = nb.load(str(curv.R))
    return curvl.darrays[0].data, curvr.darrays[0].data


# @partial(jax.custom_vjp, nondiff_argnums=(1, 2))
# def clip_pass_grad(x, max: float, min: float):
#     return jnp.clip(x, min, max)

# def f_fwd(x, max: float, min: float):
#     return clip_pass_grad(x, max=max, min=min), ()

# def f_bwd(min, max, res, g):
#     return (g,)

# clip_pass_grad.defvjp(f_fwd, f_bwd)


def div_energy_numerator(P, scale, axis: int | None = None):
    P = jnp.clip(P, 1e-5, 1 - 1e-5)
    return jnp.sum(jnp.log(P) * scale, axis=axis)


def div_energy(
    P,
    scale,
    denom: Tensor | None = None,
    axis: int | None = None,
) -> Tensor:
    num = div_energy_numerator(P, scale, axis=axis)
    #return jnp.exp(num)
    if denom is None:
        denom = num.mean()
        denom = jax.lax.stop_gradient(denom)
    # return jnp.exp(Dirichlet(jnp.asarray([scale] * 3)).log_prob(P))
    # return jnp.exp(-scale * (P * jnp.log(P))).sum(0)
    # return logit_normal_divergence(P, Q, scale).mean()
    return jnp.exp(num - denom)


def beta_plot():
    import matplotlib.pyplot as plt
    #ks = (0.01, 0.02, 0.05, 0.1, 0.2, 5, 10)
    plt.style.use('dark_background')
    x = jnp.linspace(0, 1, 100)

    ks = list(range(1, 10))
    fig, axs = plt.subplots(
        len(ks),
        1,
        figsize=(6, 3 * len(ks)),
        layout='tight',
    )
    for ax, k in zip(axs, ks):
        # if k < 1:
        #     a, b = 1, 1 / k
        # else:
        #     a, b = k, 1
        a, b = 10 - k, k
        y = jnp.exp(BetaCompat(a, b).log_prob(x))
        ax.plot(x, y)
        d = jax.vmap(
            jax.grad(lambda u: BetaCompat(a, b).log_prob(u)),
            in_axes=(-1,),
        )(x)
        axd = ax.twinx()
        axd.plot(x, d, ls='--', color='red')
        ax.set_xticks([])
        axd.set_xticks([])
        #ax.set_yticks([])
        #axd.set_yticks([])
    fig.savefig('/tmp/beta.png')


def ternary_plot_div_energy():
    import matplotlib.pyplot as plt
    import mpltern
    from mpltern.datasets import get_shanon_entropies

    t, l, r, _ = get_shanon_entropies()
    sample_points = jnp.stack((t, l, r))
    # sample_points = jnp.where(sample_points == 1, 1 - 1e-8, sample_points)
    # sample_points = jnp.where(sample_points == 0, 1e-8, sample_points)

    plt.style.use('dark_background')
    #ks = (0.1, 0.2, 0.5, 1, 2, 5, 10) # when using logit_normal_divergence
    #ks = (1.1, 1.2, 1.5, 2, 5, 10, 100) # when using dirichlet
    #ks = (.001, .002, .005, .01, .02, .05, .1, .2, .5) # unnormalised Dirichlet
    ks = (.6, .7, .8, .9, 1, 2, 5, 10, 20) # unnormalised Dirichlet

    fig, axs = plt.subplots(
        len(ks),
        1,
        subplot_kw={'projection': 'ternary'},
        figsize=(3, 3 * len(ks)),
        layout='tight',
    )
    for ax, k in zip(axs, ks):
        #v = logit_normal_maxent_divergence(jnp.stack((t, l, r)), k).mean(-2)
        #v = jnp.exp(Dirichlet(jnp.asarray([k] * 3)).log_prob(sample_points.T))
        v = div_energy(sample_points, k, axis=-2)
        ax.tripcolor(t, l, r, v, cmap='magma', shading='gouraud')
        ax.tricontour(t, l, r, v, colors='k')
        ax.taxis.set_ticks([])
        ax.laxis.set_ticks([])
        ax.raxis.set_ticks([])
    fig.savefig('/tmp/ternaryUnnormDirichlet.png')

    fig, axs = plt.subplots(
        len(ks),
        1,
        subplot_kw={'projection': 'ternary'},
        figsize=(3, 3 * len(ks)),
        layout='tight',
    )
    for ax, k in zip(axs, ks):
        v = jax.vmap(
            jax.grad(div_energy),
            in_axes=(-1, None, None),
        )(
            sample_points,
            #jnp.asarray(1 / 3),
            k,
            div_energy_numerator(sample_points, k, axis=-2).mean(),
        ).T.mean(-2)
        ax.tripcolor(t, l, r, v, cmap='magma', shading='gouraud')
        ax.tricontour(t, l, r, v, colors='k')
        ax.taxis.set_ticks([])
        ax.laxis.set_ticks([])
        ax.raxis.set_ticks([])
    fig.savefig('/tmp/ternaryUnnormDirichletGrad.png')

    plt.close('all')


def vis_layout(scheme: Literal['row', 'spread'] = 'spread'):
    if scheme == 'spread':
        layout = Cell() / Cell() << (1 / 2)
        layout = layout | Cell() | layout << (1 / 3)
        v3, v4 = 'lateral', 'medial'
    elif scheme == 'row':
        layout = Cell() | Cell() | Cell() | Cell() | Cell() << (1 / 5)
        v3, v4 = 'medial', 'lateral'
    annotations = {
        0: dict(
            hemisphere='left',
            view='lateral',
        ),
        1: dict(
            hemisphere='left',
            view='medial',
        ),
        2: dict(view='dorsal'),
        3: dict(
            hemisphere='right',
            view=v3,
        ),
        4: dict(
            hemisphere='right',
            view=v4,
        ),
    }
    layout = layout.annotate(annotations)
    return layout


def visualise_point_energy(
    selectivity_L: Tensor,
    selectivity_R: Tensor,
    spatial_L: Tensor,
    spatial_R: Tensor,
    parcellation_L: Tensor,
    parcellation_R: Tensor,
    curv_L: Tensor,
    curv_R: Tensor,
    medialwall_L: Tensor,
    medialwall_R: Tensor,
    test_name: str,
):
    layout = Cell() | Cell() << (1 / 2)
    layout = layout * layout * layout
    layout = layout | Cell() << (8 / 9)
    # sblayout = Cell() | Cell() | Cell() << (1 / 3)
    # layout = layout | sblayout << (8 / 9)
    annotations = {
        0: dict(
            hemisphere='left',
            view='lateral',
        ),
        1: dict(
            hemisphere='left',
            view='medial',
        ),
        2: dict(view='anterior'),
        3: dict(view='dorsal'),
        4: dict(view='ventral'),
        5: dict(view='posterior'),
        6: dict(
            hemisphere='right',
            view='medial',
        ),
        7: dict(
            hemisphere='right',
            view='lateral',
        ),
        8: dict(
            elements=['scalar_bar'],
        ),
        # 8: dict(elements=[{'scalar_bar': ('spatial',)}]),
        # 9: dict(elements=[{'scalar_bar': ('selectivity',)}]),
        # 10: dict(elements=[{'scalar_bar': ('combined',)}]),
    }
    layout = layout.annotate(annotations)
    if spatial_L.ndim < 2:
        multiplier = 1
    else:
        multiplier = spatial_L.shape[-1]
    canvas_size = (3600, 400 * multiplier)
    plot_f = plotdef(
        surf_from_archive(),
        surf_scalars_from_array(
            'parcellation',
            plot=False,
            allow_multihemisphere=False,
        ),
        add_surface_overlay(
            'medialwall',
            surf_scalars_from_array('medialwall', is_masked=False),
            vertex_to_face('medialwall', interpolation='mode'),
        ),
        add_surface_overlay(
            'spatial',
            surf_scalars_from_array(
                'spatial',
                is_masked=True,
            ),
            vertex_to_face('spatial', interpolation='mean'),
        ),
        add_surface_overlay(
            'selectivity',
            surf_scalars_from_array(
                'selectivity',
                is_masked=True,
            ),
            vertex_to_face('selectivity', interpolation='mean'),
        ),
        add_surface_overlay(
            'combined',
            surf_scalars_from_array(
                'combined',
                is_masked=True,
            ),
            vertex_to_face('combined', interpolation='mean'),
        ),
        add_surface_overlay(
            'curv',
            surf_scalars_from_array('curv', is_masked=False),
            vertex_to_face('curv', interpolation='mean'),
        ),
        add_surface_overlay(
            'parcellation_boundary',
            draw_surface_boundary(
                'parcellation',
                'parcellation_boundary',
                #target_domain='vertex',
                target_domain='face',
                num_steps=0,
                v2f_interpolation='mode',
            ),
        ),
        plot_to_image(),
        save_figure(
            layout_kernel=layout,
            group_spec = [
                ColGroupSpec(
                    variable='surfscalars',
                ),
            ],
            padding=0,
            canvas_size=canvas_size,
            canvas_color=(0, 0, 0),
            fname_spec='scalars-{surfscalars}',
            scalar_bar_action='collect',
        ),
    )
    plot_f(
        template='fsLR',
        surf_projection='veryinflated',
        window_size=(600, 500),
        hemisphere=['left', 'right', 'both'],
        views={
            'left': ('medial', 'lateral'),
            'right': ('medial', 'lateral'),
            'both': ('dorsal', 'ventral', 'anterior', 'posterior'),
        },
        theme=pv.themes.DarkTheme(),
        output_dir='/tmp',
        fname_spec=f'scalars-point-energy{test_name}',
        load_mask=True,
        spatial_array_left=spatial_L,
        spatial_array_right=spatial_R,
        spatial_alpha=0.75,
        spatial_cmap='Reds_r',
        spatial_clim=(90, 99.9),
        spatial_clim_percentile=True,
        spatial_below_color=(0, 0, 0, 0),
        spatial_scalar_bar_style={
            'name': 'spatial energy',
            'orientation': 'v',
        },
        selectivity_array_left=selectivity_L,
        selectivity_array_right=selectivity_R,
        selectivity_alpha=0.75,
        selectivity_cmap='Blues_r',
        selectivity_clim=(93, 99.9),
        selectivity_clim_percentile=True,
        selectivity_below_color=(0, 0, 0, 0),
        selectivity_scalar_bar_style={
            'name': 'selectivity energy',
            'orientation': 'v',
        },
        combined_array_left=selectivity_L + spatial_L,
        combined_array_right=selectivity_R + spatial_R,
        combined_alpha=0.75,
        combined_cmap='Purples_r',
        combined_clim=(97, 99.9),
        combined_clim_percentile=True,
        combined_below_color=(0, 0, 0, 0),
        combined_scalar_bar_style={
            'name': 'total energy',
            'orientation': 'v',
        },
        curv_array_left=curv_L,
        curv_array_right=curv_R,
        curv_cmap='gray',
        curv_clim=(-5e-1, 5e-1),
        curv_alpha=0.3,
        parcellation_array_left=parcellation_L,
        parcellation_array_right=parcellation_R,
        parcellation_boundary_color='black',
        parcellation_boundary_alpha=0.7,
        medialwall_array_left=medialwall_L,
        medialwall_array_right=medialwall_R,
        medialwall_cmap='binary',
        medialwall_clim=(0.99, 1),
        medialwall_below_color=(0, 0, 0, 0),
    )


def visualise_energy(
    energy_L: Tensor,
    energy_R: Tensor,
    grad_energy_L: Tensor,
    grad_energy_R: Tensor,
    parcellation_L: Tensor,
    parcellation_R: Tensor,
    curv_L: Tensor,
    curv_R: Tensor,
    medialwall_L: Tensor,
    medialwall_R: Tensor,
    test_name: str,
    layout_scheme: Literal['row', 'spread'] = 'spread',
):
    layout = vis_layout(scheme=layout_scheme)
    if energy_L.ndim < 2:
        multiplier = 1
    else:
        multiplier = energy_L.shape[-1]
    if layout_scheme == 'spread':
        canvas_size = (1200, 440 * multiplier)
    elif layout_scheme == 'row':
        canvas_size = (1500, 300 * multiplier)
    plot_f = plotdef(
        surf_from_archive(),
        surf_scalars_from_array(
            'parcellation',
            plot=False,
            allow_multihemisphere=False,
        ),
        add_surface_overlay(
            'medialwall',
            surf_scalars_from_array('medialwall', is_masked=False),
            vertex_to_face('medialwall', interpolation='mode'),
        ),
        add_surface_overlay(
            'energy',
            surf_scalars_from_array(
                'energy',
                is_masked=True,
            ),
            vertex_to_face('energy', interpolation='mean'),
        ),
        add_surface_overlay(
            'curv',
            surf_scalars_from_array('curv', is_masked=False),
            vertex_to_face('curv', interpolation='mean'),
        ),
        add_surface_overlay(
            'parcellation_boundary',
            draw_surface_boundary(
                'parcellation',
                'parcellation_boundary',
                #target_domain='vertex',
                target_domain='face',
                num_steps=0,
                v2f_interpolation='mode',
            ),
        ),
        plot_to_image(),
        save_figure(
            layout_kernel=layout,
            group_spec = [
                ColGroupSpec(
                    variable='surfscalars',
                ),
            ],
            padding=0,
            canvas_size=canvas_size,
            canvas_color=(0, 0, 0),
            fname_spec='scalars-{surfscalars}',
            scalar_bar_action='collect',
        ),
    )
    for name in ('energy', 'grad_energy'):
        if name == 'energy':
            array_left = energy_L
            array_right = energy_R
            plot_params = {
                'energy_cmap': 'inferno',
            }
        else:
            array_left = grad_energy_L
            array_right = grad_energy_R
            plot_params = {
                'energy_cmap': 'YlOrBr_r',
                'energy_cmap_negative': 'Blues_r',
            }
        if not array_left.shape:
            continue
        plot_f(
            template='fsLR',
            surf_projection='veryinflated',
            window_size=(600, 500),
            hemisphere=['left', 'right', 'both'],
            views={
                'left': ('medial', 'lateral'),
                'right': ('medial', 'lateral'),
                'both': ('dorsal',),
            },
            theme=pv.themes.DarkTheme(),
            output_dir='/tmp',
            fname_spec=f'scalars-{test_name}{name}',
            load_mask=True,
            energy_array_left=array_left,
            energy_array_right=array_right,
            curv_array_left=curv_L,
            curv_array_right=curv_R,
            curv_cmap='gray',
            curv_clim=(-5e-1, 5e-1),
            curv_alpha=0.3,
            parcellation_array_left=parcellation_L,
            parcellation_array_right=parcellation_R,
            parcellation_boundary_color='black',
            parcellation_boundary_alpha=0.3,
            medialwall_array_left=medialwall_L,
            medialwall_array_right=medialwall_R,
            medialwall_cmap='binary',
            medialwall_clim=(0.99, 1),
            medialwall_below_color=(0, 0, 0, 0),
            **plot_params,
        )


def run_analysis(
    energies_L: Energies,
    energies_R: Energies,
    grad_energy: callable,
    doublet_potential: Optional[callable] = None,
    mass_potential: Optional[callable] = None,
    div_potential: Optional[callable] = None,
    doublet_temperature: float = 1.,
):
    energy_L = energies_L(
        doublet_potential=doublet_potential,
        mass_potential=mass_potential,
        div_potential=div_potential,
        doublet_temperature=doublet_temperature,
    )
    grad_energy_L = grad_energy(
        energies_L,
        doublet_potential=doublet_potential,
        mass_potential=mass_potential,
        div_potential=div_potential,
        doublet_temperature=doublet_temperature,
        calc_point_energy=False,
    )
    grad_energy_L = jax.vmap(lambda x, i: x[i])(
        grad_energy_L.U, energies_L.Q.argmax(-1)
    )
    energy_R = energies_R(
        doublet_potential=doublet_potential,
        mass_potential=mass_potential,
        div_potential=div_potential,
        doublet_temperature=doublet_temperature,
    )
    grad_energy_R = grad_energy(
        energies_R,
        doublet_potential=doublet_potential,
        mass_potential=mass_potential,
        div_potential=div_potential,
        doublet_temperature=doublet_temperature,
        calc_point_energy=False,
    )
    grad_energy_R = jax.vmap(lambda x, i: x[i])(
        grad_energy_R.U, energies_R.Q.argmax(-1)
    )
    return energy_L, grad_energy_L, energy_R, grad_energy_R


def main():
    (
        selectivity_L,
        selectivity_R,
        spatial_L,
        spatial_R,
        adj_L,
        adj_R,
        vmf_selectivity_L,
        vmf_selectivity_R,
        vmf_spatial_L,
        vmf_spatial_R,
        prob_L,
        prob_R,
    ) = make_parcellation()
    medialwall_L, medialwall_R = medial_wall_arrays()
    curv_L, curv_R = curv_arrays()
    common_vis_args = {
        'parcellation_L': prob_L.argmax(-1),
        'parcellation_R': prob_R.argmax(-1),
        'curv_L': curv_L,
        'curv_R': curv_R,
        'medialwall_L': medialwall_L,
        'medialwall_R': medialwall_R,
    }
    for index in (
        0, 7, 26, 38, 44, 52, 67, 71, 78, 80, 93,
        105, 111, 119, 129, 130, 144, 156, 163, 175,
    ):
        visualise_point_energy(
            selectivity_L=vmf_selectivity_L.log_prob(selectivity_L)[..., index],
            selectivity_R=vmf_selectivity_R.log_prob(selectivity_R)[..., index],
            spatial_L=vmf_spatial_L.log_prob(spatial_L)[..., index],
            spatial_R=vmf_spatial_R.log_prob(spatial_R)[..., index],
            **common_vis_args,
            test_name=f'point{index}',
        )
    plot_boundaries_f, plot_confidence_f = visdef()
    plot_boundaries_f(
        name=f'parcellation-boundaries',
        array_left=prob_L,
        array_right=prob_R,
    )
    plot_confidence_f(
        name=f'parcellation-confidence',
        array_left=prob_L,
        array_right=prob_R,
    )
    disaffiliation_penalty = 1.
    num_parcels = 180
    # doublet_potential = lambda x: logit_normal_divergence(
    #     x,
    #     1 / num_parcels,
    #     scale=5.,
    # ) - x * disaffiliation_penalty
    doublet_distribution = BetaCompat(10., 1.)
    doublet_potential = lambda x: -doublet_distribution.log_prob(x)
    mass_potential = lambda x: marginal_entropy(x)
    #div_potential = lambda x: logit_normal_maxent_divergence(x.T, 1.).mean(-2)
    grad_energy = eqx.filter_grad(total_energy)
    energies_L = Energies.from_coords(
        Z=selectivity_L,
        S=spatial_L,
        D=adj_L,
        selectivity_distribution=vmf_selectivity_L,
        spatial_distribution=vmf_spatial_L,
    )
    energies_R = Energies.from_coords(
        Z=selectivity_R,
        S=spatial_R,
        D=adj_R,
        selectivity_distribution=vmf_selectivity_R,
        spatial_distribution=vmf_spatial_R,
    )
    ternary_plot_div_energy()
    beta_plot()
    ks = range(1, 10)
    for k in ks:
        # a, b = max(1, k), max(1, 1 / k)
        a, b = k, 10 - k
        doublet_distribution = BetaCompat(a, b)
        doublet_potential = lambda x: -doublet_distribution.log_prob(x)
        energy_L, grad_energy_L, energy_R, grad_energy_R = run_analysis(
            energies_L,
            energies_R,
            grad_energy,
            doublet_potential=doublet_potential,
            mass_potential=None,
            div_potential=None,
        )
        visualise_energy(
            energy_L,
            energy_R,
            grad_energy_L,
            grad_energy_R,
            **common_vis_args,
            test_name=f'doublet{k}',
            layout_scheme='row',
        )
    ks = [1000, 500, 200, 100, 50, 20, 10, 5, 2, 1]
    for k in ks:
        div_potential = lambda x: div_energy(x, 1 / k, axis=-1)
        energy_L, grad_energy_L, energy_R, grad_energy_R = run_analysis(
            energies_L,
            energies_R,
            grad_energy,
            doublet_potential=None,
            mass_potential=None,
            div_potential=div_potential,
        )
        visualise_energy(
            energy_L,
            energy_R,
            grad_energy_L,
            grad_energy_R,
            **common_vis_args,
            test_name=f'div{k}',
        )
    energy_L, grad_energy_L, energy_R, grad_energy_R = run_analysis(
        energies_L,
        energies_R,
        grad_energy,
        doublet_potential=None,
        mass_potential=mass_potential,
        div_potential=None,
    )
    visualise_energy(
        energy_L,
        energy_R,
        grad_energy_L,
        grad_energy_R,
        **common_vis_args,
        test_name='mass',
    )
    assert 0


if __name__ == '__main__':
    main()
