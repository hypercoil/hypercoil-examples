# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Spatial nulls
~~~~~~~~~~~~~
Create a null distribution of spatial-only parcellations.
"""
from typing import Mapping, Tuple

import click
import equinox as eqx
import jax
import numpy as np
import optax
from pkg_resources import resource_filename
import templateflow.api as tflow

from hypercoil.engine import Tensor
from hypercoil.engine.paramutil import _to_jax_array
from hypercoil.functional import cmass_coor
from hypercoil.init import (
    DirichletInitSurfaceAtlas,
    ProbabilitySimplexParameter,
)
from hypercoil.loss import (
    CompactnessLoss,
    DispersionLoss,
    EntropyLoss,
    EquilibriumLoss,
    InterhemisphericTetherLoss,
    LossApply,
    LossArgument,
    LossScheme,
    UnpackingLossArgument,
    mean_scalarise,
    vnorm_scalarise,
)
from hypercoil.nn import AtlasLinear
from hyve.elements import TextBuilder
from hyve.flows import plotdef
from hyve.transforms import (
    add_surface_overlay,
    parcellate_colormap,
    plot_to_image,
    save_grid,
    surf_scalars_from_array,
    surf_scalars_from_cifti,
    surf_from_archive,
)


# TODO: Don't use print statements for logging.


OUT_ROOT = '/tmp'
ATLAS_TEMPLATE = 'fsLR' # TODO: This won't work with any other template.
LABELS_PER_HEMISPHERE = 200
DISTRIBUTION_SIZE = 1
RES_START = None
RES_STOP = None
RES_STEP = None
LR = 0.05
ENTROPY_NU = 5e-4 # 5e-4 # 5e-3
EQUILIBRIUM_NU = 5e3
COMPACTNESS_NU = 5. # 2.
DISPERSION_NU = 1e-5
TETHER_NU = 2e-2 # 1e-2
MAX_EPOCH = 1001
LOG_INTERVAL = 25
KEY = 0


def create_nulls(
    out_root: str = OUT_ROOT,
    atlas_template: str = ATLAS_TEMPLATE,
    label_counts: list = [LABELS_PER_HEMISPHERE],
    distribution_size: int = DISTRIBUTION_SIZE,
    lr: float = LR,
    entropy_nu: float = ENTROPY_NU,
    equilibrium_nu: float = EQUILIBRIUM_NU,
    compactness_nu: float = COMPACTNESS_NU,
    dispersion_nu: float = DISPERSION_NU,
    tether_nu: float = TETHER_NU,
    max_epoch: int = MAX_EPOCH,
    log_interval: int = LOG_INTERVAL,
    key: int = KEY,
) -> None:
    for labels_per_hemisphere in label_counts:
        n_labels = 2 * labels_per_hemisphere
        print(f'\n[ Null parcellation | Label count : {n_labels} ]')
        for iteration in range(distribution_size):
            print(f'\n[ Null parcellation | parcellation : {iteration} ]')
            atlas, model = configure_atlas(
                template=atlas_template,
                labels_per_hemisphere=labels_per_hemisphere,
                key=key,
                iteration=iteration,
            )
            lh_coor, rh_coor = get_coor(atlas)
            loss = configure_loss(
                entropy_nu=entropy_nu,
                equilibrium_nu=equilibrium_nu,
                compactness_nu=compactness_nu,
                dispersion_nu=dispersion_nu,
                tether_nu=tether_nu,
                lh_coor=lh_coor,
                rh_coor=rh_coor,
            )
            train_model(
                atlas=atlas,
                model=model,
                loss=loss,
                lr=lr,
                max_epoch=max_epoch,
                log_interval=log_interval,
                out_root=out_root,
                iteration=iteration,
                key=key,
            )



def configure_atlas(
    template: str = 'fsLR',
    labels_per_hemisphere: int = LABELS_PER_HEMISPHERE,
    key: int = KEY,
    iteration: int = 0,
) -> Tuple[DirichletInitSurfaceAtlas, AtlasLinear]:
    key_a, key_m = jax.random.split(jax.random.PRNGKey(key))
    atlas_basename = (
        f'desc-spatialnull_res-{labels_per_hemisphere:06}_iter-{iteration:06}'
    )
    atlas_name = f'{atlas_basename}_atlas'
    cifti_template = resource_filename(
        'hyve_examples',
        'data/nullexample.nii'
    )
    atlas = DirichletInitSurfaceAtlas(
        cifti_template=cifti_template,
        mask_L=tflow.get(
            template=template,
            hemi='L',
            desc='nomedialwall',
            density='32k'),
        mask_R=tflow.get(
            template=template,
            hemi='R',
            desc='nomedialwall',
            density='32k'),
        compartment_labels={
            'cortex_L': labels_per_hemisphere,
            'cortex_R': labels_per_hemisphere,
            'subcortex': 0,
        },
        name=atlas_name,
        key=key_a,
    )
    model = AtlasLinear.from_atlas(atlas=atlas, key=key_m)
    model = ProbabilitySimplexParameter.map(
        model,
        where='weight$(cortex_L;cortex_R)',
        axis=-2
    )
    return atlas, model


def get_coor(
    atlas: DirichletInitSurfaceAtlas,
) -> Tuple[Tensor, Tensor]:
    lh_coor = (
        atlas.compartments.get('cortex_L').map_to_masked()(atlas.coors)
    ).T
    rh_coor = (
        atlas.compartments.get('cortex_R').map_to_masked()(atlas.coors)
    ).T
    return lh_coor, rh_coor


def model_array(model: AtlasLinear, compartment: str) -> Tensor:
    return np.asarray(
        _to_jax_array(model.weight[compartment]).argmax(axis=0) + 1,
    )


def configure_loss(
    entropy_nu: float = ENTROPY_NU,
    equilibrium_nu: float = EQUILIBRIUM_NU,
    compactness_nu: float = COMPACTNESS_NU,
    dispersion_nu: float = DISPERSION_NU,
    tether_nu: float = TETHER_NU,
    lh_coor: Tensor = None,
    rh_coor: Tensor = None,
) -> LossScheme:
    return LossScheme((
        LossScheme((
            EntropyLoss(
                nu=entropy_nu,
                axis=-2,
                name='LeftHemisphereEntropy',
            ),
            EquilibriumLoss(
                nu=equilibrium_nu,
                name='LeftHemisphereEquilibrium',
            ),
            CompactnessLoss(
                nu=compactness_nu,
                coor=lh_coor,
                name='LeftHemisphereCompactness',
                radius=100,
            ),
        ), apply=lambda arg: arg.lh),
        LossScheme((
            EntropyLoss(
                nu=entropy_nu,
                axis=-2,
                name='RightHemisphereEntropy',
            ),
            EquilibriumLoss(
                nu=equilibrium_nu,
                name='RightHemisphereEquilibrium',
            ),
            CompactnessLoss(
                nu=compactness_nu,
                coor=rh_coor,
                name='RightHemisphereCompactness',
                radius=100,
            ),
        ), apply=lambda arg: arg.rh),
        LossApply(
            InterhemisphericTetherLoss(
                nu=tether_nu,
                scalarisation=mean_scalarise(inner=vnorm_scalarise(axis=0)),
                name='InterhemisphericTether',
            ),
            apply=lambda arg: UnpackingLossArgument(
                lh=arg.lh,
                rh=arg.rh,
                lh_coor=arg.lh_coor,
                rh_coor=arg.rh_coor,
            ),
        ),
        LossApply(
            DispersionLoss(
                nu=dispersion_nu,
                name='LeftHemisphereDispersion'
            ),
            apply=lambda arg: cmass_coor(arg.lh, arg.lh_coor, radius=100).T,
        ),
        LossApply(
            DispersionLoss(
                nu=dispersion_nu,
                name='RightHemisphereDispersion'
            ),
            apply=lambda arg: cmass_coor(arg.rh, arg.rh_coor, radius=100).T,
        ),
    ))


def configure_report(
    epoch: int,
    n_labels: int,
    iteration: int,
) -> callable:
    annotations = {
        0: dict(
            hemisphere='left',
            view='lateral',
        ),
        1: dict(view='anterior'),
        2: dict(
            hemisphere='right',
            view='lateral',
        ),
        3: dict(view='dorsal'),
        4: dict(elements=['title', 'label_count', 'iteration', 'epoch']),
        5: dict(view='ventral'),
        6: dict(
            hemisphere='left',
            view='medial',
        ),
        7: dict(view='posterior'),
        8: dict(
            hemisphere='right',
            view='medial',
        ),
    }
    plot_f = plotdef(
        surf_from_archive(),
        add_surface_overlay(
            'model',
            surf_scalars_from_array('model', allow_multihemisphere=False),
            parcellate_colormap('network', 'model'),
        ),
        plot_to_image(),
        save_grid(
            n_cols=3, n_rows=3, padding=10,
            canvas_size=(1800, 1500),
            canvas_color=(0, 0, 0),
            fname_spec=(
                f'scalars-null_view-all_labels-{n_labels}_'
                f'iter-{iteration:06}_epoch-{epoch:06}'
            ),
            scalar_bar_action='collect',
            annotations=annotations,
        ),
    )
    return plot_f


def forward(
    model: AtlasLinear,
    loss: LossScheme,
    lh_coor: Tensor,
    rh_coor: Tensor,
    *,
    key: 'jax.random.PRNGKey',
) -> Tuple[Tensor, Mapping]:
    arg = LossArgument(
        lh = _to_jax_array(model.weight['cortex_L']),
        rh = _to_jax_array(model.weight['cortex_R']),
        lh_coor = lh_coor,
        rh_coor = rh_coor,
    )
    return loss(arg, key=key)


def train_model(
    atlas: DirichletInitSurfaceAtlas,
    model: AtlasLinear,
    loss: LossScheme,
    lr: float = LR,
    max_epoch: int = MAX_EPOCH,
    log_interval: int = LOG_INTERVAL,
    out_root: str = OUT_ROOT,
    iteration: int = 0,
    key: int = KEY,
) -> None:
    opt = optax.adam(learning_rate=lr)
    opt_state = opt.init(eqx.filter(model, eqx.is_inexact_array))
    lh_coor, rh_coor = get_coor(atlas)
    key = jax.random.PRNGKey(key)

    loss_history = []
    for epoch in range(max_epoch):
        key = jax.random.fold_in(key, epoch)
        (loss_value, meta), grad = eqx.filter_jit(eqx.filter_value_and_grad(
            forward, has_aux=True
        ))(model, loss, lh_coor, rh_coor, key=key)
        print(f'[ epoch {epoch} ]')
        print('\n'.join(str((k, v)) for k, v in meta.items()))
        if np.isnan(loss_value):
            raise ValueError('NaN loss')
        loss_history.append(loss_value)
        updates, opt_state = opt.update(
            eqx.filter(grad, eqx.is_inexact_array),
            opt_state,
            eqx.filter(model, eqx.is_inexact_array),
        )
        model = eqx.apply_updates(model, updates)

        if epoch % log_interval == 0 or epoch == max_epoch - 1:
            n_labels = (
                atlas.maps['cortex_L'].shape[0] +
                atlas.maps['cortex_R'].shape[0]
            )
            plot_f = configure_report(epoch, n_labels, iteration)
            plot_f(
                template="fsLR",
                load_mask=True,
                model_array_left=model_array(model=model, compartment='cortex_L'),
                model_array_right=model_array(model=model, compartment='cortex_R'),
                surf_projection=('veryinflated',),
                hemisphere=['left', 'right', None],
                views={
                    'left': ('medial', 'lateral'),
                    'right': ('medial', 'lateral'),
                    'both': ('dorsal', 'ventral', 'anterior', 'posterior'),
                },
                output_dir=out_root,
                window_size=(1200, 1000),
                # TODO: The below is really bad practice.
                elements={
                    'title': (
                        TextBuilder(
                            content=f'null parcellation',
                            bounding_box_height=64,
                            font_size_multiplier=0.7,
                            font_color='#cccccc',
                            priority=-3,
                        ),
                    ),
                    'label_count': (
                        TextBuilder(
                            content=f'{n_labels} labels',
                            bounding_box_height=48,
                            font_size_multiplier=0.5,
                            font_color='#cccccc',
                            priority=-2,
                        ),
                    ),
                    'iteration': (
                        TextBuilder(
                            content=f'iteration {iteration}',
                            bounding_box_height=48,
                            font_size_multiplier=0.5,
                            font_color='#cccccc',
                            priority=-1,
                        ),
                    ),
                    'epoch': (
                        TextBuilder(
                            content=f'epoch {epoch}',
                            bounding_box_height=48,
                            font_size_multiplier=0.5,
                            font_color='#cccccc',
                            priority=0,
                        ),
                    ),
                },
            )
    atlas.to_cifti(
        save=(
            f'{out_root}/labels-{n_labels:06}_'
            f'desc-null{iteration:06}_atlas.dlabel.nii'
        ),
        maps={k: _to_jax_array(v) for k, v in model.weight.items()},
    )


@click.command()
@click.option('-o', '--out-root', default=OUT_ROOT, type=str)
@click.option('-a', '--atlas-template', default=ATLAS_TEMPLATE, type=str)
@click.option(
    '-l',
    '--labels',
    'labels_per_hemisphere',
    default=LABELS_PER_HEMISPHERE,
    type=int,
)
@click.option(
    '-s', '--distribution-size', default=DISTRIBUTION_SIZE, type=int
)
@click.option('--res-start', default=RES_START, type=int)
@click.option('--res-stop', default=RES_STOP, type=int)
@click.option('--res-step', default=RES_STEP, type=int)
@click.option('--lr', default=LR, type=float)
@click.option('--entropy-nu', default=ENTROPY_NU, type=float)
@click.option('--equilibrium-nu', default=EQUILIBRIUM_NU, type=float)
@click.option('--compactness-nu', default=COMPACTNESS_NU, type=float)
@click.option('--dispersion-nu', default=DISPERSION_NU, type=float)
@click.option('--tether-nu', default=TETHER_NU, type=float)
@click.option('--max-epoch', default=MAX_EPOCH, type=int)
@click.option('--log-interval', default=LOG_INTERVAL, type=int)
@click.option('--key', default=KEY, type=int)
def main(
    out_root: str = OUT_ROOT,
    atlas_template: str = ATLAS_TEMPLATE,
    labels_per_hemisphere: int = LABELS_PER_HEMISPHERE,
    distribution_size: int = DISTRIBUTION_SIZE,
    res_start: int = RES_START,
    res_stop: int = RES_STOP,
    res_step: int = RES_STEP,
    lr: float = LR,
    entropy_nu: float = ENTROPY_NU,
    equilibrium_nu: float = EQUILIBRIUM_NU,
    compactness_nu: float = COMPACTNESS_NU,
    dispersion_nu: float = DISPERSION_NU,
    tether_nu: float = TETHER_NU,
    max_epoch: int = MAX_EPOCH,
    log_interval: int = LOG_INTERVAL,
    key: int = KEY,
):
    if res_start is not None:
        label_counts = list(range(res_start, res_stop, res_step))
        distribution_size = 1
    else:
        label_counts = [labels_per_hemisphere]
    if atlas_template is None:
        atlas_template = resource_filename(
            'hypercoil',
            'viz/resources/nullexample.nii'
        )
    create_nulls(
        out_root=out_root,
        atlas_template=atlas_template,
        label_counts=label_counts,
        distribution_size=distribution_size,
        lr=lr,
        entropy_nu=entropy_nu,
        equilibrium_nu=equilibrium_nu,
        compactness_nu=compactness_nu,
        dispersion_nu=dispersion_nu,
        tether_nu=tether_nu,
        max_epoch=max_epoch,
        log_interval=log_interval,
        key=key,
    )


if __name__ == '__main__':
    main()
