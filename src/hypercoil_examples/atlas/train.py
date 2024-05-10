# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Training loop
~~~~~~~~~~~~~
Training loop for the parcellation model
"""
import pickle
from itertools import product
from typing import Optional

import equinox as eqx
import jax
import jax.numpy as jnp
import optax
import pyvista as pv

from hypercoil.engine import _to_jax_array
from hypercoil_examples.atlas.aligned_dccc import (
    get_msc_dataset, _get_data
)
from hypercoil_examples.atlas.cross2subj import visualise
from hypercoil_examples.atlas.model import (
    init_full_model,
    forward,
    Tensor,
)
from hypercoil_examples.atlas.positional import (
    get_coors
)
from hyve import (
    Cell,
    plotdef,
    surf_from_archive,
    add_surface_overlay,
    surf_scalars_from_array,
    parcellate_colormap,
    draw_surface_boundary,
    text_element,
    plot_to_image,
    save_figure,
)

LEARNING_RATE = 0.002
MAX_EPOCH = 10000
ENCODER_ARCH = '64x64'
SERIAL_INJECTION_SITES = ('readout', 'residual')
SEED = 0
REPORT_INTERVAL = 10
CHECKPOINT_INTERVAL = 100
PATHWAYS = ('regulariser', 'full') # ('full',) ('regulariser',)
VISPATH = 'full'
SUBJECTS = ('01', '02', '03', '04', '05', '06', '07', '08', '09', '10',)
SESSIONS = ('01', '02', '03', '04', '05', '06', '07', '08', '09', '10',)
VISUALISE_MRF = True
VISUALISE_SINGLE = True
ENERGY_NU = 1.
RECON_NU = 1.
TETHER_NU = 1.
DIV_NU = 1e3
POINT_POTENTIALS_NU = 1.
DOUBLET_POTENTIALS_NU = 2.
MASS_POTENTIALS_NU = 100.


#jax.config.update('jax_debug_nans', True)
forward_backward = eqx.filter_value_and_grad(
    eqx.filter_jit(forward),
    #forward,
    has_aux=True,
)
forward_backward_bkp = eqx.filter_value_and_grad(
    forward,
    has_aux=True,
)


def visdef():
    layout = Cell() / Cell() << (1 / 2)
    layout = layout | Cell() | layout << (1 / 3)
    layout = Cell() / layout << (1 / 14)
    #layout = layout / Cell() << (1 / 15)
    annotations = {
        0: dict(elements=['title']),
        1: dict(
            hemisphere='left',
            view='lateral',
        ),
        2: dict(
            hemisphere='left',
            view='medial',
        ),
        3: dict(view='dorsal'),
        4: dict(
            hemisphere='right',
            view='lateral',
        ),
        5: dict(
            hemisphere='right',
            view='medial',
        ),
    }
    layout = layout.annotate(annotations)
    plot_f = plotdef(
        surf_from_archive(),
        add_surface_overlay(
            'parcellation',
            surf_scalars_from_array(
                'parcellation',
                is_masked=True,
                allow_multihemisphere=False,
            ),
            parcellate_colormap('parcellation'),
            draw_surface_boundary(
                'parcellation',
                'parcellation',
                copy_values_to_boundary=True,
                target_domain='face',
                num_steps=0,
                v2f_interpolation='mode',
            ),
        ),
        text_element(
            name='title',
            content=f'Model',
            bounding_box_height=128,
            font_size_multiplier=0.8,
            font_color='#cccccc',
            priority=-1,
        ),
        plot_to_image(),
        save_figure(
            layout_kernel=layout,
            padding=0,
            canvas_size=(1200, 440),
            canvas_color=(0, 0, 0),
            fname_spec='scalars-{surfscalars}',
            scalar_bar_action='collect',
        ),
    )
    return plot_f


def visualise(
    name: str,
    plot_f: callable,
    log_prob_L: Optional[jnp.ndarray] = None,
    log_prob_R: Optional[jnp.ndarray] = None,
):
    array_left = log_prob_L.argmax(-1)
    array_right = log_prob_R.argmax(-1)
    plot_f(
        template='fsLR',
        surf_projection='veryinflated',
        parcellation_array_left=array_left,
        parcellation_array_right=array_right,
        parcellation_cmap='network',
        window_size=(600, 500),
        hemisphere=['left', 'right', 'both'],
        views={
            'left': ('medial', 'lateral'),
            'right': ('medial', 'lateral'),
            'both': ('dorsal',), # 'ventral', 'anterior', 'posterior'),
        },
        theme=pv.themes.DarkTheme(),
        output_dir='/tmp',
        title_element_content=f'Model: {name}',
        fname_spec=f'scalars-{name}',
        load_mask=True,
    )


def update(
    model,
    opt_state,
    *,
    opt,
    compartment,
    coor,
    encoder,
    encoder_result,
    epoch,
    pathway,
    key,
):
    if compartment == 'cortex_R': jax.config.update('jax_debug_nans', True)
    try:
        (loss, meta), grad = forward_backward(
        #forward(
            model,
            coor=coor,
            encoder_result=encoder_result,
            encoder=encoder,
            compartment=compartment,
            mode=pathway,
            energy_nu=ENERGY_NU,
            recon_nu=RECON_NU,
            tether_nu=TETHER_NU,
            div_nu=DIV_NU,
            point_potentials_nu=POINT_POTENTIALS_NU,
            doublet_potentials_nu=DOUBLET_POTENTIALS_NU,
            mass_potentials_nu=MASS_POTENTIALS_NU,
            encoder_type=ENCODER_ARCH,
            injection_points=SERIAL_INJECTION_SITES,
            inference=False,
            key=key,
        )
    except FloatingPointError:
        forward(
            model,
            coor=coor,
            encoder_result=encoder_result,
            encoder=encoder,
            compartment=compartment,
            mode=pathway,
            energy_nu=ENERGY_NU,
            recon_nu=RECON_NU,
            tether_nu=TETHER_NU,
            div_nu=DIV_NU,
            key=key,
        )
    #return model, opt_state, 0, {}
    if jnp.isnan(loss) or jnp.isinf(loss):
        print(f'NaN or infinite loss at epoch {epoch}. Skipping update')
        print(meta)
        breakpoint()
        return model, opt_state, None, {}
    updates, opt_state = opt.update(
        eqx.filter(grad, eqx.is_inexact_array),
        opt_state,
        eqx.filter(model, eqx.is_inexact_array),
    )
    model = eqx.apply_updates(model, updates)
    del updates, grad
    return model, opt_state, loss.item(), {k: v.item() for k, v in meta.items()}


def accumulate_metadata(
    meta_acc: dict,
    meta: dict,
    epoch: int,
    num_entities: int,
) -> dict:
    epoch_complete = ((epoch % num_entities == 0) and (epoch > 0))
    old_meta_acc = None
    epoch_loss = None
    for k, v in meta.items():
        if k not in meta_acc:
            meta_acc[k] = []
        meta_acc[k] += [v]
    if epoch_complete:
        for k, v in meta_acc.items():
            meta_acc[k] = jnp.mean(jnp.asarray(v)).item()
        epoch_loss = sum(meta_acc.values())
        print('\nEPOCH RESULTS')
        print('\n'.join([f'[]{k}: {v}' for k, v in meta_acc.items()]))
        print(f'Total mean loss (train): {epoch_loss}\n')
        old_meta_acc = meta_acc
        meta_acc = {}
    return meta_acc, epoch_complete, old_meta_acc, epoch_loss


def main(
    num_parcels: int = 100,
    start_epoch: Optional[int] = None,
):
    key = jax.random.PRNGKey(SEED)
    data_entities = tuple(product(SESSIONS, SUBJECTS))
    num_entities = len(data_entities)
    coor_L, coor_R = get_coors()
    plot_f = visdef()
    # The encoder will handle data normalisation and GSR
    T = _get_data(get_msc_dataset('01', '01'), normalise=False, gsr=False)
    model, encoder, template = init_full_model(
        T=T,
        coor_L=coor_L,
        coor_R=coor_R,
        num_parcels=num_parcels,
        encoder_type=ENCODER_ARCH,
        injection_points=SERIAL_INJECTION_SITES,
    )
    #encode = encoder
    encode = eqx.filter_jit(encoder)
    opt = optax.adamw(learning_rate=LEARNING_RATE)
    opt_state = opt.init(eqx.filter(model, eqx.is_inexact_array))
    losses = []
    epoch_history = []
    coor = {
        'cortex_L': coor_L,
        'cortex_R': coor_R,
    }
    if start_epoch is not None:
        model = eqx.tree_deserialise_leaves(
            f'/tmp/parcellation_model_checkpoint{start_epoch}',
            like=model,
        )
        opt_state = eqx.tree_deserialise_leaves(
            f'/tmp/parcellation_optim_checkpoint{start_epoch}',
            like=opt_state,
        )
        try:
            with open('/tmp/epoch_history.pkl', 'rb') as f:
                epoch_history = pickle.load(f)
        except FileNotFoundError:
            print('No epoch history found--starting new record')
    else:
        start_epoch = -1
    meta_acc = {}
    for i in range(start_epoch + 1, MAX_EPOCH + 1):
        session, subject = data_entities[i % num_entities]
        print(f'Epoch {i} (sub-{subject} ses-{session})')
        try:
            # The encoder will handle data normalisation and GSR
            T = _get_data(
                get_msc_dataset(subject, session),
                normalise=False,
                gsr=False,
            )
        except FileNotFoundError:
            print(
                f'Data entity sub-{subject} ses-{session} is absent. '
                'Skipping'
            )
        if jnp.any(jnp.isnan(T)):
            print(
                f'Invalid data for entity sub-{subject} ses-{session}. '
                'Skipping'
            )
            continue
        encoder_result = encode(
            T=T,
            coor_L=coor_L,
            coor_R=coor_R,
            M=template,
        )
        if any([
            jnp.any(jnp.isnan(encoder_result[0][i][compartment])).item()
            for compartment in ('cortex_L', 'cortex_R')
            for i in range(3)
        ]):
            print(
                f'Invalid data for entity sub-{subject} ses-{session}. '
                'Skipping'
            )
            continue
        key_e = jax.random.fold_in(key, i)
        key_l, key_r = jax.random.split(key_e)
        meta_L = {}
        meta_R = {}
        loss_ = 0
        for pathway in PATHWAYS:
            model, opt_state, loss_L, meta_L[pathway] = update(
                model=model,
                opt_state=opt_state,
                opt=opt,
                compartment='cortex_L',
                coor=coor,
                encoder=encoder,
                encoder_result=encoder_result,
                epoch=i,
                pathway=pathway,
                key=key_l,
            )
            if True:
                model, opt_state, loss_R, meta_R[pathway] = update(
                    model=model,
                    opt_state=opt_state,
                    opt=opt,
                    compartment='cortex_R',
                    coor=coor,
                    encoder=encoder,
                    encoder_result=encoder_result,
                    epoch=i,
                    pathway=pathway,
                    key=key_r,
                )
            else:
                loss_R = 0
                meta_R[pathway] = {k: 0 for k in meta_L}
            loss_ += (loss_L + loss_R)
        meta_L = {
            f'{t}_{p}': v
            for p, e in meta_L.items()
            for t, v in e.items()
        }
        meta_R = {
            f'{t}_{p}': v
            for p, e in meta_R.items()
            for t, v in e.items()
        }
        losses += [loss_]
        meta = {k: meta_L[k] + meta_R[k] for k in meta_L}
        print('\n'.join([f'[]{k}: {v}' for k, v in meta.items()]))
        (
            meta_acc, epoch_complete, old_meta_acc, epoch_loss
        ) = accumulate_metadata(meta_acc, meta, i, num_entities)
        if epoch_complete:
            epoch_history += [(epoch_loss, old_meta_acc)]
            with open('/tmp/epoch_history.pkl', 'wb') as f:
                pickle.dump(epoch_history, f)
        if i % REPORT_INTERVAL == 0:
            if VISUALISE_MRF:
                visualise(
                    name=f'MRF_epoch-{i}',
                    plot_f=plot_f,
                    log_prob_L=model.regulariser[
                        'cortex_L'
                    ].selectivity_distribution.log_prob(
                        template['cortex_L']
                    ) + model.regulariser[
                        'cortex_L'
                    ].spatial_distribution.log_prob(
                        coor_L
                    ),
                    log_prob_R=model.regulariser[
                        'cortex_R'
                    ].selectivity_distribution.log_prob(
                        template['cortex_R']
                    ) + model.regulariser[
                        'cortex_R'
                    ].spatial_distribution.log_prob(
                        coor_R
                    ),
                )
            #TODO: Load a specific set of subjects and sessions
            if VISUALISE_SINGLE:
                fwd = model if VISPATH == 'full' else model.regulariser_path
                P, _, _ = eqx.filter_jit(fwd)(
                    coor={
                        'cortex_L': coor_L,
                        'cortex_R': coor_R,
                    },
                    encoder=encoder,
                    encoder_result=encoder_result,
                    compartments=('cortex_L', 'cortex_R'),
                    encoder_type=ENCODER_ARCH,
                    injection_points=SERIAL_INJECTION_SITES,
                    inference=True,
                    key=key,
                )
                visualise(
                    name=f'SingleSubj_epoch_{i}',
                    log_prob_L=P['cortex_L'].T,
                    log_prob_R=P['cortex_R'].T,
                    plot_f=plot_f,
                )
            if i % CHECKPOINT_INTERVAL == 0:
                print('Serialising model and optimiser state for checkpoint')
                eqx.tree_serialise_leaves(
                    f'/tmp/parcellation_model_checkpoint{i}',
                    model,
                )
                eqx.tree_serialise_leaves(
                    f'/tmp/parcellation_optim_checkpoint{i}',
                    opt_state,
                )
    jnp.save('/tmp/cortexL.npy', P['cortex_L'], allow_pickle=False)
    jnp.save('/tmp/cortexR.npy', P['cortex_R'], allow_pickle=False)
    import matplotlib.pyplot as plt
    plt.plot([e[0] for e in epoch_history]) # training loss
    plt.savefig('/tmp/losses.png')
    assert 0


if __name__ == '__main__':
    main()
