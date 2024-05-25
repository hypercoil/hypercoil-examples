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
from hypercoil.functional import sample_overlapping_windows_existing_ax
from hypercoil_examples.atlas.const import HCP_DATA_SPLIT_DEF_ROOT
from hypercoil_examples.atlas.cross2subj import visualise
from hypercoil_examples.atlas.data import (
    get_hcp_dataset, get_msc_dataset, _get_data
)
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
MAX_EPOCH = 19600
ENCODER_ARCH = '64x64'
SERIAL_INJECTION_SITES = ('readout', 'residual')
PATHWAYS = ('regulariser', 'full') # ('full',) ('regulariser',)
SEED = 0

REPORT_INTERVAL = 196
CHECKPOINT_INTERVAL = 196
MSC_SUBJECTS = ('01', '02', '03', '04', '05', '06', '07', '08', '09', '10',)
MSC_SESSIONS = ('01', '02', '03', '04', '05', '06', '07', '08', '09', '10',)
MSC_SUBJECTS = ('05', '06')
MSC_TASKS = (
    'rest', 'motor_run-01', 'motor_run-02',
    'glasslexical_run-01', 'glasslexical_run-02',
    'memoryfaces', 'memoryscenes', 'memorywords',
)
HCP_TASKS = (
    'REST1', 'REST2', 'EMOTION', 'GAMBLING',
    'LANGUAGE', 'MOTOR', 'RELATIONAL', 'SOCIAL', 'WM',
)
DATASETS = ('HCP', 'MSC')
VISPATH = 'full'
VISUALISE_TEMPLATE = True
VISUALISE_SINGLE = True

ELLGAT_DROPOUT = 0.1
ENERGY_NU = 1.
RECON_NU = 1.
TETHER_NU = 1.
DIV_NU = 1e3
TEMPLATE_ENERGY_NU = 1.
POINT_POTENTIALS_NU = 1.
DOUBLET_POTENTIALS_NU = 2.
MASS_POTENTIALS_NU = 100.
VMF_SPATIAL_KAPPA = 50.
VMF_SELECTIVITY_KAPPA = 20.

# Temperature sampler takes the form of a tuple:
# The first element is the number of samples to take
# The second element is a callable that takes a single seed argument and
# returns a temperature value
# The third element is an integer that is folded into the epoch key to
# determine the seed for the temperature sampler
TEMPERATURE_SEED = 3829
NUM_TEMPERATURE_SAMPLES = 5
TEMPERATURE_SAMPLER = (
    NUM_TEMPERATURE_SAMPLES,
    lambda s: jnp.exp(-2 * jax.random.uniform(s, shape=(1,))),
    TEMPERATURE_SEED,
)
TEMPERATURE_SAMPLER = None
# Window sampler takes the form of a tuple:
# The first element is the number of samples to take
# The second element is a callable that takes time series and seed arguments
# and returns a windowed time series
# The third element is an integer that is folded into the epoch key to
# determine the seed for the window sampler
WINDOW_SIZE = 500
WINDOW_SEED = 2704
NUM_WINDOW_SAMPLES = 3
WINDOW_SAMPLER = (
    NUM_WINDOW_SAMPLES,
    lambda x, s: sample_overlapping_windows_existing_ax(
        x,
        WINDOW_SIZE,
        key=s,
    ),
    WINDOW_SEED,
)
WINDOW_SAMPLER = None

DATA_SAMPLER_KEY = 9902


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
    temperature,
    key,
):
    #if compartment == 'cortex_R': jax.config.update('jax_debug_nans', True)
    template_energy_nu = (
        TEMPLATE_ENERGY_NU if pathway == 'regulariser' else 0.
    )
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
            template_energy_nu=template_energy_nu,
            point_potentials_nu=POINT_POTENTIALS_NU,
            doublet_potentials_nu=DOUBLET_POTENTIALS_NU,
            mass_potentials_nu=MASS_POTENTIALS_NU,
            encoder_type=ENCODER_ARCH,
            injection_points=SERIAL_INJECTION_SITES,
            temperature=temperature,
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
    print_results: bool = True,
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
        if print_results:
            print('\nEPOCH RESULTS')
            print('\n'.join([f'[]{k}: {v}' for k, v in meta_acc.items()]))
            print(f'Total mean loss (train): {epoch_loss}\n')
        old_meta_acc = meta_acc
        meta_acc = {}
    return meta_acc, epoch_complete, old_meta_acc, epoch_loss


def main(
    num_parcels: int = 200,
    start_epoch: Optional[int] = None,
):
    key = jax.random.PRNGKey(SEED)
    data_entities = []
    if 'MSC' in DATASETS:
        data_entities = data_entities + [
            {'ds': 'MSC', 'session': ses, 'subject': sub, 'task': task}
            for ses, sub, task in product(
                MSC_SESSIONS, MSC_SUBJECTS, MSC_TASKS
            )
        ]
    if 'HCP' in DATASETS:
        with open(f'{HCP_DATA_SPLIT_DEF_ROOT}/split_template.txt', 'r') as f:
            hcp_subjects = f.read().splitlines()
        data_entities = data_entities + [
            {'ds': 'HCP', 'run': run, 'subject': sub, 'task': task}
            for run, sub, task in product(
                ('LR', 'RL'), hcp_subjects, HCP_TASKS
            )
        ]
    num_entities = len(data_entities)
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
        spatial_kappa=VMF_SPATIAL_KAPPA,
        selectivity_kappa=VMF_SELECTIVITY_KAPPA,
        dropout=ELLGAT_DROPOUT,
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
    # Configuration for data augmentation
    (
        n_temperature_samples,
        temperature_sampler,
        temperature_seed,
    ) = TEMPERATURE_SAMPLER or (1, lambda x: 1., 0)
    (
        n_window_samples,
        window_sampler,
        window_seed,
    ) = WINDOW_SAMPLER or (1, lambda x, s: x, 0)
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
        key_e = jax.random.fold_in(key, i)
        entity = data_entities[i % num_entities]

        try:
            ds = entity.get('ds')
            # The encoder will handle data normalisation and GSR
            if ds == 'MSC':
                subject = entity.get('subject')
                session = entity.get('session')
                task = entity.get('task')
                T = _get_data(
                    *get_msc_dataset(subject, session, task, get_confounds=True,),
                    normalise=False,
                    gsr=False,
                    pad_to_size=WINDOW_SIZE,
                    key=jax.random.fold_in(key_e, DATA_SAMPLER_KEY),
                )
            elif ds == 'HCP':
                subject = entity.get('subject')
                session = entity.get('run')
                task = entity.get('task')
                T = _get_data(
                    *get_hcp_dataset(subject, session, task, get_confounds=True,),
                    normalise=False,
                    gsr=False,
                    pad_to_size=WINDOW_SIZE,
                    key=jax.random.fold_in(key_e, DATA_SAMPLER_KEY),
                )
        except FileNotFoundError:
            print(
                f'Data entity {entity} is absent. '
                'Skipping'
            )
            continue
        print(f'Epoch {i} (ds-{ds} sub-{subject} ses-{session} task-{task})')
        if jnp.any(jnp.isnan(T)):
            print(
                f'Invalid data for entity sub-{subject} ses-{session}. '
                'Skipping'
            )
            breakpoint()
            continue
        Ts = [
            window_sampler(
                T,
                jax.random.fold_in(key_e, (s + 1) * window_seed),
            )
            for s in range(n_window_samples)
        ]
        meta = {}
        for j, T in enumerate(Ts):
            T = jnp.where(
                jnp.isclose(T.std(-1), 0)[..., None],
                jax.random.normal(jax.random.fold_in(key_e, 54), T.shape),
                T,
            )
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
                    f'Invalid encoding for entity sub-{subject} '
                    f'ses-{session}. Skipping'
                )
                continue
            key_l, key_r = jax.random.split(key_e)
            temperatures = [
                temperature_sampler(
                    jax.random.fold_in(key_e, (s + 1) * temperature_seed),
                )
                for s in range(n_temperature_samples)
            ]
            for k, temperature in enumerate(temperatures):
                meta_L_call = {}
                meta_R_call = {}
                #loss_ = 0
                for pathway in PATHWAYS:
                    model, opt_state, loss_L, meta_L_call[pathway] = update(
                        model=model,
                        opt_state=opt_state,
                        opt=opt,
                        compartment='cortex_L',
                        coor=coor,
                        encoder=encoder,
                        encoder_result=encoder_result,
                        epoch=i,
                        pathway=pathway,
                        temperature=temperature,
                        key=key_l,
                    )
                    model, opt_state, loss_R, meta_R_call[pathway] = update(
                        model=model,
                        opt_state=opt_state,
                        opt=opt,
                        compartment='cortex_R',
                        coor=coor,
                        encoder=encoder,
                        encoder_result=encoder_result,
                        epoch=i,
                        pathway=pathway,
                        temperature=temperature,
                        key=key_r,
                    )
                    #loss_ += (loss_L + loss_R)
                meta_L_call = {
                    f'{t}_{p}': v
                    for p, e in meta_L_call.items()
                    for t, v in e.items()
                }
                meta_R_call = {
                    f'{t}_{p}': v
                    for p, e in meta_R_call.items()
                    for t, v in e.items()
                }
                meta_call = {
                    k: meta_L_call[k] + meta_R_call[k]
                    for k in meta_L_call
                }
                (
                    meta, _, new_meta, loss_
                ) = accumulate_metadata(
                    meta,
                    meta_call,
                    j * n_temperature_samples + k + 1,
                    n_temperature_samples * n_window_samples,
                    print_results=False,
                )
        meta = new_meta
        losses += [loss_]
        print('\n'.join([f'[]{k}: {v}' for k, v in meta.items()]))
        (
            meta_acc, epoch_complete, old_meta_acc, epoch_loss
        ) = accumulate_metadata(meta_acc, meta, i, num_entities)
        if epoch_complete:
            epoch_history += [(epoch_loss, old_meta_acc)]
            with open('/tmp/epoch_history.pkl', 'wb') as f:
                pickle.dump(epoch_history, f)
        if i % REPORT_INTERVAL == 0:
            if VISUALISE_TEMPLATE:
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
