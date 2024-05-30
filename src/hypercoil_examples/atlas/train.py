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
from typing import Any, Mapping, Optional, Tuple

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
    ForwardParcellationModel,
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

LEARNING_RATE = 0.00002
MAX_EPOCH = 24000
ENCODER_ARCH = '64x64'
SERIAL_INJECTION_SITES = ('readout', 'residual')
PATHWAYS = ('regulariser', 'full') # ('full',) ('regulariser',)
SEED = 0

REPORT_INTERVAL = 200
CHECKPOINT_INTERVAL = 200
EPOCH_SIZE = {'HCP': 5, 'MSC': 5}
VAL_SIZE = {'HCP': 5, 'MSC': 5}
EPOCH_SIZE = {'HCP': 500, 'MSC': 100}
VAL_SIZE = {'HCP': 160, 'MSC': 160}
#MSC_SUBJECTS = ('01', '02', '03', '04', '05', '06', '07', '08', '09', '10',)
MSC_SESSIONS = ('01', '02', '03', '04', '05', '06', '07', '08', '09', '10',)
MSC_SUBJECTS_TRAIN = ('01', '02', '03', '08')
MSC_SUBJECTS_VAL = ('04', '07')
TASKS = {
    'MSC': (
        'rest', 'motor_run-01', 'motor_run-02',
        'glasslexical_run-01', 'glasslexical_run-02',
        'memoryfaces', 'memoryscenes', 'memorywords',
    ),
    'HCP': (
        'REST1', 'REST2', 'EMOTION', 'GAMBLING',
        'LANGUAGE', 'MOTOR', 'RELATIONAL', 'SOCIAL', 'WM',
    ),
}
DATASETS = ('HCP', 'MSC')
VISPATH = 'full'
VISUALISE_TEMPLATE = True
VISUALISE_SINGLE = True

ELLGAT_DROPOUT = 0.1
ENERGY_NU = 1.
RECON_NU = 1.
TETHER_NU = 1.
DIV_NU = 1e3
CLASSIFIER_NU = 5.
TEMPLATE_ENERGY_NU = 1.
POINT_POTENTIALS_NU = 1.
DOUBLET_POTENTIALS_NU = 10.
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

DATA_SHUFFLE_KEY = 7834
DATA_SAMPLER_KEY = 9902
READOUT_INIT_KEY = 5310


jax.config.update('jax_debug_nans', True)
forward_eval = eqx.filter_jit(forward)
forward_backward = eqx.filter_value_and_grad(
    eqx.filter_jit(forward),
    #forward,
    has_aux=True,
)
forward_backward_bkp = eqx.filter_value_and_grad(
    forward,
    has_aux=True,
)


class InvalidValueException(FloatingPointError):
    pass


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
    temperature: float = 1.,
    classify: Optional[Tuple[str, Tensor]] = None,
    key: 'jax.random.PRNGKey',
):
    #if compartment == 'cortex_R': jax.config.update('jax_debug_nans', True)
    template_energy_nu = (
        TEMPLATE_ENERGY_NU if pathway == 'regulariser' else 0.
    )
    if classify is not None:
        classifier_nu = CLASSIFIER_NU
        readout_name, classifier_target = classify
    else:
        classifier_nu = 0
        readout_name = classifier_target = None
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
            classifier_nu=classifier_nu,
            classifier_target=classifier_target,
            readout_name=readout_name,
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
        raise InvalidValueException
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


def add_readouts(
    model: ForwardParcellationModel,
    num_parcels: int,
    key: 'jax.random.PRNGKey',
):
    in_dim = int(num_parcels * (num_parcels - 1) / 2)
    keys = jax.random.split(key, len(TASKS.keys()))
    readouts = {
        ds: jax.random.normal(keys[i], shape=(len(tasks), in_dim))
        for i, (ds, tasks) in enumerate(TASKS.items())
    }

    class ForwardParcellationModelWithReadouts(ForwardParcellationModel):
        regulariser: Mapping[str, eqx.Module]
        approximator: eqx.Module
        readouts: Mapping[str, Tensor]

    return ForwardParcellationModelWithReadouts(
        regulariser=model.regulariser,
        approximator=model.approximator,
        readouts=readouts
    )


def main(
    num_parcels: int = 200,
    start_epoch: Optional[int] = 16799,
    classify_task: bool = True,
):
    key = jax.random.PRNGKey(SEED)
    data_entities = {}
    val_entities = {}
    if 'MSC' in DATASETS:
        data_entities = {**data_entities, 'MSC': [
            {'ds': 'MSC', 'session': ses, 'subject': sub, 'task': task}
            for ses, sub, task in product(
                MSC_SESSIONS, MSC_SUBJECTS_TRAIN, TASKS['MSC']
            )
        ]}
        val_entities = {**val_entities, 'MSC': [
            {'ds': 'MSC', 'session': ses, 'subject': sub, 'task': task}
            for ses, sub, task in product(
                MSC_SESSIONS, MSC_SUBJECTS_VAL, TASKS['MSC']
            )
        ]}
    if 'HCP' in DATASETS:
        with open(f'{HCP_DATA_SPLIT_DEF_ROOT}/split_train.txt', 'r') as f:
            hcp_subjects_train = f.read().splitlines()
        data_entities = {**data_entities, 'HCP': [
            {'ds': 'HCP', 'run': run, 'subject': sub, 'task': task}
            for run, sub, task in product(
                ('LR', 'RL'), hcp_subjects_train, TASKS['HCP']
            )
        ]}
        with open(f'{HCP_DATA_SPLIT_DEF_ROOT}/split_val.txt', 'r') as f:
            hcp_subjects_val = f.read().splitlines()
        val_entities = {**val_entities, 'HCP': [
            {'ds': 'HCP', 'run': run, 'subject': sub, 'task': task}
            for run, sub, task in product(
                ('LR', 'RL'), hcp_subjects_val, TASKS['HCP']
            )
        ]}
    num_entities = {
        **{k: len(v) for k, v in data_entities.items()},
        'total': sum([len(v) for v in data_entities.values()]),
    }
    # Crude. We should at least make sure our classes are appropriately
    # represented.
    val_entities = sum(
        [val_entities[ds][:VAL_SIZE[ds]] for ds in DATASETS], []
    )
    total_epoch_size = sum(EPOCH_SIZE.values())
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
    if classify_task:
        model = add_readouts(
            model,
            num_parcels=num_parcels,
            key=jax.random.fold_in(key, READOUT_INIT_KEY),
        )
    #encode = encoder
    encode = eqx.filter_jit(encoder)
    opt = optax.adamw(learning_rate=LEARNING_RATE)
    opt_state = opt.init(eqx.filter(model, eqx.is_inexact_array))
    losses = []
    epoch_history = []
    epoch_history_val = []
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
        try:
            with open('/tmp/epoch_history_val.pkl', 'rb') as f:
                epoch_history_val = pickle.load(f)
        except FileNotFoundError:
            print('No evaluation history found--starting new record')
        start_epoch = start_epoch // total_epoch_size
    else:
        start_epoch = -1
    last_report = last_checkpoint = ((start_epoch + 1) * total_epoch_size)
    meta_acc = {}
    meta_acc_val = {}
    avail_entities = {k: [] for k in EPOCH_SIZE}
    for i in range(start_epoch + 1, MAX_EPOCH + 1):
        key_e = jax.random.fold_in(key, i)
        for j, (k, v) in enumerate(avail_entities.items()):
            if len(v) < EPOCH_SIZE[k]:
                avail_entities[k] = avail_entities.get(k, []) + [
                    data_entities[k][e]
                    for e in jax.random.choice(
                        jax.random.fold_in(key, i * j * DATA_SHUFFLE_KEY),
                        num_entities[k],
                        shape=(num_entities[k],),
                        replace=False,
                    )
                ]
        epoch_entities = sum([
            avail_entities[k][:v]
            for k, v in EPOCH_SIZE.items()
        ], [])
        avail_entities = {
            k: avail_entities[k][v:]
            for k, v in EPOCH_SIZE.items()
        }
        epoch_entities = [
            epoch_entities[e] for e in jax.random.choice(
                jax.random.fold_in(key, i * DATA_SHUFFLE_KEY),
                total_epoch_size,
                shape=(total_epoch_size,),
                replace=False,
            )
        ]

        for j in range(total_epoch_size):
            k = i * total_epoch_size + j
            entity = epoch_entities[j]

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
            print(f'Epoch {i} / Pass {k} (ds-{ds} sub-{subject} ses-{session} task-{task})')
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
            for u, T in enumerate(Ts):
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
                    jnp.any(jnp.isnan(encoder_result[0][m][compartment])).item()
                    for compartment in ('cortex_L', 'cortex_R')
                    for m in range(3)
                ]):
                    print(
                        f'Invalid encoding for entity sub-{subject} '
                        f'ses-{session}. Skipping'
                    )
                    continue
                if classify_task:
                    readout_name = ds
                    tasks = TASKS[ds]
                    classifier_target = jnp.zeros((len(tasks))).at[tasks.index(task)].set(1.)
                    classifier_args = (readout_name, classifier_target)
                else:
                    classifier_args = None
                key_l, key_r = jax.random.split(key_e)
                temperatures = [
                    temperature_sampler(
                        jax.random.fold_in(key_e, (s + 1) * temperature_seed),
                    )
                    for s in range(n_temperature_samples)
                ]
                for w, temperature in enumerate(temperatures):
                    meta_L_call = {}
                    meta_R_call = {}
                    #loss_ = 0
                    for pathway in PATHWAYS:
                        try:
                            model, opt_state, loss_L, meta_L_call[pathway] = update(
                                model=model,
                                opt_state=opt_state,
                                opt=opt,
                                compartment='cortex_L',
                                coor=coor,
                                encoder=encoder,
                                encoder_result=encoder_result,
                                epoch=k,
                                pathway=pathway,
                                classify=classifier_args if pathway == 'full' else None,
                                temperature=temperature,
                                key=key_l,
                            )
                        except InvalidValueException:
                            continue
                        try:
                            model, opt_state, loss_R, meta_R_call[pathway] = update(
                                model=model,
                                opt_state=opt_state,
                                opt=opt,
                                compartment='cortex_R',
                                coor=coor,
                                encoder=encoder,
                                encoder_result=encoder_result,
                                epoch=k,
                                pathway=pathway,
                                classify=classifier_args if pathway == 'full' else None,
                                temperature=temperature,
                                key=key_r,
                            )
                        except InvalidValueException:
                            continue
                        #loss_ += (loss_L + loss_R)
                    meta_L_call = {
                        f'{z}_{p}': v
                        for p, e in meta_L_call.items()
                        for z, v in e.items()
                    }
                    meta_R_call = {
                        f'{z}_{p}': v
                        for p, e in meta_R_call.items()
                        for z, v in e.items()
                    }
                    meta_call = {
                        c: meta_L_call[c] + meta_R_call[c]
                        for c in meta_L_call
                    }
                    (
                        meta, _, new_meta, loss_
                    ) = accumulate_metadata(
                        meta,
                        meta_call,
                        u * n_temperature_samples + w + 1,
                        n_temperature_samples * n_window_samples,
                        print_results=False,
                    )
            meta = new_meta
            losses += [loss_]
            print('\n'.join([f'[]{q}: {z}' for q, z in meta.items()]))
            (
                meta_acc, epoch_complete, old_meta_acc, epoch_loss
            ) = accumulate_metadata(meta_acc, meta, k + 1, total_epoch_size)
            if epoch_complete:
                epoch_history += [(epoch_loss, old_meta_acc)]
                with open('/tmp/epoch_history.pkl', 'wb') as f:
                    pickle.dump(epoch_history, f)
            if (k - last_report) // REPORT_INTERVAL > 0:
                last_report += REPORT_INTERVAL
                if VISUALISE_TEMPLATE:
                    visualise(
                        name=f'MRF_pass-{k}',
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
                        name=f'SingleSubj_pass-{k}',
                        log_prob_L=P['cortex_L'].T,
                        log_prob_R=P['cortex_R'].T,
                        plot_f=plot_f,
                    )
            if (k - last_checkpoint) // CHECKPOINT_INTERVAL > 0:
                last_checkpoint += CHECKPOINT_INTERVAL
                print('Serialising model and optimiser state for checkpoint')
                eqx.tree_serialise_leaves(
                    f'/tmp/parcellation_model_checkpoint{k}',
                    model,
                )
                eqx.tree_serialise_leaves(
                    f'/tmp/parcellation_optim_checkpoint{k}',
                    opt_state,
                )

        # EVALUATION
        for j in range(len(val_entities)):
            k = i * len(val_entities) + j
            entity = val_entities[j]

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
            print(f'Evaluation {i} / Pass {k} (ds-{ds} sub-{subject} ses-{session} task-{task})')
            if jnp.any(jnp.isnan(T)):
                print(
                    f'Invalid data for entity sub-{subject} ses-{session}. '
                    'Skipping'
                )
                breakpoint()
                continue
            meta = {}
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
                jnp.any(jnp.isnan(encoder_result[0][m][compartment])).item()
                for compartment in ('cortex_L', 'cortex_R')
                for m in range(3)
            ]):
                print(
                    f'Invalid encoding for entity sub-{subject} '
                    f'ses-{session}. Skipping'
                )
                continue
            if classify_task:
                readout_name = ds
                tasks = TASKS[ds]
                classifier_target = jnp.zeros((len(tasks))).at[tasks.index(task)].set(1.)
                classifier_args = (readout_name, classifier_target)
            else:
                classifier_args = None
            key_l, key_r = jax.random.split(key_e)
            temperatures = [
                temperature_sampler(
                    jax.random.fold_in(key_e, (s + 1) * temperature_seed),
                )
                for s in range(n_temperature_samples)
            ]
            for w, temperature in enumerate(temperatures):
                meta_call = {'cortex_L': {}, 'cortex_R': {}}
                val_loss = {}
                #loss_ = 0
                for pathway in PATHWAYS:
                    for compartment in ('cortex_L', 'cortex_R'):
                        val_loss[compartment], meta_call[compartment][pathway] = forward_eval(
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
                            template_energy_nu=TEMPLATE_ENERGY_NU,
                            point_potentials_nu=POINT_POTENTIALS_NU,
                            doublet_potentials_nu=DOUBLET_POTENTIALS_NU,
                            mass_potentials_nu=MASS_POTENTIALS_NU,
                            classifier_nu=CLASSIFIER_NU,
                            classifier_target=classifier_target,
                            readout_name=ds,
                            encoder_type=ENCODER_ARCH,
                            injection_points=SERIAL_INJECTION_SITES,
                            temperature=temperature,
                            inference=True,
                            key=key,
                        )
                    #loss_ += (loss_L + loss_R)
                meta_call = {
                    compartment: {
                        f'{z}_{p}': v
                        for p, e in loss_meta.items()
                        for z, v in e.items()
                    }
                    for compartment, loss_meta in meta_call.items()
                }
                meta_call = {
                    c: meta_call['cortex_L'][c] + meta_call['cortex_R'][c]
                    for c in meta_call['cortex_L']
                }
                (
                    meta, _, new_meta, loss_
                ) = accumulate_metadata(
                    meta,
                    meta_call,
                    w + 1,
                    n_temperature_samples,
                    print_results=False,
                )
            meta = new_meta
            losses += [loss_]
            print('\n'.join([f'[]{q}: {z}' for q, z in meta.items()]))
            (
                meta_acc_val, epoch_complete, old_meta_acc, epoch_loss
            ) = accumulate_metadata(meta_acc_val, meta, k + 1, len(val_entities))
            if epoch_complete:
                epoch_history_val += [(epoch_loss, old_meta_acc)]
                with open('/tmp/epoch_history_val.pkl', 'wb') as f:
                    pickle.dump(epoch_history_val, f)

    jnp.save('/tmp/cortexL.npy', P['cortex_L'], allow_pickle=False)
    jnp.save('/tmp/cortexR.npy', P['cortex_R'], allow_pickle=False)
    import matplotlib.pyplot as plt
    plt.plot([e[0] for e in epoch_history]) # training loss
    plt.savefig('/tmp/losses.png')
    assert 0


if __name__ == '__main__':
    main()
