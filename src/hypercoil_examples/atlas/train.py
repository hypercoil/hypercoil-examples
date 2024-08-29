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
import jax.tree_util as jtu
import optax
import numpy as np
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

LEARNING_RATE = 0.002
MAX_EPOCH = 24000
MAX_EPOCH = 40
ENCODER_ARCH = '64x64'
BLOCK_ARCH = 'ELLGAT'
SERIAL_INJECTION_SITES = ('readout', 'residual')
PATHWAYS = ('regulariser', 'full') # ('full',) # ('regulariser',) #
SEED = 47

REPORT_INTERVAL = 100 # 1 #
CHECKPOINT_INTERVAL = 100 # 1 #
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
        ('rest', 'rest'),
        ('motor_run-01', 'motor'),
        ('motor_run-02', 'motor'),
        ('glasslexical_run-01', 'glasslexical'),
        ('glasslexical_run-02', 'glasslexical'),
        ('memoryfaces', 'memoryfaces'),
        ('memoryscenes', 'memoryscenes'),
        ('memorywords', 'memorywords'),
    ),
    'HCP': (
        ('REST1', 'REST'),
        ('REST2', 'REST'),
        ('EMOTION', 'EMOTION'),
        ('GAMBLING', 'GAMBLING'),
        ('LANGUAGE', 'LANGUAGE'),
        ('MOTOR', 'MOTOR'),
        ('RELATIONAL', 'RELATIONAL'),
        ('SOCIAL', 'SOCIAL'),
        ('WM', 'WM'),
    ),
}
TASKS_FILES = {k: tuple(v[0] for v in vs) for k, vs in TASKS.items()}
TASKS_TARGETS = {k: tuple(sorted(set(v[1] for v in vs))) for k, vs in TASKS.items()}
DATASETS = ('HCP', 'MSC')
VISPATH = 'full' # 'regulariser'
VISUALISE_TEMPLATE = True
VISUALISE_SINGLE = True

ELLGAT_DROPOUT = 0.1
ENERGY_NU = 1. # 0. #
RECON_NU = 1. # 0. #
TETHER_NU = 1. # 0. #
DIV_NU = 1e3 # 0. #
CLASSIFIER_NU = 5. # 0. #
TEMPLATE_ENERGY_NU = 1. # 0. #
POINT_POTENTIALS_NU = 1. # 0. #
DOUBLET_POTENTIALS_NU = 2.5 # 0. #
MASS_POTENTIALS_NU = 100. # 0. #
VMF_SPATIAL_KAPPA = 50.
VMF_SELECTIVITY_KAPPA = 20.
FIXED_KAPPA = False
BIG_KAPPA_NU = 1e-5 # 0. #
SMALL_KAPPA_NU = 1e0 # 0. #

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
# forward_backward = forward


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
    plot_boundaries_f = plotdef(
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
    plot_confidence_f = plotdef(
        surf_from_archive(),
        add_surface_overlay(
            'parcel_confidence',
            surf_scalars_from_array(
                'parcel_confidence',
                is_masked=True,
            ),
        ),
        text_element(
            name='title',
            content=f'Model confidence',
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
    return plot_boundaries_f, plot_confidence_f


#TODO:
# These are NOT log-probabilities; they are just probabilities! This code was
# copied from another submodule and should be fixed once we have a better
# editor!
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


def visualise_confidence(
    name: str,
    plot_f: callable,
    log_prob_L: Optional[jnp.ndarray] = None,
    log_prob_R: Optional[jnp.ndarray] = None,
):
    array_left = log_prob_L.max(-1)
    array_right = log_prob_R.max(-1)
    plot_f(
        template='fsLR',
        surf_projection='veryinflated',
        parcel_confidence_array_left=array_left,
        parcel_confidence_array_right=array_right,
        parcel_confidence_cmap='magma',
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
    #div_nu = (
    #    DIV_NU if pathway == 'regulariser' else 0.
    #)
    div_nu = DIV_NU
    if classify is not None:
        classifier_nu = CLASSIFIER_NU
        readout_name, classifier_target = classify
    else:
        classifier_nu = 0
        readout_name = classifier_target = None
    if FIXED_KAPPA:
        spatial_kappa_energy = None
        selectivity_kappa_energy = None
    else:
        spatial_kappa_energy = (VMF_SPATIAL_KAPPA, BIG_KAPPA_NU, SMALL_KAPPA_NU)
        selectivity_kappa_energy = (VMF_SELECTIVITY_KAPPA, BIG_KAPPA_NU, SMALL_KAPPA_NU)
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
            div_nu=div_nu,
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
            div_nu=div_nu,
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
    grad_info = introspect_approximator_gradients(grad.approximator, model.approximator)
    updates_info = introspect_approximator_gradients(updates.approximator, model.approximator)
    model = eqx.apply_updates(model, updates)
    del updates, grad
    return model, opt_state, loss.item(), {k: v.item() for k, v in meta.items()}, grad_info, updates_info


def get_param_clip_rotation(
    max_norm_ratio: float, rot_weight: float = 0.05, eps: float = 1e-6
) -> callable:
    def rotate_param_to_clipped_norm(updates, params):
        g_norm = jnp.linalg.norm(updates)
        p_norm = jnp.linalg.norm(params)
        ratio = g_norm / (p_norm + eps)
        if ratio < max_norm_ratio:
            return updates
        sgn = jnp.sign(updates)
        max_single = 1 / jnp.sqrt(jnp.size(updates))
        rot_target = jnp.where(sgn * params > max_single, sgn * max_single, params)
        rotated = (1 - rot_weight) * updates + rot_weight * rot_target
        return rotated / (jnp.linalg.norm(rotated) + eps) * p_norm * max_norm_ratio

    return rotate_param_to_clipped_norm


def rotate_to_clipped_norm(
    max_norm_ratio: float, rot_weight: float = 0.05
) -> optax.GradientTransformation:
    def update_fn(updates, state, params):
        if params is None:
            raise ValueError() #optax.base.NO_PARAMS_MSG)
        return jtu.tree_map(get_param_clip_rotation(max_norm_ratio, rot_weight), updates, params), state
    return optax.GradientTransformation(lambda _: optax.EmptyState(), update_fn)


def get_params(model):
    return jtu.tree_flatten(
        eqx.filter(model, eqx.is_inexact_array)
    )[0]


def model_and_delta_params(grad, model):
    return list(zip(get_params(grad), get_params(model)))


def introspect_gradients(grad, model):
    norm_fn = jnp.linalg.norm
    max_fn = lambda x: jnp.max(jnp.abs(x))
    mean_fn = lambda x: jnp.mean(jnp.abs(x))
    params = model_and_delta_params(grad, model)
    norm_ = [norm_fn(g) for g, _ in params]
    relnorm_ = [gv / norm_fn(m) for gv, (_, m) in zip(norm_, params)]
    max_ = [max_fn(g) for g, _ in params]
    relmax_ = [gv / max_fn(m) for gv, (_, m) in zip(max_, params)]
    mean_ = [mean_fn(g) for g, _ in params]
    relmean_ = [gv / mean_fn(m) for gv, (_, m) in zip(mean_, params)]
    return {
        'norm': jnp.asarray(norm_).reshape(1, -1),
        'relnorm': jnp.asarray(relnorm_).reshape(1, -1),
        'max': jnp.asarray(max_).reshape(1, -1),
        'relmax': jnp.asarray(relmax_).reshape(1, -1),
        'mean': jnp.asarray(mean_).reshape(1, -1),
        'relmean': jnp.asarray(relmean_).reshape(1, -1),
    }


def introspect_approximator_gradients(grad, model):
    return {
        'contractive': introspect_gradients(grad.contractive, model.contractive),
        'expansive': introspect_gradients(grad.expansive, model.expansive),
        'resample': introspect_gradients(grad.resample, model.resample),
        'ingress': introspect_gradients(grad.ingress, model.ingress),
        'readout': introspect_gradients(grad.readout, model.readout),
    }


def accumulate_gradinfo(
    gradinfo_acc: dict, gradinfo: dict
) -> dict:
    if not gradinfo_acc:
        return gradinfo
    return {
        k: {
            j: jnp.concatenate((w, gradinfo[k][j]))
            for j, w in v.items()
        }
        for k, v in gradinfo_acc.items()
    }


def flatten_gradinfo(grad_info):
    return {
        f'{k}.{j}': v[j]
        for k, v in grad_info.items()
        for j, w in v.items()
    }


def unflatten_gradinfo(grad_info):
    unflat = {}
    for kj, v in grad_info.items():
        k, j = kj.split('.')
        unflat[k] = unflat.get(k, {})
        unflat[k][j] = v
    return unflat


class GradientInfo(eqx.Module):
    record: dict


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
    keys = jax.random.split(key, len(TASKS_TARGETS.keys()))
    readouts = {
        ds: jax.random.normal(keys[i], shape=(len(tasks), in_dim))
        for i, (ds, tasks) in enumerate(TASKS_TARGETS.items())
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
    start_step: Optional[int] = None, # 235, #
    classify_task: bool = True,
):
    key = jax.random.PRNGKey(SEED)
    key_d, key_m = jax.random.split(key)
    data_entities = {}
    val_entities = {}
    if 'MSC' in DATASETS:
        data_entities = {**data_entities, 'MSC': [
            {'ds': 'MSC', 'session': ses, 'subject': sub, 'task': task}
            for ses, sub, task in product(
                MSC_SESSIONS, MSC_SUBJECTS_TRAIN, TASKS_FILES['MSC']
            )
        ]}
        val_entities = {**val_entities, 'MSC': [
            {'ds': 'MSC', 'session': ses, 'subject': sub, 'task': task}
            for ses, sub, task in product(
                MSC_SESSIONS, MSC_SUBJECTS_VAL, TASKS_FILES['MSC']
            )
        ]}
    if 'HCP' in DATASETS:
        with open(f'{HCP_DATA_SPLIT_DEF_ROOT}/split_train.txt', 'r') as f:
            hcp_subjects_train = f.read().splitlines()
        data_entities = {**data_entities, 'HCP': [
            {'ds': 'HCP', 'run': run, 'subject': sub, 'task': task}
            for run, sub, task in product(
                ('LR', 'RL'), hcp_subjects_train, TASKS_FILES['HCP']
            )
        ]}
        with open(f'{HCP_DATA_SPLIT_DEF_ROOT}/split_val.txt', 'r') as f:
            hcp_subjects_val = f.read().splitlines()
        val_entities = {**val_entities, 'HCP': [
            {'ds': 'HCP', 'run': run, 'subject': sub, 'task': task}
            for run, sub, task in product(
                ('LR', 'RL'), hcp_subjects_val, TASKS_FILES['HCP']
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
    total_steps = total_epoch_size * MAX_EPOCH
    steps_by_dataset = {k: v * MAX_EPOCH for k, v in EPOCH_SIZE.items()}
    iter_builder_dict = {
        k: (v // num_entities[k], v % num_entities[k])
        for k, v in steps_by_dataset.items()
    }
    instance_indices = {
        k: jnp.concatenate(
            [
                jax.random.choice(
                    jax.random.fold_in(key_d, e * DATA_SHUFFLE_KEY),
                    num_entities[k],
                    shape=(num_entities[k],),
                    replace=False,
                )
                for e in range(v[0] + 1)
            ]
        )[:steps_by_dataset[k]].tolist()
        for k, v in iter_builder_dict.items()
    }
    instance_index_iter = [
        sum(
            [
                [(k, f)]
                for k, v in instance_indices.items()
                for f in v[EPOCH_SIZE[k]*e:EPOCH_SIZE[k]*(e + 1)]
            ],
            [],
        )
        for e in range(MAX_EPOCH)
    ]
    index_shuffle = [
        jax.random.choice(
            jax.random.fold_in(key_d, 2 * e * DATA_SHUFFLE_KEY),
            total_epoch_size,
            shape=(total_epoch_size,),
            replace=False,
        )
        for e in range(MAX_EPOCH)
    ]
    instance_index_iter = [
        np.asarray(e)[i].tolist()
        for e, i in zip(instance_index_iter, index_shuffle)
    ]
    coor_L, coor_R = get_coors()
    plot_boundaries_f, plot_confidence_f = visdef()
    # The encoder will handle data normalisation and GSR
    T = _get_data(get_msc_dataset('01', '01'), normalise=False, gsr=False)
    model, encoder, template = init_full_model(
        T=T,
        coor_L=coor_L,
        coor_R=coor_R,
        num_parcels=num_parcels,
        encoder_type=ENCODER_ARCH,
        block_arch=BLOCK_ARCH,
        injection_points=SERIAL_INJECTION_SITES,
        spatial_kappa=VMF_SPATIAL_KAPPA,
        selectivity_kappa=VMF_SELECTIVITY_KAPPA,
        fixed_kappa=FIXED_KAPPA,
        dropout=ELLGAT_DROPOUT,
        key=key_m,
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
    #opt = optax.chain(optax.adaptive_grad_clip(0.02), opt)
    opt = optax.chain(rotate_to_clipped_norm(0.1), opt)
    opt_state = opt.init(eqx.filter(model, eqx.is_inexact_array))
    losses = []
    epoch_history = []
    epoch_history_val = []
    grad_info_L = {}
    updates_info_L = {}
    grad_info_R = {}
    updates_info_R = {}
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
    if start_step is not None:
        model = eqx.tree_deserialise_leaves(
            f'/tmp/parcellation_model_checkpoint{start_step}',
            like=model,
        )
        opt_state = eqx.tree_deserialise_leaves(
            f'/tmp/parcellation_optim_checkpoint{start_step}',
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
        try:
            with open('/tmp/grad_info_L.pkl', 'rb') as f:
                grad_info_L = unflatten_gradinfo(pickle.load(f))
            with open('/tmp/updates_info_L.pkl', 'rb') as f:
                updates_info_L = unflatten_gradinfo(pickle.load(f))
            with open('/tmp/grad_info_R.pkl', 'rb') as f:
                grad_info_R = unflatten_gradinfo(pickle.load(f))
            with open('/tmp/updates_info_R.pkl', 'rb') as f:
                updates_info_R = unflatten_gradinfo(pickle.load(f))
        except FileNotFoundError:
            print('No gradient diagnostics found--starting new record')
        start_epoch = start_step // total_epoch_size
        # add 1 because saving is at epoch end
        start_step = start_step % total_epoch_size + 1
    else:
        start_epoch = 0
        start_step = 0
    last_report = last_checkpoint = (
        (start_epoch) * total_epoch_size
        - max(REPORT_INTERVAL, CHECKPOINT_INTERVAL) - 1
    )
    meta_acc = {}
    meta_acc_val = {}
    avail_entities = {k: [] for k in EPOCH_SIZE}
    for i in range(start_epoch, MAX_EPOCH + 1):
        epoch_entities = [
            data_entities[ds][int(idx)] for ds, idx in instance_index_iter[0]
        ]

        for j in range(start_step, total_epoch_size):
            k = i * total_epoch_size + j
            key_e = jax.random.fold_in(key, k)
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
                    tasks = dict(TASKS[ds])
                    tasks_targets = TASKS_TARGETS[ds]
                    classifier_target = jnp.zeros((len(tasks_targets))).at[tasks_targets.index(tasks[task])].set(1.)
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
                            model, opt_state, loss_L, meta_L_call[pathway], grad_info_L_, updates_info_L_ = update(
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
                            model, opt_state, loss_R, meta_R_call[pathway], grad_info_R_, updates_info_R_ = update(
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
                        if pathway == 'full':
                            grad_info_L = accumulate_gradinfo(grad_info_L, grad_info_L_)
                            updates_info_L = accumulate_gradinfo(updates_info_L, updates_info_L_)
                            grad_info_R = accumulate_gradinfo(grad_info_R, grad_info_R_)
                            updates_info_R = accumulate_gradinfo(updates_info_R, updates_info_R_)
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
                last_report = (k // REPORT_INTERVAL) * REPORT_INTERVAL
                if VISUALISE_TEMPLATE:
                    visualise(
                        name=f'MRF_pass-{k}',
                        plot_f=plot_boundaries_f,
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
                        plot_f=plot_boundaries_f,
                    )
                    visualise_confidence(
                        name=f'confidence_pass-{k}',
                        log_prob_L=P['cortex_L'].T,
                        log_prob_R=P['cortex_R'].T,
                        plot_f=plot_confidence_f,
                    )
            if (k - last_checkpoint) // CHECKPOINT_INTERVAL > 0:
                last_checkpoint = (k // CHECKPOINT_INTERVAL) * CHECKPOINT_INTERVAL
                print('Serialising model and optimiser state for checkpoint')
                eqx.tree_serialise_leaves(
                    f'/tmp/parcellation_model_checkpoint{k}',
                    model,
                )
                eqx.tree_serialise_leaves(
                    f'/tmp/parcellation_optim_checkpoint{k}',
                    opt_state,
                )
                with open('/tmp/grad_info_L.pkl', 'wb') as f:
                    pickle.dump(flatten_gradinfo(grad_info_L), f)
                with open('/tmp/updates_info_L.pkl', 'wb') as f:
                    pickle.dump(flatten_gradinfo(updates_info_L), f)
                with open('/tmp/grad_info_R.pkl', 'wb') as f:
                    pickle.dump(flatten_gradinfo(grad_info_R), f)
                with open('/tmp/updates_info_R.pkl', 'wb') as f:
                    pickle.dump(flatten_gradinfo(updates_info_R), f)


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
                tasks = dict(TASKS[ds])
                tasks_targets = TASKS_TARGETS[ds]
                classifier_target = jnp.zeros((len(tasks_targets))).at[tasks_targets.index(tasks[task])].set(1.)
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
