# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Evaluation loop
~~~~~~~~~~~~~~~
Evaluation loop and metrics for the parcellation model
"""
import logging
import os
import re
import pickle
from itertools import product
from pathlib import Path
from typing import Any, Literal, Mapping, Optional, Sequence, Tuple

import equinox as eqx
import nibabel as nb
import numpy as np
import pandas as pd
import jax
import jax.numpy as jnp
import optax
import pyvista as pv
import templateflow.api as tflow
from scipy.io import loadmat

from hypercoil.engine import _to_jax_array
from hypercoil.functional import residualise, sym2vec
from hypercoil_examples.atlas.behavioural import HCP_MEASURES
from hypercoil_examples.atlas.const import (
    HCP_DATA_SPLIT_DEF_ROOT,
    HCP_EXTRA_SUBJECT_MEASURES,
)
from hypercoil_examples.atlas.cross2subj import visualise
from hypercoil_examples.atlas.data import (
    get_hcp_dataset,
    get_msc_dataset,
    _get_data,
    SubjectRecord,
    HCP_RECORD_DEFAULTS,
    MSC_RECORD_DEFAULTS,
    TASKS,
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
from hypercoil_examples.atlas.train import (
    ENCODER_ARCH,
    BLOCK_ARCH,
    SERIAL_INJECTION_SITES,
    VMF_SPATIAL_KAPPA,
    VMF_SELECTIVITY_KAPPA,
    FIXED_KAPPA,
    visdef,
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

# This eventually should go away as we will want to use all of the data
SEED = 3923
PATHWAYS = ('regulariser', 'full') # ('full',) ('regulariser',)

VAL_SIZE = {'HCP': 800, } #'MSC': 160}
MSC_SESSIONS = ('01', '02', '03', '04', '05', '06', '07', '08', '09', '10',)
MSC_SUBJECTS_TRAIN = ('01', '02', '03', '08')
MSC_SUBJECTS_VAL = ('01', '02', '03', '04', '05', '06', '07', '08', '09', '10',)
DATASETS = ('MSC', 'HCP',) # ('MSC',) # ('HCP',) #
NUM_SPLITS = {'HCP': 2, 'MSC': 5}

DATA_SHUFFLE_KEY = 7834
DATA_SAMPLER_KEY = 9902
READOUT_INIT_KEY = 5310

OUTPUT_DIR = '/mnt/andromeda/Data/atlas_ts/'
#OUTPUT_DIR = '/Users/rastkociric/Downloads/atlas_ts/'
ATLAS_ROOT = '/mnt/andromeda/Data/atlases/atlases/'
BASELINES = {
    'glasser360': f'{ATLAS_ROOT}/desc-glasser_res-0360_atlas.nii',
    #'gordon333': f'{ATLAS_ROOT}/desc-gordon_res-0333_atlas.nii',
    'schaefer400': f'{ATLAS_ROOT}/desc-schaefer_res-0400_atlas.nii',
    'kong400': None
}
KONG_IDS = '/mnt/andromeda/Data/Kong2022_ArealMSHBM/HCP_subject_list.txt'
KONG_PARCELLATIONS = (
    '/mnt/andromeda/Data/Kong2022_ArealMSHBM/Parcellations/400/'
    'HCP_1029sub_400Parcels_Kong2022_gMSHBM.mat'
)
LH_MASK = nb.load(
    tflow.get('fsLR', density='32k', hemi='L', desc='nomedialwall')
).darrays[0].data.astype(bool)
RH_MASK = nb.load(
    tflow.get('fsLR', density='32k', hemi='R', desc='nomedialwall')
).darrays[0].data.astype(bool)
LH_COOR = nb.load(
    tflow.get('fsLR', density='32k', hemi='L', space=None, suffix='sphere')
).darrays[0].data[LH_MASK]
RH_COOR = nb.load(
    tflow.get('fsLR', density='32k', hemi='R', space=None, suffix='sphere')
).darrays[0].data[RH_MASK]


def _test_mode(
    kong_ids,
    kong_parcellations,
):
    import matplotlib.pyplot as plt
    import h5py
    kong_idx = kong_ids.index('106016')
    kong_asgt = np.concatenate((
        kong_parcellations['lh_labels_all'][LH_MASK, kong_idx],
        kong_parcellations['rh_labels_all'][RH_MASK, kong_idx],
    ))
    kong_atlas = np.where(
        (kong_asgt < 0)[..., None],
        0,
        np.eye(kong_asgt.max() + 1)[kong_asgt],
    ).squeeze().T[1:]
    T0 = _get_data(*get_hcp_dataset('106016', 'LR', get_confounds=True), denoising='mgtr+18')
    T1 = _get_data(*get_hcp_dataset('106016', 'RL', get_confounds=True), denoising='mgtr+18')
    Tp = np.linalg.lstsq(kong_atlas.T, np.concatenate((T0, T1), -1))[0]
    with h5py.File('/mnt/andromeda/Data/Kong2022_ArealMSHBM/FC/400/HCP_1029sub_400Parcels_Kong2022_gMSHBM_FC_group1.mat', 'r') as f:
        refconn = f['corr_mat_sub_group'][:]
    ref = refconn[-5]
    print(np.corrcoef(sym2vec(np.corrcoef(Tp)), sym2vec(ref))[0, 1])
    plt.imshow(np.corrcoef(Tp), vmin=-0.25, vmax=0.25, cmap='RdYlBu_r'); plt.colorbar(); plt.show()
    breakpoint()


def aggregate_entities():
    records = {}
    if 'MSC' in DATASETS:
        records['MSC'] = [
            SubjectRecord(ident=e, **MSC_RECORD_DEFAULTS)
            for e in MSC_SUBJECTS_VAL
        ]
    if 'HCP' in DATASETS:
        with open(f'{HCP_DATA_SPLIT_DEF_ROOT}/split_val.txt', 'r') as f:
            hcp_subjects_val = f.read().splitlines()
        with open(f'{HCP_DATA_SPLIT_DEF_ROOT}/split_template.txt', 'r') as f:
            hcp_subjects_val += f.read().splitlines()
        records['HCP'] = [
            SubjectRecord(ident=e, **HCP_RECORD_DEFAULTS)
            for e in hcp_subjects_val
        ]

    records = sum(
        [records[ds] for ds in DATASETS], []
    )
    return records


def aggregate_time_series(ds: str, name: str, split: int = 0):
    tasks = dict(TASKS[ds])
    if ds == 'MSC':
        subjects = MSC_SUBJECTS_VAL
        sessions = MSC_SESSIONS
        if split != 0:
            sessions = (2 * split - 1, 2 * split)
            sessions = tuple(f'{e:02d}' for e in sessions)
        runs = None
        rest_tasks = ('rest',)
    elif ds == 'HCP':
        with open(f'{HCP_DATA_SPLIT_DEF_ROOT}/split_val.txt', 'r') as f:
            subjects = f.read().splitlines()
        with open(f'{HCP_DATA_SPLIT_DEF_ROOT}/split_template.txt', 'r') as f:
            subjects += f.read().splitlines()
        sessions = None
        runs = ('LR', 'RL')
        rest_tasks = ('REST1', 'REST2')
        match split:
            case 1:
                tasks.pop('REST2')
                runs = ('LR',)
                rest_tasks = ('REST1',)
            case 2:
                tasks.pop('REST2')
                runs = ('RL',)
                rest_tasks = ('REST1',)
            case 3:
                tasks.pop('REST1')
                runs = ('LR',)
                rest_tasks = ('REST2',)
            case 4:
                tasks.pop('REST1')
                runs = ('RL',)
                rest_tasks = ('REST2',)
    records = [
        SubjectRecord(
            ds=ds,
            ident=e,
            image_pattern=(
                f'{OUTPUT_DIR}/ts/ds-{ds}_' +
                'sub-{subject}_ses-{session}_task-{task}_' + #'run-{run}'
                f'atlas-{name}_ts.npy'
            ),
            tasks=tuple((k, v) for k, v in tasks.items()),
            rest_tasks=rest_tasks,
            sessions=sessions,
        )
        for e in subjects
    ]
    return records


def aggregate_parcellations(ds: str, name: str, split: int = 0):
    tasks = dict(TASKS[ds])
    if ds == 'MSC':
        raise ValueError('Parcellation-based prediction is only compatible with HCP')
    elif ds == 'HCP':
        with open(f'{HCP_DATA_SPLIT_DEF_ROOT}/split_val.txt', 'r') as f:
            subjects = f.read().splitlines()
        with open(f'{HCP_DATA_SPLIT_DEF_ROOT}/split_template.txt', 'r') as f:
            subjects += f.read().splitlines()
        sessions = None
        runs = None
        tasks = ((None, None),)
        rest_tasks = ()
    records = [
        SubjectRecord(
            ds=ds,
            ident=e,
            image_pattern=(
                f'{OUTPUT_DIR}/parcellations/ds-{ds}_' +
                'subject-{subject}_' +
                f'split-{split}_atlas-{name}_probseg.npy'
            ),
            tasks=tasks,
            rest_tasks=rest_tasks,
            sessions=sessions,
        )
        for e in subjects
    ]
    return records


def template_parcellation(model, template, coor, name: str):
    visualise, _ = visdef()
    P = {
        compartment: jax.nn.softmax(
            model.regulariser[compartment].selectivity_distribution.log_prob(
                template[compartment]
            ) + model.regulariser[compartment].spatial_distribution.log_prob(
                coor[compartment]
            ),
            -1,
        )
        for compartment in ('cortex_L', 'cortex_R')
    }
    P = np.block([
        [P['cortex_L'], np.zeros_like(P['cortex_L'])],
        [np.zeros_like(P['cortex_R']), P['cortex_R']],
    ])
    visualise(
        name=f'atlas-{name}',
        array=P,
    )
    np.save(
        f'{OUTPUT_DIR}/parcellations/{name}.npy',
        np.asarray(P),
        allow_pickle=False,
    )


def create_parcellations(
    num_parcels: int = 200,
    start_epoch: Optional[int] = 15600,
    rerun_all: bool = False,
):
    os.makedirs(f'{OUTPUT_DIR}/parcellations/', exist_ok=True)
    key = jax.random.PRNGKey(SEED)
    visualise, _ = visdef()
    records = aggregate_entities()
    T = _get_data(
        records[0].get_dataset(), normalise=False, denoising=None
    )
    coor_L, coor_R = get_coors()
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
        dropout=0,
    )
    template_parcellation(
        model,
        template,
        coor={
            'cortex_L': coor_L,
            'cortex_R': coor_R,
        },
        name='groupTemplateInitOnly',
    )
    encode = eqx.filter_jit(encoder)
    #encode = encoder
    if start_epoch is not None:
        model = eqx.tree_deserialise_leaves(
            f'/tmp/parcellation_model_checkpoint{start_epoch}',
            like=model,
        )
        try:
            with open('/tmp/epoch_history.pkl', 'rb') as f:
                epoch_history = pickle.load(f)
        except FileNotFoundError:
            print('No epoch history found--starting new record')
        template_parcellation(
            model,
            template,
            coor={
                'cortex_L': coor_L,
                'cortex_R': coor_R,
            },
            name='groupTemplate',
        )
    forward_modes = [
        ('parametric', eqx.filter_jit(model.parametric_path)),
        ('full', eqx.filter_jit(model)),
    ]
    for i, record in enumerate(records):
        if not rerun_all:
            outputs = Path(f'{OUTPUT_DIR}').glob(
                f'parcellations/*{record.ident}split*_atlas-*_ts.npy'
            )
            if len(list(outputs)) == (2 * (NUM_SPLITS[record.ds] + 1)):
                logging.info(f'Found complete results for {record.ident}!')
                continue
        logging.info(
            f'Fetching records for subject {record.ident} '
            f'[{i + 1} / {len(records)}]'
        )
        data = [
            _get_data(*e) for e in record.rest_iterator()
        ]
        data = [np.asarray(e) for e in data if e is not None]
        num_records = len(data)
        if num_records < (2 * NUM_SPLITS[record.ds]):
            logging.warning(f'Insufficient data for {record.ident}!')
            continue
        block_size = num_records // NUM_SPLITS[record.ds]
        logging.info(
            f'Computing {2 * (NUM_SPLITS[record.ds] + 1)} parcellations '
            f'for subject {record.ident}'
        )
        for split in range(NUM_SPLITS[record.ds] + 1):
            match split:
                case 0:
                    data_split = data
                case _:
                    data_split = data[
                        ((split - 1) * block_size):(split * block_size)
                    ]
            T = jnp.concatenate(data_split, -1)
            encoder_result = encode(
                T=T,
                coor_L=coor_L,
                coor_R=coor_R,
                M=template,
            )
            for name, fwd in forward_modes:
                P, _, _ = fwd(
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
                    key=key, # Should do nothing
                )
                P = np.block([
                    [P['cortex_L'], np.zeros_like(P['cortex_R'])],
                    [np.zeros_like(P['cortex_L']), P['cortex_R']],
                ])
                if (i % 3) == 0:
                    visualise(
                        name=f'atlas-{name}{record.ident}split{split}',
                        array=P.T,
                    )
                np.save(
                    f'{OUTPUT_DIR}/parcellations/ds-{record.ds}_subject-{record.ident}_split-{split}_atlas-{name}_probseg.npy',
                    np.asarray(P),
                    allow_pickle=False,
                )
    assert 0


def prepare_timeseries():
    os.makedirs(f'{OUTPUT_DIR}/ts/', exist_ok=True)
    os.makedirs(f'{OUTPUT_DIR}/metrics/', exist_ok=True)
    key = jax.random.PRNGKey(SEED)
    records = aggregate_entities()

    with open(KONG_IDS) as f:
        kong_ids = f.read().split('\n')
    kong_parcellations = loadmat(KONG_PARCELLATIONS)
    atlas = {}
    visualise, _ = visdef()
    for baseline, path in BASELINES.items():
        if path is None:
            continue # Get it on a per-subject basis below
        atlas[baseline] = nb.load(path)
        atlas[baseline] = atlas[baseline].get_fdata().astype(int) - 1
        atlas[baseline] = np.where(
            (atlas[baseline] < 0)[..., None],
            0,
            np.eye(atlas[baseline].max() + 1)[atlas[baseline]],
        ).squeeze().T
        visualise(
            name=f'atlas-{baseline}',
            array=atlas[baseline].T,
        )
    #_test_mode(kong_ids, kong_parcellations)
    for template in ('groupTemplate', 'groupTemplateInitOnly'):
        atlas[template] = np.load(
            f'{OUTPUT_DIR}/parcellations/{template}.npy'
        ).T
    assert 0

    # EVALUATION
    model_keys = {}
    for model_type in ('full', 'parametric'):
        model_keys[model_type] = [model_type] + [
            f'{model_type}_split-{i + 1}'
            for i in range(max(NUM_SPLITS.values()))
        ]
    recon_err = {
        'ds': [], 'subject': [], 'session': [], 'task': [], 'run': [],
        **{b: [] for b in BASELINES},
        **{a: [] for a in atlas.keys()},
        **{f: [] for f in model_keys['full']},
        **{p: [] for p in model_keys['parametric']},
    }
    for i, record in enumerate(records):
        subject = record.ident
        for e, (task, session, run) in record.iterator(identifiers=True):
            logging.info(
                f'Computing time series and explained variance for subject='
                f'{record.ident} session={session} run={run} task={task}'
            )
            try:
                T = _get_data(
                    *e,
                    key=jax.random.fold_in(key, DATA_SAMPLER_KEY),
                )
            except Exception:
                T = None
            if T is None:
                logging.warning(
                    f'Data entity subject={subject} session={session} '
                    f'run={run} task={task} is absent. Skipping'
                )
                continue
            if record.ds == 'HCP':
                try:
                    kong_idx = kong_ids.index(subject)
                    kong_asgt = np.concatenate((
                        kong_parcellations['lh_labels_all'][LH_MASK, kong_idx],
                        kong_parcellations['rh_labels_all'][RH_MASK, kong_idx],
                    ))
                    atlas['kong400'] = np.where(
                        (kong_asgt < 0)[..., None],
                        0,
                        np.eye(kong_asgt.max() + 1)[kong_asgt],
                    ).squeeze().T[1:]
                except ValueError:
                    atlas['kong400'] = None
            else:
                atlas['kong400'] = None

            for model_type in ('parametric', 'full'):
                for i in range(max(NUM_SPLITS.values()) + 1):
                    match i:
                        case 0:
                            model_key = model_type
                        case _:
                            model_key = f'{model_type}_split-{i}'
                    model_path = Path(
                        f'{OUTPUT_DIR}/parcellations/subject-{record.ident}_'
                        f'split-{i}_atlas-{model_type}_probseg.npy'
                    )
                    if model_path.exists():
                        atlas[model_key] = np.load(model_path)
                    else:
                        atlas[model_key] = None
            ts = {
                name: jnp.linalg.lstsq(parc.T, T)[0]
                for name, parc in atlas.items()
                if parc is not None
            }
            recon_err_instance = {
                name: (
                    1 - ((parc.T @ ts[name] - T) ** 2).sum() /
                    (T ** 2).sum()
                ).item()
                if parc is not None
                else np.nan
                for name, parc in atlas.items()
            }
            recon_err['ds'] += [record.ds]
            recon_err['subject'] += [subject]
            recon_err['session'] += [session]
            recon_err['run'] += [run]
            recon_err['task'] += [task]
            for k, v in recon_err_instance.items():
                recon_err[k] += [v]
            for name, data in ts.items():
                np.save(
                    f'{OUTPUT_DIR}/ts/ds-{record.ds}_sub-{subject}_'
                    f'ses-{session}_task-{task}_run-{run}_atlas-{name}_ts.npy',
                    np.asarray(data),
                    allow_pickle=False,
                )

        pd.DataFrame(recon_err).to_csv(f'{OUTPUT_DIR}/metrics/recon_error.tsv', sep='\t', index=False)
    pd.DataFrame(recon_err).to_csv(f'{OUTPUT_DIR}/metrics/recon_error.tsv', sep='\t', index=False)
    assert 0


def recon_error_plot():
    import matplotlib.pyplot as plt
    import seaborn as sns
    recon_df = pd.read_csv(f'{OUTPUT_DIR}/metrics/recon_error.tsv', sep='\t')
    atlases = tuple(BASELINES.keys()) + ('full', 'parametric')
    datasets = recon_df.ds.unique()
    fig, ax = plt.subplots(len(datasets), 2, figsize=(12, 6 * len(datasets)), layout='tight')
    for i, dataset in enumerate(datasets):
        recon_ds_df = recon_df[recon_df.ds == dataset]
        recon_ds_df = recon_ds_df.set_index(['ds', 'subject', 'task', 'session', 'run'])
        avail_atlases = [
            k
            for k, v in
            recon_ds_df[list(atlases)].isna().all().to_dict().items()
            if not v
        ]
        missing = recon_ds_df[avail_atlases].isna().any(axis=1)
        recon_ds_df = recon_ds_df[~missing]
        recon_ds_df['full_min'] = recon_ds_df[
            [e for e in recon_ds_df.columns if 'full' in e]
        ].min(1, skipna=True)
        recon_ds_df['parametric_min'] = recon_ds_df[
            [e for e in recon_ds_df.columns if 'parametric' in e]
        ].min(1, skipna=True)
        to_plot = recon_ds_df[avail_atlases]
        sns.violinplot(to_plot, palette='rocket', ax=ax[i][0])
        ax[i][0].set_xlabel('Atlas')
        ax[i][0].set_ylabel('variance explained')
        ax[i][0].set_title(f'Reconstruction: {dataset}')
        to_plot = recon_ds_df.sort_values('full').reset_index()[avail_atlases + ['full_min', 'parametric_min']]
        to_plot_l = to_plot.select_dtypes('number').reset_index().melt(id_vars='index')
        is_min = [e[-3:] == 'min' for e in to_plot_l['variable']]
        to_plot_l['variable'] = [e[:-4] if e[-4:] == '_min' else e for e in to_plot_l['variable']]
        to_plot_l['kind'] = ['min' if e else '' for e in is_min]
        to_plot_l = to_plot_l.rename(columns={'index': 'arg sort full', 'value': 'variance explained', 'variable': 'atlas'})
        sns.lineplot(to_plot_l, x='arg sort full', y='variance explained', hue='atlas', style='kind', palette='rocket', linewidth=0.75, ax=ax[i][1])
        ax[i][1].set_title(f'Instance reconstruction: {dataset}')
    fig.savefig('/tmp/reconstruction_error.svg')


def incorporate_block(blockwise: Tensor, overall: Tuple[Tensor, int]):
    if len(blockwise):
        blockwise = (blockwise.mean(0), blockwise.shape[0])
        denom = (overall[1] + blockwise[1])
        overall = (overall[0] * overall[1] + blockwise[0] * blockwise[1]) / denom, denom
    return overall


def prepare_replication_material(model: str = 'full'):
    from hypercoil.loss.functional import js_divergence
    os.makedirs(f'{OUTPUT_DIR}/replication/', exist_ok=True)
    os.makedirs(f'{OUTPUT_DIR}/metrics/', exist_ok=True)
    block_size = 2
    for ds in DATASETS:
        ds_parcellations = sorted(tuple(Path(f'{OUTPUT_DIR}/parcellations/').glob(f'ds-{ds}*')))
        ds_parcellations = [
            e for e in
            sorted(tuple(Path(f'{OUTPUT_DIR}/parcellations/').glob(f'ds-{ds}*atlas-{model}*')))
            if re.match('.*split-[^0].*', str(e))
        ]
        num_parcellations = len(ds_parcellations)
        num_loci = np.load(ds_parcellations[0]).shape[-1]
        distance_matrix = jnp.zeros((num_parcellations, num_parcellations))
        coinstance_matrix = jnp.zeros((num_parcellations, num_parcellations)).astype(bool)
        intra = (jnp.zeros(num_loci,), 0)
        inter = (jnp.zeros(num_loci,), 0)
        confidence = (jnp.zeros(num_loci,), 0)
        nblocks_ax = np.ceil(num_parcellations / block_size).astype(int)
        nblocks = ((nblocks_ax + 1) * nblocks_ax / 2).astype(int)
        ax0 = list(ds_parcellations)
        index0 = 0
        while ax0:
            ax1 = list(ax0)
            if block_size == 1: ax1 = ax1[1:]
            index1 = index0
            block0, ax0 = ax0[:block_size], ax0[block_size:]
            ident0 = [e.parts[-1].split('_')[1] for e in block0[:5]]
            material0 = jnp.stack([np.load(e).T for e in block0])
            confidence = incorporate_block(material0.max(-1), confidence)
            while ax1:
                logging.info(f'Outer block [{index0 + block_size} / {num_parcellations}]')
                logging.info(f'Inner block [{index1 + block_size} / {num_parcellations}]')
                block1, ax1 = ax1[:block_size], ax1[block_size:]
                ident1 = [e.parts[-1].split('_')[1] for e in block1[:5]]
                block_coinstance = np.asarray(ident0)[:, None] == np.asarray(ident1)[None, :]
                coinstance_matrix = coinstance_matrix.at[
                    index0:(index0 + block_size), index1:(index1 + block_size)
                ].set(block_coinstance)
                coinstance_matrix = coinstance_matrix.at[
                    index1:(index1 + block_size), index0:(index0 + block_size)
                ].set(block_coinstance.T)
                if index1 == index0:
                    material1 = material0
                else:
                    material1 = jnp.stack([np.load(e).T for e in block1])
                div = js_divergence(material0[:, None, ...], material1[None, ...], axis=-1).squeeze(-1)
                block_distance = jnp.nanmean(jnp.sqrt(div), -1)
                distance_matrix = distance_matrix.at[
                    index0:(index0 + block_size), index1:(index1 + block_size)
                ].set(block_distance)
                distance_matrix = distance_matrix.at[
                    index1:(index1 + block_size), index0:(index0 + block_size)
                ].set(block_distance.T)
                if index1 == index0:
                    mask = jnp.zeros_like(block_coinstance).at[
                        jnp.triu_indices_from(block_coinstance, k=1)
                    ].set(1).astype(bool)
                else:
                    mask = jnp.ones_like(block_coinstance).astype(bool)
                intra = incorporate_block(div[block_coinstance & mask], intra)
                inter = incorporate_block(div[~block_coinstance & mask], inter)
                index1 += block_size
            index0 += block_size
        divoverall = (intra[0] * intra[1] + inter[0] * inter[1]) / (intra[0] + inter[1])
        np.save(f'{OUTPUT_DIR}/replication/ds-{ds}_atlas-{model}_distance.npy', np.asarray(distance_matrix), allow_pickle=False)
        np.save(f'{OUTPUT_DIR}/replication/ds-{ds}_atlas-{model}_coinstance.npy', np.asarray(coinstance_matrix), allow_pickle=False)
        np.save(f'{OUTPUT_DIR}/replication/ds-{ds}_atlas-{model}_divinter.npy', np.asarray(inter[0]), allow_pickle=False)
        np.save(f'{OUTPUT_DIR}/replication/ds-{ds}_atlas-{model}_divintra.npy', np.asarray(intra[0]), allow_pickle=False)
        np.save(f'{OUTPUT_DIR}/replication/ds-{ds}_atlas-{model}_divtotal.npy', np.asarray(divoverall), allow_pickle=False)
        np.save(f'{OUTPUT_DIR}/replication/ds-{ds}_atlas-{model}_confidence.npy', np.asarray(confidence[0]), allow_pickle=False)
        with open(f'{OUTPUT_DIR}/replication/ds-{ds}_atlas-{model}_divinterdenom', 'w') as f:
            f.write(str(inter[1]))
        with open(f'{OUTPUT_DIR}/replication/ds-{ds}_atlas-{model}_divintradenom', 'w') as f:
            f.write(str(intra[1]))
        with open(f'{OUTPUT_DIR}/replication/ds-{ds}_atlas-{model}_confidencedenom', 'w') as f:
            f.write(str(confidence[1]))


def discr(
    distance: jnp.ndarray,
    ident: np.ndarray,
) -> jnp.ndarray:
    key = ident[distance.argsort(-1)]
    replicate = (key[..., 0][..., None] == key[..., 1:])
    num_replicates = replicate.sum(-1, keepdims=True)
    cumrepl = replicate.cumsum(-1)
    violations = ((num_replicates - cumrepl) * ~replicate).sum(-1)
    num_replicates = num_replicates.squeeze()
    return 1 - violations / (
        (violations.size - 1 - num_replicates) * num_replicates
    )


def replication_analysis(permutation_key: int = 73, null_samples: int = 100):
    import matplotlib.pyplot as plt
    import pyvista as pv
    import seaborn as sns
    from hyve import (
        Cell,
        plotdef,
        add_surface_overlay,
        draw_surface_boundary,
        plot_to_image,
        save_figure,
        surf_from_archive,
        surf_scalars_from_array,
        vertex_to_face,
    )
    from hypercoil_examples.atlas.energy import medial_wall_arrays, curv_arrays
    layout = Cell() / Cell() << (1 / 2)
    layout = layout | Cell() | layout << (1 / 3)
    layout = layout | Cell() << (4 / 5)
    medialwall_L, medialwall_R = medial_wall_arrays()
    curv_L, curv_R = curv_arrays()
    parcellation = np.load(f'{OUTPUT_DIR}/parcellations/groupTemplate.npy').argmax(-1)
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
            view='lateral',
        ),
        4: dict(
            hemisphere='right',
            view='medial',
        ),
        5: dict(
            elements=['scalar_bar'],
        ),
    }
    layout = layout.annotate(annotations)
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
            'dataset',
            surf_scalars_from_array(
                'dataset',
                is_masked=True,
            ),
            vertex_to_face('dataset', interpolation='mean'),
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
            padding=0,
            canvas_size=(1500, 500),
            canvas_color=(0, 0, 0),
            fname_spec='scalars-{surfscalars}',
            scalar_bar_action='collect',
        ),
    )

    for ds in DATASETS:
        for model in ('parametric', 'full'):
            distance = np.load(f'{OUTPUT_DIR}/replication/ds-{ds}_atlas-{model}_distance.npy')
            coinstance = np.load(f'{OUTPUT_DIR}/replication/ds-{ds}_atlas-{model}_coinstance.npy')
            div_intra = np.load(f'{OUTPUT_DIR}/replication/ds-{ds}_atlas-{model}_divintra.npy')
            div_inter = np.load(f'{OUTPUT_DIR}/replication/ds-{ds}_atlas-{model}_divinter.npy')
            div_total = np.load(f'{OUTPUT_DIR}/replication/ds-{ds}_atlas-{model}_divtotal.npy')
            confidence = np.load(f'{OUTPUT_DIR}/replication/ds-{ds}_atlas-{model}_confidence.npy')
            levels, ident = jnp.unique(coinstance, axis=0, return_inverse=True)
            fig, ax = plt.subplots(figsize=(8, 8), layout='tight')
            ax.imshow(distance, cmap='rocket_r', vmin=0.15, vmax=0.3)
            ax.set_xticks([])
            ax.set_yticks([])
            num_instances = distance.shape[0]
            interlw = 1.5 if num_instances < 100 else 0.5
            for i, (l0, l1) in enumerate(zip(ident[1:], ident[:-1])):
                if l0 == l1 and num_instances < 100:
                    ax.axvline(num_instances - i - 1.5, color='white', linewidth=.5)
                    ax.axhline(i + 0.5, color='white', linewidth=.5)
                elif l0 != l1:
                    ax.axvline(num_instances - i - 1.5, color='white', linewidth=interlw)
                    ax.axhline(i + 0.5, color='white', linewidth=interlw)
            fig.savefig(f'/tmp/measure-distance_ds-{ds}_atlas-{model}.svg')
            per_instance_discr = discr(distance, ident)
            jnp.where(levels, per_instance_discr[None, :], jnp.nan)
            null_key = jax.random.PRNGKey(permutation_key)
            discr_null = [None for _ in range(null_samples)]
            for i in range(null_samples):
                ident_null = jax.random.permutation(jax.random.fold_in(null_key, i), ident)
                discr_null[i] = discr(distance, ident_null)
            discr_null = jnp.stack(discr_null)
            order = jnp.argsort(per_instance_discr)
            discr_null = discr_null[..., order]
            discr_ordered = per_instance_discr[order]
            discr_null_df = pd.DataFrame({i: discr_null[..., i] for i in range(num_instances)})
            plt.figure(figsize=(10, 8), layout='tight')
            if num_instances < 100:
                sns.swarmplot(discr_null_df, size=2, color='grey')
            else:
                sns.violinplot(discr_null_df, cut=0, color='grey')
            sns.despine()
            plt.scatter(range(num_instances), discr_ordered, color='purple', s=25)
            plt.xticks([])
            plt.xlabel('Data instance (arg sort)')
            plt.ylabel('Discriminability')
            plt.savefig(f'/tmp/measure-discr_ds-{ds}_atlas-{model}.svg')
            div_resid = div_inter - div_inter @ div_intra / jnp.linalg.norm(div_intra) ** 2 * div_intra
            for name, short, scalars in (
                ('JS divergence (within)', 'divintra', div_intra),
                ('JS divergence (between)', 'divinter', div_inter),
                ('JS divergence', 'divtotal', div_total),
                ('Delta divergence', 'divdelta', div_inter - div_intra),
                ('Divergence residuals', 'divresid', div_resid),
                ('confidence', 'confidence', confidence),
            ):
                if short == 'divresid':
                    cmap_negative = 'bone'
                else:
                    cmap_negative = None
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
                    fname_spec=f'measure-{short}_ds-{ds}_atlas-{model}',
                    load_mask=True,
                    dataset_array=scalars,
                    dataset_cmap='rocket',
                    dataset_cmap_negative=cmap_negative,
                    dataset_scalar_bar_style={
                        'name': name,
                        'orientation': 'v',
                    },
                    curv_array_left=curv_L,
                    curv_array_right=curv_R,
                    curv_cmap='gray',
                    curv_clim=(-5e-1, 5e-1),
                    curv_alpha=0.3,
                    parcellation_array=parcellation,
                    parcellation_boundary_color='black',
                    parcellation_boundary_alpha=0.6,
                    medialwall_array_left=medialwall_L,
                    medialwall_array_right=medialwall_R,
                    medialwall_cmap='binary',
                    medialwall_clim=(0.99, 1),
                    medialwall_below_color=(0, 0, 0, 0),
                )
    assert 0


def corr_kernel(X, y=None):
    if y is None: y = X
    val = np.corrcoef(X, y)[:X.shape[-2], X.shape[-2]:]
    return val



def grid_search(grid_params: Mapping | Sequence[Mapping]):
    """
    Return an iterator over all param configurations, because
    sklearn CV is unfortunately useless for our case
    """
    from functools import reduce
    from itertools import chain, product
    if isinstance(grid_params, Mapping):
        grid_params = [grid_params]
    yield from (
        reduce(lambda e, f: {**e, **f}, z, {}) for z in
        chain(*(
            product(*[[{k: e} for e in v] for k, v in p.items()])
            for p in grid_params
        ))
    )


def predict(
    predict_on: Literal['connectomes', 'parcellations'],
    num_parcels: int = 200, # not needed, for now
):
    import pathlib
    from functools import partial
    from sklearn import svm
    from sklearn.base import clone
    from sklearn.kernel_ridge import KernelRidge
    from sklearn.metrics.pairwise import linear_kernel, rbf_kernel, cosine_similarity
    from sklearn.model_selection import GridSearchCV, cross_validate, GroupKFold, GroupShuffleSplit, StratifiedGroupKFold
    from hypercoil_examples.atlas.data import TASKS, TASKS_TARGETS
    NUM_TRIALS = 5 # 100 #
    # Let's just always have the same count of inner and outer folds
    FOLDS_OUTER = 4 # 20 #
    FOLDS_INNER = 4 # 20 #
    # CBIG likes this one, but it feels like cheating if I'm being honest
    METRIC = 'corr' # 'predictiveCOD' #
    models = list(BASELINES.keys()) + ['parametric', 'full', 'groupTemplate', 'groupTemplateInitOnly']
    ds_ = []
    model_ = []
    target_ = []
    score_ = []
    scores = {}
    results = {}
    match predict_on:
        case 'connectomes':
            aggregate = aggregate_time_series
        case 'parcellations':
            aggregate = aggregate_parcellations
            sphere_coor = np.concatenate((LH_COOR, RH_COOR))
        case _:
            raise ValueError(f'Invalid feature set: {predict_on}')
    for ds, tasks in TASKS.items():
        objectives = ('task',)
        if ds == 'HCP':
            extra_measures = pd.read_csv(HCP_EXTRA_SUBJECT_MEASURES, sep='\t')
            objectives = ('task', 'beh')
        if predict_on == 'parcellations':
            objectives = tuple(obj for obj in objectives if obj != 'task')
        scores[ds] = {}
        results[ds] = {}
        for objective, name in product(objectives, models):
            scores[ds][name] = {}
            results[ds][name] = {}
            if objective == 'task' and ds == 'MSC':
                split = 0
            else:
                split = 1
            records = sorted(
                aggregate(ds, name, split=split),
                key=lambda x: x.ident,
            )
            features = []
            targets = []
            subject_ids = []
            i = 0
            for record in records:
                subject = record.ident
                if objective == 'task' or predict_on == 'parcellations':
                    iterator = record.iterator
                else:
                    iterator = record.rest_iterator
                for features_path, (task, session, run) in iterator(
                    identifiers=True,
                    get_confounds=False,
                ):
                    data = np.load(features_path)
                    subject_ids += [subject]
                    if predict_on == 'connectomes':
                        features += [sym2vec(np.corrcoef(data))]
                    elif predict_on == 'parcellations':
                        parcel_sizes = data.sum(-1)
                        parcel_coor = data @ sphere_coor
                        parcel_coor = (
                            parcel_coor / np.linalg.norm(parcel_coor, axis=-1, keepdims=True)
                        )
                        features += [
                            np.concatenate(
                                (parcel_sizes, parcel_coor.ravel())
                            )
                        ]
                    if objective == 'task':
                        task = dict(TASKS[ds])[task]
                        targets += [
                            jnp.zeros(len(TASKS_TARGETS[ds])).at[
                                TASKS_TARGETS[ds].index(task)
                            ].set(1.),
                        ]
                    else:
                        targets += [
                                extra_measures.loc[
                                extra_measures['Subject'] == int(subject)
                            ].values[0][1:]
                        ]
            if len(features) == 0:
                continue
            features = np.asarray(features)
            targets = np.asarray(targets)
            ids_ref = np.asarray(subject_ids)
            subject_ids = (
                np.array(subject_ids)[..., None] ==
                np.unique(subject_ids)[None, ...]
            ).argmax(-1)
            if objective == 'task':
                targets = [(targets.argmax(-1), 'task', 'categorical')]
            elif ds == 'HCP':
                age_targets = (
                    targets[..., -3:].argmax(-1),
                    'age',
                    'categorical',
                )
                continuous_targets = [
                    (e.astype(float), list(HCP_MEASURES.keys())[i], 'continuous')
                    for i, e in enumerate(targets[..., :-3].T)
                ]
                targets = [age_targets] + continuous_targets
            # Run nested cross-validation
            # Based on https://scikit-learn.org/stable/auto_examples/ ...
            # ... model_selection/plot_nested_cross_validation_iris.html
            # EDIT: sklearn approach is useless with precomputed kernel, and
            #       miserably slow/wasteful if the kernel is computed every
            #       time, especially because KRR explicitly doesn't allow
            #       vectorisation. Weird design choice, but whatever,
            #       Add it to the list of reasons I dislike turn-key,
            #       hack-unfriendly software. Stage and remove this after
            #       we manually build the CV loop in a non-terrible IDE
            linear_kernels = [linear_kernel(features)]
            rbf_kernels = [
                rbf_kernel(features, gamma=gamma)
                for gamma in (
                    .001,
                    .0001,
                    1 / (features.shape[-1] * features.var()),
                )
            ]
            # corr and cosine *should* be the same since we're centering our
            # data, and in practice they *are* very very close, but whatever
            corr_kernels = [corr_kernel(features)]
            cosine_kernels = [cosine_similarity(features)]
            kernel_spec = linear_kernels + rbf_kernels + corr_kernels + cosine_kernels
            for i, (target, target_name, var_kind) in enumerate(targets):
                nested_scores = np.zeros(NUM_TRIALS)
                scores[ds][name][target_name] = {}
                results[ds][name][target_name] = [None for _ in range(NUM_TRIALS)]
                #regularisation_grid = [0.1, 1, 10, 100, 1000]
                regularisation_grid = [
                    1e-8, 0.00001, 0.0001, 0.001, 0.004, 0.007,
                    0.01, 0.04, 0.07, 0.1,
                    0.4, 0.7, 1, 1.5, 2, 2.5, 3, 3.5, 4, 5, 10, 15,
                    20, 30, 40, 50, 60, 70, 80, 100, 150, 200,
                    300, 500, 700, 1000, 10000, 100000, 1000000,
                ]
                if var_kind == 'categorical':
                    model_base = svm.SVC()
                    cv_base = StratifiedGroupKFold
                    cv_params = {'shuffle': True}
                    regularisation = {'C': [1 / (2 * alpha) for alpha in regularisation_grid]}
                elif var_kind == 'continuous':
                    model_base = KernelRidge()
                    cv_base = GroupShuffleSplit
                    cv_params = {
                        'test_size': 1 / FOLDS_OUTER,
                    }
                    regularisation = {
                        'alpha': [alpha for alpha in regularisation_grid]
                    }
                param_grid = [
                    {
                        'X': kernel_spec,
                        'kernel': ['precomputed'],
                        **regularisation,
                    },
                ]
                valid_index = np.where(~np.isnan(target))[0]
                valid_features = features[valid_index]
                valid_target = target[valid_index]
                valid_group_id = subject_ids[valid_index]
                for j in range(NUM_TRIALS):
                    logging.info(
                        f'Running trial {j + 1} / {NUM_TRIALS} for '
                        f'parcellation {name}, measure {target_name}, '
                        f'{len(features)} instances'
                    )
                    inner_cv = cv_base(
                        n_splits=FOLDS_INNER,
                        random_state=i * NUM_TRIALS + j,
                        **cv_params,
                    )
                    outer_cv = cv_base(
                        n_splits=FOLDS_OUTER,
                        random_state=i * NUM_TRIALS + j,
                        **cv_params,
                    )
                    non_nested_scores = np.zeros(FOLDS_OUTER)
                    for k, (train_index_o, test_index_o) in enumerate(
                        outer_cv.split(
                            X=valid_features,
                            y=valid_target,
                            groups=valid_group_id,
                        )
                    ):
                        for l, (train_index_i, test_index_i) in enumerate(
                            inner_cv.split(
                                X=valid_features[train_index_o],
                                y=valid_target[train_index_o],
                                groups=valid_group_id[train_index_o],
                            )
                        ):
                            logging.info(
                                f'Nested CV: outer {k + 1} / {FOLDS_OUTER}, '
                                f'inner {l + 1} / {FOLDS_INNER}'
                            )
                            best_score = -np.inf
                            best_params = best_kernel = None
                            for param_cfg in grid_search(param_grid):
                                X = param_cfg.pop('X')
                                X_valid = X[valid_index][:, valid_index]
                                X_split = X_valid[train_index_o][:, train_index_o]
                                X_train = X_split[train_index_i][:, train_index_i]
                                X_test = X_split[test_index_i][:, train_index_i]
                                y_split = valid_target[train_index_o]
                                y_train = y_split[train_index_i]
                                y_test = y_split[test_index_i]
                                estimator = clone(model_base).set_params(
                                    **clone(param_cfg, safe=False)
                                )
                                estimator.fit(X_train, y_train)
                                y_pred = estimator.predict(X_test)
                                # CBIG reference implementation:
                                # https://github.com/ThomasYeoLab/CBIG/blob/master/ ...
                                # utilities/matlab/predictive_models/utilities/ ...
                                # CBIG_compute_prediction_acc_and_loss.m
                                match METRIC:
                                    case 'COD':
                                        score = (
                                            1 -
                                            ((y_pred - y_test) ** 2).sum() /
                                            ((y_test.mean() - y_test) **2).sum()
                                        )
                                    case 'predictiveCOD':
                                        score = (
                                            1 -
                                            ((y_pred - y_test) ** 2).sum() /
                                            ((y_train.mean() - y_test) **2).sum()
                                        )
                                    case 'corr':
                                        score = np.corrcoef(y_pred, y_test)[0, 1]
                                    case 'MAE':
                                        score = np.abs(y_pred - y_test).mean()
                                    case 'MSE':
                                        score = ((y_pred - y_test) ** 2).mean()
                                    case 'MAEnorm':
                                        score = np.abs(y_pred - y_test).mean() / y_test.std()
                                    case 'MSEnorm':
                                        # Don't know why it's variance here and
                                        # std in the other case, but I'm just
                                        # trying to be consistent with the
                                        # implementation from CBIG
                                        score = ((y_pred - y_test) ** 2).mean() / y_test.var()
                                score = estimator.score(X_test, y_test)
                                if score > best_score:
                                    best_score = score
                                    best_params = param_cfg
                                    best_kernel = X_valid
                        logging.info(f'Best score: {best_score}')
                        X_train = best_kernel[train_index_o][:, train_index_o]
                        X_test = best_kernel[test_index_o][:, train_index_o]
                        y_train = valid_target[train_index_o]
                        y_test = valid_target[test_index_o]
                        estimator = clone(model_base).set_params(
                            **clone(best_params, safe=False)
                        )
                        estimator.fit(X_train, y_train)
                        non_nested_scores[k] = estimator.score(X_test, y_test)
                    nested_scores[j] = non_nested_scores.mean()
                    ds_ += [ds]
                    model_ += [name]
                    target_ += [target_name]
                    score_ += [nested_scores[j].item()]

                scores[ds][name][target_name] = nested_scores
    results_df = pd.DataFrame({'ds': ds_, 'model': model_, 'target': target_, 'score': score_})
    results_df.to_csv(f'{OUTPUT_DIR}/metrics/on-{predict_on}_metric-{METRIC}_prediction.tsv', sep='\t', index=None)
    assert 0
    # {k: np.mean(v['age']) for k, v in scores['HCP'].items()}


def plot_prediction_result():
    import matplotlib.pyplot as plt
    import seaborn as sns
    # hardcode for now
    df = pd.read_csv('/mnt/andromeda/Data/atlas_ts/metrics/on-connectomes_metric-predictiveCOD_prediction.tsv', sep='\t')
    df['variable'] = [f'{j} ({i})' for i, j in zip(df.ds, df.target)]
    plt.figure(figsize=(12, 9), layout='tight')
    sns.boxplot(df, x='variable', y='score', hue='model')
    plt.xticks(rotation=80); plt.axhline(ls=':', color='grey')
    plt.savefig('/tmp/prediction.svg')


def main(
    extract_ts: bool = False,
    num_parcels: int = 200,
):
    logging.basicConfig(
        level=logging.INFO,
        format='[%(asctime)s] [%(levelname)s] - %(message)s',
    )
    #create_parcellations(num_parcels=num_parcels)
    #prepare_timeseries()
    #recon_error_plot()
    #prepare_replication_material('full')
    #prepare_replication_material('parametric');
    #replication_analysis()
    predict(predict_on='parcellations', num_parcels=num_parcels)


if __name__ == '__main__':
    main()
