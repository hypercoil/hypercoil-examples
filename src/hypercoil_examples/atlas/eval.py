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
from typing import Any, Mapping, Optional, Tuple

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
from hypercoil_examples.atlas.const import (
    HCP_DATA_SPLIT_DEF_ROOT,
    HCP_EXTRA_SUBJECT_MEASURES,
)
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
from hypercoil_examples.atlas.train import (
    ENCODER_ARCH,
    BLOCK_ARCH,
    SERIAL_INJECTION_SITES,
    VMF_SPATIAL_KAPPA,
    VMF_SELECTIVITY_KAPPA,
    FIXED_KAPPA,
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
TASKS = {
    'MSC': (
        'rest', 'motor', 'glasslexical',
        'memoryfaces', 'memoryscenes', 'memorywords',
    ),
    'HCP': (
        'REST', 'EMOTION', 'GAMBLING',
        'LANGUAGE', 'MOTOR', 'RELATIONAL', 'SOCIAL', 'WM',
    ),
}
DATASETS = ('HCP',) # 'MSC')

DATA_SHUFFLE_KEY = 7834
DATA_SAMPLER_KEY = 9902
READOUT_INIT_KEY = 5310

OUTPUT_DIR = '/mnt/andromeda/Data/atlas_ts/'
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


def prepare_timeseries(
    num_parcels: int = 200,
    start_epoch: Optional[int] = 24101,
):
    key = jax.random.PRNGKey(SEED)
    val_entities = {}
    if 'MSC' in DATASETS:
        val_entities = {**val_entities, 'MSC': [
            {'ds': 'MSC', 'session': ses, 'subject': sub, 'task': task}
            for ses, sub, task in product(
                MSC_SESSIONS, MSC_SUBJECTS_VAL, TASKS['MSC']
            )
        ]}
    if 'HCP' in DATASETS:
        with open(f'{HCP_DATA_SPLIT_DEF_ROOT}/split_val.txt', 'r') as f:
            hcp_subjects_val = f.read().splitlines()
        val_entities = {**val_entities, 'HCP': [
            {'ds': 'HCP', 'run': run, 'subject': sub, 'task': task}
            for run, sub, task in product(
                ('LR', 'RL'), hcp_subjects_val, TASKS['HCP']
            )
        ]}

    val_entities = sum(
        [val_entities[ds] for ds in DATASETS], []
    )

    with open(KONG_IDS) as f:
        kong_ids = f.read().split('\n')
    kong_parcellations = loadmat(KONG_PARCELLATIONS)
    atlas = {}
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

    T = _get_data(get_msc_dataset('01', '01'), normalise=False, gsr=False)
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
    #encode = eqx.filter_jit(encoder)
    encode = encoder
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

    # EVALUATION
    recon_err = {
        'ds': [], 'subject': [], 'session': [], 'task': [],
        'full': [], 'parametric': [], **{b: [] for b in BASELINES},
    }
    for i in range(len(val_entities)):
        entity = val_entities[i]
        key_e = jax.random.fold_in(key, i)

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
                    pad_to_size=None,
                    key=jax.random.fold_in(key_e, DATA_SAMPLER_KEY),
                )
                atlas['kong400'] = None
            elif ds == 'HCP':
                subject = entity.get('subject')
                session = entity.get('run')
                task = entity.get('task')
                T = _get_data(
                    *get_hcp_dataset(subject, session, task, get_confounds=True,),
                    normalise=False,
                    gsr=False,
                    pad_to_size=None,
                    key=jax.random.fold_in(key_e, DATA_SAMPLER_KEY),
                )
                # If it's HCP, evaluate the Kong parcellation
                try:
                    kong_idx = kong_ids.index(entity['subject'])
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
        except FileNotFoundError:
            print(
                f'Data entity {entity} is absent. '
                'Skipping'
            )
            continue
        print(f'Evaluation {i} (ds-{ds} sub-{subject} ses-{session} task-{task})')
        if jnp.any(jnp.isnan(T)):
            print(
                f'Invalid data for entity sub-{subject} ses-{session}. '
                'Skipping'
            )
            breakpoint()
            continue
        T_in = jnp.where(
            jnp.isclose(T.std(-1), 0)[..., None],
            jax.random.normal(jax.random.fold_in(key_e, 54), T.shape),
            T,
        )

        encoder_result = encode(
            T=T_in,
            coor_L=coor_L,
            coor_R=coor_R,
            M=template,
        )
        ts = {}

        for name, fwd in [('parametric', model.parametric_path), ('full', model)]:
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
            (_, (_, _, _, _, T_in)) = encoder_result
            csize = {
                compartment: encoder.temporal.encoders[0].limits[compartment]
                for compartment in ('cortex_L', 'cortex_R')
            }
            compartment_T = {
                compartment: T_in[..., start:(start + size), :]
                for compartment, (start, size) in csize.items()
            }
            compartment_ts = {
                compartment: jnp.linalg.lstsq(
                    P[compartment].T, TT
                )[0]
                for compartment, TT in compartment_T.items()
            }
            ts[name] = jnp.concatenate(list(compartment_ts.values()), 0)
            recon_err[name] += [
                (1 - sum({
                    compartment: (
                        ((P[compartment].T @ ts - compartment_T[compartment]) ** 2).sum() /
                        (T_in **2).sum()
                    )
                    for compartment, ts in compartment_ts.items()
                }.values())).item()
            ]

        T_in = T - T.mean(-1, keepdims=True)
        T_in = T_in / T_in.std(-1, keepdims=True)
        T_in = jnp.where(jnp.isnan(T_in), 0, T_in)
        gs = T_in.mean(0, keepdims=True)
        T_in = residualise(T_in, gs)
        T_in = jnp.where(
            jnp.isclose(T_in.std(-1), 0)[..., None],
            jax.random.normal(jax.random.fold_in(key_e, 54), T_in.shape),
            T_in,
        )

        ts = {
            **ts,
            **{
                baseline: jnp.linalg.lstsq(atlas[baseline].T, T_in)[0]
                for baseline in atlas
                if atlas[baseline] is not None
            },
        }
        recon_err_baseline = {
            baseline: (
                1 - ((atlas[baseline].T @ ts[baseline] - T_in) ** 2).sum() /
                (T_in ** 2).sum()
            )
            if atlas[baseline] is not None
            else jnp.asarray(jnp.inf)
            for baseline in atlas
        }
        recon_err['ds'] += [ds]
        recon_err['subject'] += [subject]
        recon_err['session'] += [session]
        recon_err['task'] += [task]
        for k, v in recon_err_baseline.items():
            recon_err[k] += [v.item()]
        for name, data in ts.items():
            np.save(
                f'{OUTPUT_DIR}/ds-{ds}_sub-{subject}_ses-{session}_task-{task}_atlas-{name}_ts.npy',
                np.asarray(data),
                allow_pickle=False,
            )
        if i % 10 == 0:
            pd.DataFrame(recon_err).to_csv(f'{OUTPUT_DIR}/recon_error.tsv', sep='\t')
    pd.DataFrame(recon_err).to_csv(f'{OUTPUT_DIR}/recon_error.tsv', sep='\t')


def predict(
    num_parcels: int = 200, # not needed, for now
):
    import pathlib
    from sklearn import svm
    from sklearn.model_selection import GridSearchCV, cross_val_score, KFold
    DATA_GETTER = {
        'MSC': get_msc_dataset,
        'HCP': get_hcp_dataset,
    }
    models = list(BASELINES.keys()) + ['parametric', 'full']
    scores = {}
    TASKS.pop('MSC')
    for ds, tasks in TASKS.items():
        if ds == 'HCP':
            extra_measures = pd.read_csv(HCP_EXTRA_SUBJECT_MEASURES, sep='\t')
        scores[ds] = {}
        for name in models:
            tss = sorted(
                pathlib.Path(
                    f'{OUTPUT_DIR}'
                ).glob(
                    f'ds-{ds}_sub-*_ses-*_task-*_atlas-{name}_ts.npy'
                )
            )
            num_parcels = np.load(tss[0]).shape[0]
            connectomes = np.zeros(
                (len(tss), int(num_parcels * (num_parcels - 1) / 2))
            )
            task_targets = np.zeros((len(tss), len(tasks)))
            if ds == 'HCP':
                extra_targets = np.zeros((len(tss), extra_measures.shape[1] - 1))
            for i, ts in enumerate(tss):
                data = np.load(ts)
                subject_id = re.match(
                    '.*_sub-(?P<subject>[A-Z0-9]*)_.*',
                    str(ts),
                ).group('subject')
                session_id = re.match(
                    '.*_ses-(?P<session>[A-Z0-9]*)_.*',
                    str(ts),
                ).group('session')
                taskname = [
                    i for i in str(ts).split('/')[-1].split('_')
                    if i[:4] == 'task'
                ][0][5:]
                get_dataset = DATA_GETTER[ds]
                _, confounds = get_dataset(
                    subject_id,
                    session_id,
                    taskname,
                    get_confounds=True,
                )
                if os.path.exists(confounds):
                    # Censor the parcellated timeseries
                    data = _get_data(
                        cifti=None,
                        data=data,
                        confounds=confounds,
                        normalise=False,
                        gsr=False,
                        filter_rps=True,
                        censor_thresh=0.15,
                        pad_to_size=None,
                        censor_method='drop',
                        key=None,
                    )
                else:
                    logging.warning(
                        f'No confounds found for {subject_id} {session_id} {taskname}'
                    )
                if taskname[:4] == 'REST':
                    taskname = 'REST'
                connectomes[i] = sym2vec(np.corrcoef(data))
                task_targets[i, tasks.index(taskname)] = 1
                if ds == 'HCP':
                    extra_targets[i] = extra_measures.loc[
                        extra_measures['Subject'] == int(subject_id)
                    ].values[0][1:]
            # Run nested cross-validation
            # Based on https://scikit-learn.org/stable/auto_examples/ ...
            # ... model_selection/plot_nested_cross_validation_iris.html
            NUM_TRIALS = 100 # 5 #
            FOLDS_OUTER = 20 # 4 #
            FOLDS_INNER = 20 # 4 #
            param_grid = [
                {
                    'kernel': ['linear'],
                    'C': [0.1, 1, 10, 100, 1000],
                },
                {
                    'kernel': ['rbf'],
                    'C': [0.1, 1, 10, 100, 1000],
                    'gamma': [0.001, 0.0001, 'scale'],
                },
                {
                    # Kong et al use a correlation kernel
                    'kernel': [lambda x: np.corrcoef(x)],
                    'C': [0.1, 1, 10, 100, 1000],
                },
            ]
            model_base = svm.SVC()
            non_nested_scores = np.zeros(NUM_TRIALS)
            nested_scores = np.zeros(NUM_TRIALS)
            for i in range(NUM_TRIALS):
                logging.info(f'Running trial {i}')
                inner_cv = KFold(
                    n_splits=FOLDS_INNER,
                    shuffle=True,
                    random_state=i,
                )
                outer_cv = KFold(
                    n_splits=FOLDS_OUTER,
                    shuffle=True,
                    random_state=i,
                )
                # Non_nested parameter search and scoring
                # model = GridSearchCV(
                #     estimator=model_base,
                #     param_grid=param_grid,
                #     cv=outer_cv,
                #     n_jobs=-1,
                # )
                #model.fit(connectomes, task_targets.argmax(-1))
                #non_nested_scores[i] = model.best_score_
                # Nested CV with parameter optimization
                model = GridSearchCV(
                    estimator=model_base,
                    param_grid=param_grid,
                    cv=inner_cv,
                    n_jobs=-1,
                )
                nested_score = cross_val_score(
                    model,
                    X=connectomes,
                    y=task_targets.argmax(-1),
                    cv=outer_cv,
                    n_jobs=-1,
                )
                nested_scores[i] = nested_score.mean()

            scores[ds][name] = (nested_scores, non_nested_scores)
    assert 0


def main(
    extract_ts: bool = False,
    num_parcels: int = 200,
):
    if extract_ts:
        prepare_timeseries(num_parcels=num_parcels)
    predict(num_parcels=num_parcels)


if __name__ == '__main__':
    main()
