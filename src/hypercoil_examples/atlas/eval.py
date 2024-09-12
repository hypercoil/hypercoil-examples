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
from typing import Any, Mapping, Optional, Sequence, Tuple

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
TASKS_TARGETS = {
    k: tuple(sorted(set(v[1] for v in vs)))
    for k, vs in TASKS.items()
}
DATASETS = ('HCP',) # 'MSC')

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


def prepare_timeseries(
    num_parcels: int = 200,
    start_epoch: Optional[int] = 15600,
):
    key = jax.random.PRNGKey(SEED)
    val_entities = {}
    if 'MSC' in DATASETS:
        val_entities = {**val_entities, 'MSC': [
            {'ds': 'MSC', 'session': ses, 'subject': sub, 'task': task}
            for ses, sub, task in product(
                MSC_SESSIONS, MSC_SUBJECTS_VAL, TASKS_FILES['MSC']
            )
        ]}
    if 'HCP' in DATASETS:
        with open(f'{HCP_DATA_SPLIT_DEF_ROOT}/split_val.txt', 'r') as f:
            hcp_subjects_val = f.read().splitlines()
        val_entities = {**val_entities, 'HCP': [
            {'ds': 'HCP', 'run': run, 'subject': sub, 'task': task}
            for run, sub, task in product(
                ('LR', 'RL'), hcp_subjects_val, TASKS_FILES['HCP']
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


def corr_kernel(X, y=None):
    if y is None: y = X
    val = np.corrcoef(X, y)[:X.shape[-2], X.shape[-2]:]
    return val


def predict(
    num_parcels: int = 200, # not needed, for now
):
    import pathlib
    from functools import partial
    from sklearn import svm
    from sklearn.base import clone
    from sklearn.kernel_ridge import KernelRidge
    from sklearn.metrics.pairwise import linear_kernel, rbf_kernel, cosine_similarity
    from sklearn.model_selection import GridSearchCV, cross_validate, GroupKFold, GroupShuffleSplit, StratifiedGroupKFold
    DATA_GETTER = {
        'MSC': get_msc_dataset,
        'HCP': get_hcp_dataset,
    }
    models = list(BASELINES.keys()) + ['parametric', 'full']
    scores = {}
    results = {}
    TASKS.pop('MSC')
    for ds, tasks in TASKS.items():
        if ds == 'HCP':
            extra_measures = pd.read_csv(HCP_EXTRA_SUBJECT_MEASURES, sep='\t')
        scores[ds] = {}
        results[ds] = {}
        for name in models:
            scores[ds][name] = {}
            results[ds][name] = {}
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
            task_targets = np.zeros((len(tss), len(TASKS_TARGETS[ds])))
            subject_ids = [None for _ in range(len(tss))]
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
                if taskname[:4] == 'REST':
                    taskname = 'REST'
                subject_ids[i] = subject_id
                connectomes[i] = sym2vec(np.corrcoef(data))
                task_targets[i, TASKS_TARGETS[ds].index(dict(TASKS[ds])[taskname])] = 1
                if ds == 'HCP':
                    extra_targets[i] = extra_measures.loc[
                        extra_measures['Subject'] == int(subject_id)
                    ].values[0][1:]
            ids_ref = np.array(subject_ids)
            subject_ids = (
                np.array(subject_ids)[..., None] ==
                np.unique(subject_ids)[None, ...]
            ).argmax(-1)
            task_targets = (task_targets.argmax(-1), 'task', 'categorical')
            targets = [task_targets]
            if ds == 'HCP':
                age_targets = (
                    extra_targets[..., -3:].argmax(-1),
                    'age',
                    'categorical',
                )
                continuous_targets = [
                    (e, list(HCP_MEASURES.keys())[i], 'continuous')
                    for i, e in enumerate(extra_targets[..., :-3].T)
                ]
                targets = [task_targets] + [age_targets] + continuous_targets
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
            NUM_TRIALS = 5 # 100 #
            # Let's just always have the same count of inner and outer folds
            FOLDS_OUTER = 4 # 20 #
            FOLDS_INNER = 4 # 20 #
            # CBIG likes this one, but it feels like cheating if I'm being honest
            METRIC = 'predictive_COD' # 'corr' #
            linear_kernels = [linear_kernel(connectomes)]
            rbf_kernels = [
                rbf_kernel(connectomes, gamma=gamma)
                for gamma in (
                    .001,
                    .0001,
                    1 / (connectomes.shape[-1] * connectomes.var()),
                )
            ]
            # corr and cosine *should* be the same since we're centering our
            # data, and in practice they *are* very very close, but whatever
            corr_kernels = [corr_kernel(connectomes)]
            cosine_kernels = [cosine_similarity(connectomes)]
            kernel_spec = linear_kernels + rbf_kernels + corr_kernels + cosine_kernels
            for i, (target, target_name, var_kind) in enumerate(targets[3:4]):
                nested_scores = np.zeros(NUM_TRIALS)
                scores[ds][name][target_name] = {}
                results[ds][name][target_name] = [None for _ in range(NUM_TRIALS)]
                regularisation_grid = [0.1, 1, 10, 100, 1000]
                regularisation_grid = [
                    1e-8, 0.00001, 0.0001, 0.001, 0.004, 0.007, 0.01, 0.04, 0.07, 0.1,
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
                valid_connectomes = connectomes[valid_index]
                valid_target = target[valid_index]
                valid_group_id = subject_ids[valid_index]
                for j in range(NUM_TRIALS):
                    logging.info(
                        f'Running trial {j + 1} / {NUM_TRIALS} for '
                        f'parcellation {name}, measure {target_name}'
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
                            X=valid_connectomes,
                            y=valid_target,
                            groups=valid_group_id,
                        )
                    ):
                        for l, (train_index_i, test_index_i) in enumerate(
                            inner_cv.split(
                                X=valid_connectomes[train_index_o],
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
                                    case 'predictive_COD':
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
                                    case 'MAE_norm':
                                        score = np.abs(y_pred - y_test).mean() / y_test.std()
                                    case 'MSE_norm':
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
                # for j in range(NUM_TRIALS):
                #     logging.info(
                #         f'Running trial {i} for parcellation '
                #         f'{name}, measure {target_name}'
                #     )
                #     inner_cv = cv_base(
                #         n_splits=FOLDS_INNER,
                #         random_state=i * NUM_TRIALS + j,
                #         **cv_params,
                #     )
                #     outer_cv = cv_base(
                #         n_splits=FOLDS_OUTER,
                #         random_state=i * NUM_TRIALS + j,
                #         **cv_params,
                #     )
                #     model = GridSearchCV(
                #         estimator=model_base,
                #         param_grid=param_grid,
                #         cv=inner_cv,
                #         error_score='raise',
                #         #n_jobs=-1,
                #     )
                #     nested_result = cross_validate(
                #         model,
                #         X=connectomes,
                #         y=target,
                #         groups=subject_ids,
                #         cv=outer_cv,
                #         return_estimator=True,
                #         verbose=3,
                #         fit_params={'groups': subject_ids},
                #         error_score='raise',
                #         #n_jobs=-1,
                #     )
                #    nested_scores[j] = nested_result['test_score'].mean()
                #    results[ds][target_name][name][j] = nested_result

                scores[ds][name][target_name] = nested_scores
    assert 0
    # {k: np.mean(v['age']) for k, v in scores['HCP'].items()}


from functools import reduce
from itertools import chain, product

def grid_search(grid_params: Mapping | Sequence[Mapping]):
    """
    Return an iterator over all param configurations,
    because sklearn CV is useless
    """
    if isinstance(grid_params, Mapping):
        grid_params = [grid_params]
    yield from (
        reduce(lambda e, f: {**e, **f}, z, {}) for z in
        chain(*(
            product(*[[{k: e} for e in v] for k, v in p.items()])
            for p in grid_params
        ))
    )


def main(
    extract_ts: bool = False,
    num_parcels: int = 200,
):
    #os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    #os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
    #os.environ['XLA_PYTHON_CLIENT_ALLOCATOR'] = 'platform'
    if extract_ts:
        prepare_timeseries(num_parcels=num_parcels)
    predict(num_parcels=num_parcels)


if __name__ == '__main__':
    main()
