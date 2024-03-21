# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Group-level alignment analyses
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Group-level analyses of aligned parcellations: DCBC and discriminability.
"""
import glob
import pickle

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import nibabel as nb
import numpy as np

from hypercoil.loss.functional import js_divergence

ATLAS_FNAME_SPEC = (
    '/tmp/atlas_MSC_sub-MSC{subject}_ses-{session}_{mode}_cortex_{hemi}'
)
MODES = ('individual', 'mle', 'energy')


def load_dccc_data():
    dccc_paths = sorted(
        glob.glob('/Users/rastkociric/Downloads/msc_dccc_??.pkl')
    )
    dccc_data = {}
    for path in dccc_paths:
        sub_id = int(path.split('_')[-1][:2])
        with open(path, 'rb') as f:
            dccc_data[sub_id] = pickle.load(f)
    return dccc_data


def plot_dccc(dccc_data: dict):
    def mean_var_dccc_level(level: str):
        between = jnp.stack([
            dccc_data[sub][ses][level]['corr_between']
            for sub in subjects for ses in sessions
        ])
        within = jnp.stack([
            dccc_data[sub][ses][level]['corr_within']
            for sub in subjects for ses in sessions
        ])
        diff = within - between
        dccc = jnp.asarray([
            dccc_data[sub][ses][level]['dccc'].sum()
            for sub in subjects for ses in sessions
        ])
        return diff.mean(0), diff.std(0) / jnp.sqrt(diff.shape[0]), dccc

    subjects = tuple(dccc_data.keys())
    sessions = tuple(dccc_data[subjects[0]].keys())
    mean_template, sem_template, dccc_template = mean_var_dccc_level('template')
    mean_individual, sem_individual, dccc_individual = mean_var_dccc_level('individual')
    mean_mle, sem_mle, dccc_mle = mean_var_dccc_level('mle')
    mean_energy, sem_energy, dccc_energy = mean_var_dccc_level('energy')

    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    ax[0].errorbar(
        dccc_data[subjects[0]][sessions[0]]['template']['integral_samples'],
        mean_template,
        yerr=sem_template,
        label='Template',
        fmt='o',
    )
    ax[0].errorbar(
        dccc_data[subjects[0]][sessions[0]]['individual']['integral_samples'],
        mean_individual,
        yerr=sem_individual,
        label='Aligned',
        fmt='o',
    )
    ax[0].errorbar(
        dccc_data[subjects[0]][sessions[0]]['mle']['integral_samples'],
        mean_mle,
        yerr=sem_mle,
        label='MLE',
        fmt='o',
    )
    ax[0].errorbar(
        dccc_data[subjects[0]][sessions[0]]['energy']['integral_samples'],
        mean_energy,
        yerr=sem_energy,
        label='Energy',
        fmt='o',
    )
    ax[0].legend()
    ax[0].set_title('Within - Between')
    ax[0].set_xlabel('Distance')
    ax[0].set_ylabel('Within - Between')
    dccc = jnp.stack([dccc_template, dccc_individual, dccc_mle, dccc_energy])
    ax[1].bar(
        ('Template', 'Aligned', 'MLE', 'Energy'),
        dccc.mean(1),
        yerr=dccc.std(1) / jnp.sqrt(dccc.shape[1]),
    )
    ax[1].set_title('Mean DCCC')
    ax[1].set_ylabel('DCCC')
    dccc = jnp.take_along_axis(
        dccc, jnp.argsort(dccc.mean(0))[None, ...], axis=1
    )
    ax[2].plot(dccc.T, alpha=0.5)
    ax[2].set_title('Instances')
    ax[2].set_ylabel('DCCC')
    ax[2].set_xlabel('Data instances')
    ax[2].legend(('Template', 'Aligned', 'MLE', 'Energy'))
    fig.savefig('/tmp/dccc_group.png')


def construct_distance_matrix() -> tuple:
    paths = glob.glob(ATLAS_FNAME_SPEC.format(
        subject='*', session='*', mode='*', hemi='*'
    ))
    paths = sorted(paths)
    paths_left = [p for p in paths if 'cortex_L' in p]
    paths_right = [p for p in paths if 'cortex_R' in p]

    data_left = {}
    data_index = set()
    for path in paths_left:
        print(f'Loading {path}')
        subject = path.split('_')[-5][7:]
        session = path.split('_')[-4][4:]
        mode = path.split('_')[-3]
        #hemi = path.split('_')[-1]
        data_left[mode] = data_left.get(mode, {})
        data_left[mode][subject] = data_left[mode].get(subject, {})
        data_left[mode][subject][session] = nb.load(path).darrays[0].data
        data_index = data_index.union(set([(subject, session)]))
    data_index = sorted(list(data_index))
    distance = {}
    for mode in MODES:
        print(f'Computing distance for {mode}')
        distance[mode] = jnp.zeros((len(data_index), len(data_index)))
        for i, (subject_i, session_i) in enumerate(data_index):
            log_prob_i = data_left[mode][subject_i][session_i]
            #data_i = jax.nn.softmax(log_prob_i, axis=0)
            data_i = jnp.eye(log_prob_i.shape[-1])[log_prob_i.argmax(-1)]
            for j, (subject_j, session_j) in enumerate(data_index):
                if i == j:
                    continue
                log_prob_j = data_left[mode][subject_j][session_j]
                #data_j = jax.nn.softmax(data_left[mode][subject_j][session_j], axis=0)
                data_j = jnp.eye(log_prob_j.shape[-1])[log_prob_j.argmax(-1)]
                distance[mode] = distance[mode].at[i, j].set(
                    jnp.sqrt(jnp.nanmean((js_divergence(data_i, data_j))))
                )
    return distance, data_index


def discr_impl(
    distance: jnp.ndarray,
    ident: np.ndarray,
) -> jnp.ndarray:
    key = ident[distance.argsort(-1)]
    replicate = (key[..., 0][..., None] == key[..., 1:])
    cumrepl = replicate.cumsum(-1)
    violations = (
        (cumrepl < cumrepl.max(-1, keepdims=True)) & ~replicate
    ).sum(-1)
    return 1 - violations / (
        #violations.size * (violations.size - 1) - replicate.sum(-1)
        (violations.size - 1) - replicate.sum(-1)
    )


def discr(
    distance: jnp.ndarray,
    data_index: list,
    modes: tuple = MODES,
) -> dict:
    index = np.asarray([sub for sub, _ in data_index])
    discriminability = {}
    for mode in modes:
        discriminability[mode] = discr_impl(distance[mode], index)
    return discriminability


def plot_discr(
    distance: jnp.ndarray,
    discriminability: dict,
    modes: tuple = MODES,
):
    for mode in modes:
        fig, ax = plt.subplots(1, 2, figsize=(12, 5))
        i = ax[0].imshow(distance[mode], cmap='inferno')
        fig.colorbar(i, ax=ax[0])
        ax[0].set_title('Distance')
        ax[1].bar(
            np.arange(len(discriminability[mode])),
            discriminability[mode],
        )
        ax[1].set_title('Discriminability')
        fig.savefig(f'/tmp/disciminability_{mode}.png')


def plot_distances(distance: dict, modes: tuple = MODES):
    n_modes = len(modes)
    fig, ax = plt.subplots(1, n_modes, figsize=(6 * n_modes, 5))
    fig.set_tight_layout(True)
    for i, mode in enumerate(modes):
        dmat = distance[mode]
        vmin, vmax = np.quantile(dmat, [0.05, 0.95])
        im = ax[i].imshow(dmat, cmap='inferno', vmin=vmin, vmax=vmax)
        fig.colorbar(im, ax=ax[i])
        ax[i].set_title(mode)
        ax[i].set_xticks([])
        ax[i].set_yticks([])
    fig.savefig('/tmp/distances.png')


def within_between_distance(
    distance: dict,
    modes: tuple = MODES,
):
    index = (
        (10 * np.floor(np.arange(100) / 10))[:, None] ==
        (10 * np.floor(np.arange(100) / 10))
    )
    wbdist = {}
    wbndist = {}
    for mode in modes:
        bdist = distance[mode][~index].mean()
        wdist = distance[mode][index].mean()
        wbdist[mode] = bdist - wdist
        wbndist[mode] = (bdist - wdist) / distance[mode].mean()
    return wbdist, wbndist


def main():
    dccc_data = load_dccc_data()
    plot_dccc(dccc_data)
    distance, data_index = construct_distance_matrix()
    discriminability = discr(distance, data_index)
    plot_discr(distance, discriminability)
    with open(f'/tmp/distances.pkl', 'wb') as handle:
        pickle.dump(distance, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print(within_between_distance(distance))
    assert 0


if __name__ == '__main__':
    main()
