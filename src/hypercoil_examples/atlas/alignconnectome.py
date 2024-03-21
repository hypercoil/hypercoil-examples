# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Group-level alignment analyses: connectomes
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Group-level analyses of aligned parcellations: distance and discriminability
of connectomes.
"""
import pickle
import jax
import matplotlib.pyplot as plt
import nibabel as nb
import numpy as np
from hypercoil_examples.atlas.aligned_dccc import (
    _get_data,
    create_template,
    get_msc_dataset,
)
from hypercoil_examples.atlas.aligngrouplevel import (
    discr, plot_discr, plot_distances, within_between_distance
)

SUBJECTS = ('01', '02', '03', '04', '05', '06', '07', '08', '09', '10')
SESSIONS = ('01', '02', '03', '04', '05', '06', '07', '08', '09', '10')
ATLAS_FNAME_SPEC = (
    '/Users/rastkociric/Downloads/MSC_individual_atlases/'
    'atlas_MSC_sub-MSC{subject}_ses-{session}_{mode}_cortex_{hemi}.gii'
)
MODES = ('individual', 'mle', 'energy')
NUM_NODES = 360


def template_atlas():
    print('Creating template atlas...')
    (
        _,
        encoder,
        template_left,
        template_right,
        spatial_loc_left,
        spatial_loc_right,
        spatial_data,
        temporal_L,
        temporal_R,
        spatial_L,
        spatial_R,
        log_prob_L,
        log_prob_R,
    ) = create_template()
    atlas_L_label = np.argmax(log_prob_L, axis=-1)
    atlas_R_label = np.argmax(log_prob_R, axis=-1) + log_prob_L.shape[-1]
    atlas = np.eye(log_prob_L.shape[-1] + log_prob_R.shape[-1])[
        np.concatenate((atlas_L_label, atlas_R_label))
    ]
    atlas = atlas / atlas.sum(0, keepdims=True)
    return atlas


def compute_connectomes(
    template: np.ndarray,
):
    modes = ('template',) + MODES
    connectomes = np.zeros((
        len(modes),
        len(SUBJECTS),
        len(SESSIONS),
        NUM_NODES * (NUM_NODES - 1) // 2
    ))
    data_index = []
    for subject in SUBJECTS:
        for session in SESSIONS:
            bold = _get_data(get_msc_dataset(subject, session))
            data_index += [(subject, session)]
            for mode in modes:
                print(f'Processing {subject} {session} {mode}...')
                if mode != 'template':
                    atlas_L_path = ATLAS_FNAME_SPEC.format(
                        subject=subject, session=session, mode=mode, hemi='L')
                    atlas_R_path = ATLAS_FNAME_SPEC.format(
                        subject=subject, session=session, mode=mode, hemi='R')
                    atlas_L = nb.load(atlas_L_path).darrays[0].data
                    atlas_R = nb.load(atlas_R_path).darrays[0].data
                    atlas_L_label = np.argmax(atlas_L, axis=-1)
                    atlas_R_label = np.argmax(atlas_R, axis=-1) + atlas_L.shape[-1]
                    atlas = np.eye(atlas_L.shape[-1] + atlas_R.shape[-1])[
                        np.concatenate((atlas_L_label, atlas_R_label))
                    ]
                    atlas = atlas / atlas.sum(0, keepdims=True)
                else:
                    atlas = template
                connectome = np.corrcoef(atlas.T @ bold)
                cvec = connectome[np.triu_indices_from(connectome, 1)]
                connectomes[modes.index(mode), int(subject) - 1, int(session) - 1] = cvec
    return connectomes, data_index


def compute_distances(connectomes: np.ndarray):
    modes = ('template',) + MODES
    n_modes = len(modes)
    assert connectomes.shape[0] == n_modes
    connectomes = (
        connectomes / np.linalg.norm(connectomes, axis=-1, keepdims=True)
    )
    distance = {}
    for i, mode in enumerate(modes):
        cmode = connectomes[i].reshape(-1, connectomes.shape[-1])
        distance[mode] = np.arccos(np.clip(cmode @ cmode.T, -1, 1))
    return distance


# def compute_and_plot_distances(connectomes, data_index):
#     modes = ('template',) + MODES
#     n_modes = len(modes)
#     assert connectomes.shape[0] == n_modes
#     index = np.asarray([sub for sub, _ in data_index])
#     fig, ax = plt.subplots(1, n_modes, figsize=(6 * n_modes, 5))
#     fig.set_tight_layout(True)
#     connectomes = (
#         connectomes / np.linalg.norm(connectomes, axis=-1, keepdims=True)
#     )
#     discriminability = {}
#     for i, mode in enumerate(modes):
#         cmode = connectomes[i].reshape(-1, connectomes.shape[-1])
#         dmat = np.arccos(np.clip(cmode @ cmode.T, -1, 1))
#         vmin, vmax = np.quantile(dmat, [0.05, 0.95])
#         im = ax[i].imshow(dmat, cmap='inferno', vmin=vmin, vmax=vmax)
#         fig.colorbar(im, ax=ax[i])
#         ax[i].set_title(mode)
#         ax[i].set_xticks([])
#         ax[i].set_yticks([])
#         discriminability[mode] = discr_impl(dmat, index)
#     fig.savefig('/tmp/distances_connectome.png')
#     return discriminability


def main():
    modes = ('template',) + MODES
    template = template_atlas()
    connectomes, data_index = compute_connectomes(template=template)
    distance = compute_distances(connectomes)
    plot_distances(distance, modes=modes)
    discriminability = discr(distance, data_index, modes=modes)
    plot_discr(distance, discriminability, modes=modes)
    with open(f'/tmp/distances_connectome.pkl', 'wb') as handle:
        pickle.dump(distance, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print(discriminability)
    print({mode: np.mean(d) for mode, d in discriminability.items()})
    print(within_between_distance(distance, modes=modes))
    assert 0
    np.save('connectomes.npy', connectomes)


if __name__ == '__main__':
    main()
