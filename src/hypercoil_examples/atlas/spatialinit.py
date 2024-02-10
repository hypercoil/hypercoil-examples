# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Spatial prior initialisation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Initialise a sparse neighbourhood-based spatial prior for the functional
alignment model.
"""
import numpy as np
import templateflow.api as tflow
from hyve import (
    Cell,
    plotdef,
    plot_to_display,
    plot_to_image,
    save_figure,
    surf_from_archive,
    surf_scalars_from_array,
)
from hyve.surf import CortexTriSurface


PATH = (
    '/Users/rastkociric/Downloads/hypercoil-examples/src/hypercoil_examples/'
    'data/surf/tpl-fsLR_hemi-{hemi}_range-{range}_adjacency.npy'
)


def get_range_adjacency(hemi: str, range: int):
    return np.load(PATH.format(hemi=hemi, range=range))


def get_hemi_ranges(hemi: str = 'L'):
    range1 = get_range_adjacency(hemi, 1)
    range2 = get_range_adjacency(hemi, 2)
    range3 = get_range_adjacency(hemi, 3)

    range1padded = np.zeros_like(range2)
    range1padded[:range1.shape[0], :range1.shape[1]] = range1
    range2padded = np.zeros_like(range3)
    range2padded[:range2.shape[0], :range2.shape[1]] = range2

    range1unique = np.sort(range1[..., 1:], axis=-1)
    range2unique = np.sort(
        np.where(
            range1padded == range2,
            -1,
            range2,
        ),
        axis=-1,
    )[..., 7:]
    range3unique = np.sort(
        np.where(
            range2padded == range3,
            -1,
            range3,
        ),
        axis=-1,
    )[..., 19:]
    return (
        range1unique,
        range2unique,
        range3unique,
    )


def init_hemi_spatial_prior(hemi: str = 'L'):
    range1, range2, range3 = get_hemi_ranges(hemi)
    loc = np.concatenate((
        np.arange(range1.shape[0])[:, None],
        range1,
        range2,
        range3,
    ), axis=-1)
    surf = CortexTriSurface.from_tflow(template='fsLR', load_mask=True)
    mask = getattr(
        surf, {'L': 'left', 'R': 'right'}[hemi]
    ).point_data[surf.mask]
    key = np.where(mask[loc], loc, -1)[mask]
    remapped = np.cumsum(mask) - 1
    remapped = remapped[key]
    return np.where(key == -1, -1, remapped)


def init_spatial_priors():
    left = init_hemi_spatial_prior('L')
    right = init_hemi_spatial_prior('R')
    weight = np.asarray(
        (1,) +
        6 * (np.exp(-1),) +
        12 * (np.exp(-2),) +
        18 * (np.exp(-3),)
    )
    return left, right, weight


# def create_fslr_surface():
#    surf = CortexTriSurface.from_tflow(template='fsLR')
#    neighbourhood = []
#    for i in range(surf.left.n_points):
#        print(f'Processing point {i} of {surf.left.n_points}')
#        neighbours = surf.left.point_neighbors_levels(i, 3)
#        neighbourhood.append(tuple(neighbours))


def main():
    left, right, weight = init_spatial_priors()
    W = -np.round(np.log(weight)).astype(int) + 1
    L = np.zeros(left.shape[0])
    R = np.zeros(right.shape[0])
    I = list(range(11)) + list(range(1000, 29001, 1000))
    for i in I:
        L[[list([*left[i]])]] = W
        R[[list([*right[i]])]] = W
        L[-1] = 0
        R[-1] = 0
    layout = Cell() | Cell() << (1 / 2)
    layout = layout * layout
    annotations = {
        0: dict(
            hemisphere='left',
            view='lateral',
        ),
        1: dict(
            hemisphere='left',
            view='medial',
        ),
        2: dict(
            hemisphere='right',
            view='medial',
        ),
        3: dict(
            hemisphere='right',
            view='lateral',
        ),
    }
    layout = layout.annotate(annotations)
    plot_f = plotdef(
        surf_from_archive(),
        surf_scalars_from_array(
            'neighbourhood',
            is_masked=True,
        ),
        #plot_to_display(),
        plot_to_image(),
        save_figure(
            layout_kernel=layout,
            padding=10,
            canvas_size=(1200, 300),
            canvas_color=(0, 0, 0),
            fname_spec=f'scalars-neighbourhoods',
            scalar_bar_action='collect',
        ),
    )
    plot_f(
        template='fsLR',
        surf_projection='veryinflated',
        neighbourhood_array_left=L,
        neighbourhood_array_right=R,
        window_size=(600, 500),
        surf_scalars_cmap='jet',
        #surf_scalars_clim='robust',
        # theme=pv.themes.DarkTheme(),
        hemisphere=['left', 'right', 'both'],
        views={
            'left': ('medial', 'lateral'),
            'right': ('medial', 'lateral'),
            'both': ('dorsal', 'ventral', 'anterior', 'posterior'),
        },
        # theme=pv.themes.DarkTheme(),
        output_dir='/tmp',
    )
    assert 0


if __name__ == '__main__':
    main()
