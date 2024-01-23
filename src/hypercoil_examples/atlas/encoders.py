# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Atlas encoders
~~~~~~~~~~~~~~
Initialise and visualise atlas encoders
"""
from typing import Literal
import nibabel as nb
import templateflow.api as tflow
import numpy as np
import pyvista as pv
from hypercoil.engine import Tensor
from hypercoil.functional.sphere import icosphere
from hyve import (
    Cell,
    plotdef,
    parcellate_colormap,
    surf_from_archive,
    surf_scalars_from_array,
    vertex_to_face,
    plot_to_display,
    plot_to_image,
    save_figure,
    text_element,
)


SUBDIVISIONS = 5


def get_distmat(coor: Tensor, hemi: Literal['L', 'R'] = 'L') -> Tensor:
    mask = nb.load(
        tflow.get(
            'fsLR', density='32k', hemi=hemi, space=None, desc='nomedialwall'
        )
    )
    mask = mask.darrays[0].data.astype(bool)
    ref = nb.load(
        tflow.get(
            'fsLR', density='32k', hemi=hemi, space=None, suffix='sphere'
        )
    )
    ref = ref.darrays[0].data
    ref = ref / np.linalg.norm(ref, axis=-1, keepdims=True)
    return np.arctan2(
        np.linalg.norm(np.cross(coor[..., None, :], ref[None, ...]), axis=-1),
        coor @ ref.T,
    ), mask


def get_discs(
    coor: Tensor, hemi: Literal['L', 'R'] = 'L', scale: float=1.
) -> Tensor:
    distmat, mask = get_distmat(coor=coor, hemi=hemi)
    valid = mask[distmat.argmin(-1)]
    # coor_masked = coor[valid, ...]
    distmat_masked = distmat[valid, ...]
    return (distmat_masked < (scale * np.arctan(2) / 2 / (SUBDIVISIONS + 1)))


def main():
    r_icosphere = icosphere(
        subdivisions=SUBDIVISIONS,
        target=np.array((0., 0., 1.)),
        secondary=np.array((0., 1., 0.)),
    )

    disc_L = get_discs(coor=r_icosphere, hemi='L', scale=0.5)
    disc_R = get_discs(coor=r_icosphere, hemi='R', scale=0.5)

    # print((disc_L.argmax(0) * disc_L.max(0)), (disc_R.argmax(0) * disc_R.max(0)))
    # print(np.histogram((disc_L.argmax(0) + 1) * disc_L.max(0)))
    # plot_f = plotdef(
    #     surf_from_archive(),
    #     surf_scalars_from_array('icosphere', is_masked=False, allow_multihemisphere=False),
    #     parcellate_colormap('icosphere', 'network', template='fsLR'),
    #     #vertex_to_face('icosphere', interpolation='mode'),
    #     plot_to_display(),
    # )
    # plot_f(
    #     template='fsLR',
    #     surf_projection='veryinflated',
    #     icosphere_array_left=((disc_L.argmax(0) + 1) * disc_L.max(0)),
    #     icosphere_array_right=((disc_R.argmax(0) + 1) * disc_R.max(0)),
    #     window_size=(600, 500),
    #     # empty_builders=True,
    #     # hemisphere='left',
    # )

    layout = Cell() / Cell() / Cell() << (1 / 3)
    layout = layout | layout | layout << (1 / 3)
    annotations = {
        0: dict(
            hemisphere='left',
            view='lateral',
        ),
        1: dict(view='dorsal'),
        2: dict(
            hemisphere='left',
            view='medial',
        ),
        3: dict(view='anterior'),
        4: dict(elements=['title']),
        5: dict(view='posterior'),
        6: dict(
            hemisphere='right',
            view='lateral',
        ),
        7: dict(view='ventral'),
        8: dict(
            hemisphere='right',
            view='medial',
        ),
    }
    layout = layout.annotate(annotations)

    plot_f = plotdef(
        surf_from_archive(),
        surf_scalars_from_array(
            'icosphere',
            is_masked=False,
            allow_multihemisphere=False,
        ),
        parcellate_colormap('icosphere', 'network', template='fsLR'),
        #vertex_to_face('icosphere', interpolation='mode'),
        #plot_to_display(),
        text_element(
            name='title',
            content='Icosphere Encoder',
            bounding_box_height=128,
            font_size_multiplier=0.2,
            font_color='#cccccc',
            priority=-1,
        ),
        plot_to_image(),
        save_figure(
            layout_kernel=layout,
            padding=10,
            canvas_size=(900, 750),
            canvas_color=(0, 0, 0),
            fname_spec='scalars-{surfscalars}',
            scalar_bar_action='collect',
        ),
    )
    plot_f(
        template='fsLR',
        surf_projection='veryinflated',
        icosphere_array_left=((disc_L.argmax(0) + 1) * disc_L.max(0)),
        icosphere_array_right=((disc_R.argmax(0) + 1) * disc_R.max(0)),
        window_size=(600, 500),
        hemisphere=['left', 'right', 'both'],
        views={
            'left': ('medial', 'lateral'),
            'right': ('medial', 'lateral'),
            'both': ('dorsal', 'ventral', 'anterior', 'posterior'),
        },
        # theme=pv.themes.DarkTheme(),
        output_dir='/tmp',
        load_mask=True,
        # hemisphere='left',
    )


if __name__ == '__main__':
    main()
