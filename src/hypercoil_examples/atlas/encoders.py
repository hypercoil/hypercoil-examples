# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Atlas encoders
~~~~~~~~~~~~~~
Initialise and visualise atlas encoders
"""
from typing import Literal, Mapping, Optional, Sequence, Tuple, Union
import nibabel as nb
import templateflow.api as tflow
import numpy as np
import pyvista as pv

import jax
import jax.numpy as jnp
import equinox as eqx
from hypercoil.engine import Tensor
from hypercoil.functional import pairedcorr, compartmentalised_linear
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
IMAGE_DIM = 25
LOCUS_COUNT = (IMAGE_DIM // 5 or 1) ** 2
LOCUS_RADIUS = 3


class LocusEncoder(eqx.Module):
    encoding: Mapping[str, Tensor]
    limits: Mapping[str, Tuple[int, int]]

    def __call__(
        self,
        X: Tensor,
        *,
        key: Optional['jax.random.PRNGKey'] = None,
    ) -> Tensor:
        references = compartmentalised_linear(
            input=X,
            weight=self.encoding,
            limits=self.limits,
        )
        return pairedcorr(references, X)


class LocusDiscEncoder(LocusEncoder):
    encoding: Mapping[str, Tensor]
    limits: Mapping[str, Tuple[int, int]]

    def __init__(
        self,
        locus_coords: Tensor = None,
        point_coords: Tensor = None,
        locus_radius: int = LOCUS_RADIUS,
        *,
        key: 'jax.random.PRNGKey',
    ):
        distance = (
            (point_coords ** 2).sum(axis=-1, keepdims=True)
            + (locus_coords ** 2).sum(axis=-1, keepdims=True).T
            - 2 * point_coords @ locus_coords.T
        )
        self.encoding = {'all': jnp.where(distance < locus_radius ** 2, 1., 0.)}
        self.limits = {'all': (0, self.encoding['all'].shape[0])}


class IcosphereEncoder(eqx.Module):
    encoding: Mapping[str, Tensor]
    limits: Mapping[str, Tuple[int, int]]

    def __init__(
        self,
        point_coords: Union[Tensor, Mapping[str, Tensor]],
        subdivisions: int = SUBDIVISIONS,
        scale: float = 1.,
        point_mask: Optional[Union[Tensor, Mapping[str, Tensor]]] = None,
        rotation_target: Optional[Tensor] = None,
        rotation_secondary: Optional[Tensor] = None,
        *,
        key: Optional['jax.random.PRNGKey'] = None,
    ):
        if not isinstance(point_coords, Mapping):
            point_coords = {'all': point_coords}
        locus_coords = icosphere(
            subdivisions=subdivisions,
            target=rotation_target,
            secondary=rotation_secondary,
        )
        point_coords = {
            k: (v / np.linalg.norm(v, axis=-1, keepdims=True))
            for k, v in point_coords.items()
        }
        distance = {
            k: np.arctan2(
                np.linalg.norm(
                    np.cross(
                        v[..., None, :],
                        locus_coords[None, ...],
                    ),
                    axis=-1,
                ),
                v @ locus_coords.T,
            )
            for k, v in point_coords.items()
        }
        if point_mask is not None:
            if not isinstance(point_mask, Mapping):
                point_mask = {'all': point_mask}
            valid = {k: v[distance[k].argmin(0)] for k, v in point_mask.items()}
            distance = {k: v[..., valid[k]] for k, v in distance.items()}
        self.encoding = {
            k: jnp.where(
                v < (scale * np.arctan(2) / 2 / (subdivisions + 1)),
                1.,
                0.,
            )
            for k, v in distance.items()
        }
        start = 0
        limits = {}
        for k, v in self.encoding.items():
            limits[k] = (start, start + v.shape[0])
            start += v.shape[0]
        self.limits = limits


def get_coords_and_mask_fslr(hemi: Literal['L', 'R'] = 'L') -> Tensor:
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
    return ref, mask


def main():
    coords_L, mask_L = get_coords_and_mask_fslr(hemi='L')
    coords_R, mask_R = get_coords_and_mask_fslr(hemi='R')
    encoder = IcosphereEncoder(
        subdivisions=SUBDIVISIONS,
        scale=0.5,
        point_coords={'L': coords_L, 'R': coords_R},
        point_mask={'L': mask_L, 'R': mask_R},
        rotation_target=np.array((0., 0., 1.)),
        rotation_secondary=np.array((0., 1., 0.)),
    )

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
    disc_L = (encoder.encoding['L'].argmax(-1) + 1) * encoder.encoding['L'].max(-1)
    disc_R = (encoder.encoding['R'].argmax(-1) + 1) * encoder.encoding['R'].max(-1)
    plot_f(
        template='fsLR',
        surf_projection='veryinflated',
        icosphere_array_left=disc_L,
        icosphere_array_right=disc_R,
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
