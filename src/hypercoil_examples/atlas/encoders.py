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
from hypercoil.functional import (
    compartmentalised_linear,
    cmass_coor,
    pairedcorr,
    residualise,
)
from hypercoil.functional.sphere import icosphere
from hypercoil.init.atlas import (
    BaseAtlas,
    CortexSubcortexCIfTIAtlas,
    IcosphereAtlas,
)
from hypercoil.init.vmf import VonMisesFisher
from hypercoil.nn.atlas import AtlasLinear
from hyve import (
    Cell,
    plotdef,
    parcellate_colormap,
    surf_from_archive,
    surf_scalars_from_array,
    vertex_to_face,
    plot_to_display,
    plot_to_image,
    plot_to_html,
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


def create_icosphere_encoder(
    n_subdivisions: int = 5,
    rotation_target: Tensor = np.array((0., 0., 1.)),
    rotation_secondary: Tensor = np.array((0., 1., 0.)),
    disc_scale: float = 0.5,
) -> IcosphereAtlas:
    return IcosphereAtlas(
        name='Icosphere',
        n_subdivisions=n_subdivisions,
        rotation_target=rotation_target,
        rotation_secondary=rotation_secondary,
        disc_scale=disc_scale,
    )


def create_consensus_encoder(
    threshold: float = 0.605,
) -> CortexSubcortexCIfTIAtlas:
    return CortexSubcortexCIfTIAtlas(
        ref_pointer=(
            '/Users/rastkociric/Downloads/combined_clusters/'
            f'abcd_template_matching_combined_clusters_thresh{threshold}'
            '.dlabel.nii'
        ),
        mask_L=None,
        mask_R=None,
        surf_L=tflow.get(
            'fsLR', density='32k', hemi='L', space=None, suffix='sphere'
        ),
        surf_R=tflow.get(
            'fsLR', density='32k', hemi='R', space=None, suffix='sphere'
        ),
    )


def icosphere_encode(
    model: callable,
    data: Tensor,
    atlas: BaseAtlas,
) -> Tuple[Tensor, Tensor]:
    basis = model(data, encode=False)
    vertices_enc = model.enc(basis, ref=data)
    coors =  jnp.concatenate((
        atlas.vertices['cortex_L'], atlas.vertices['cortex_R']
    ))
    return coors, vertices_enc, atlas.coors


def consensus_encode(
    model: callable,
    data: Tensor,
    atlas: BaseAtlas,
) -> Tuple[Tensor, Tensor]:
    basis = model(data, encode=False)[:64]
    coors_L = cmass_coor(
        model.weight['cortex_L'],
        atlas.coors[atlas.compartments.compartments['cortex_L'].mask_array].T,
        radius=100,
    )
    coors_R = cmass_coor(
        model.weight['cortex_R'],
        atlas.coors[atlas.compartments.compartments['cortex_R'].mask_array].T,
        radius=100,
    )
    coors = jnp.concatenate((coors_L.T, coors_R.T))
    parcels_enc = model.enc(basis, ref=data)
    return (
        coors,
        parcels_enc,
        atlas.coors[~atlas.ref.imobj.header.get_axis(1).volume_mask],
    )


def visualise_surface_encoder(
    encoder_name: str,
    array_L: Tensor,
    array_R: Tensor,
    is_masked: bool = True,
):
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
            'encoder',
            is_masked=is_masked,
            allow_multihemisphere=False,
        ),
        parcellate_colormap('encoder', 'network', template='fsLR'),
        #vertex_to_face('encoder', interpolation='mode'),
        #plot_to_display(),
        text_element(
            name='title',
            content=f'{encoder_name} Encoder',
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
            fname_spec=f'scalars-{encoder_name}',
            scalar_bar_action='collect',
        ),
    )
    plot_f(
        template='fsLR',
        surf_projection='veryinflated',
        encoder_array_left=array_L,
        encoder_array_right=array_R,
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


def main():
    encoder = create_icosphere_encoder()
    map_L = encoder.maps['cortex_L']
    map_R = encoder.maps['cortex_R']
    visualise_surface_encoder(
        encoder_name='Icosphere',
        array_L=(map_L.argmax(0) + 1) * (map_L.max(0) > 0),
        array_R=(map_R.argmax(0) + 1) * (map_R.max(0) > 0),
        is_masked=True,
    )

    encoder = create_consensus_encoder()
    map_L = encoder.maps['cortex_L']
    map_R = encoder.maps['cortex_R']
    visualise_surface_encoder(
        encoder_name='Consensus Atlas',
        array_L=(map_L.argmax(0) + 1) * (map_L.max(0) > 0),
        array_R=(map_R.argmax(0) + 1) * (map_R.max(0) > 0),
        is_masked=True,
    )


if __name__ == '__main__':
    main()
