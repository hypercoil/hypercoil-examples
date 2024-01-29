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
from hypercoil.init.atlas import (
    BaseAtlas,
    CortexSubcortexCIfTIAtlas,
    IcosphereAtlas,
)
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


class ConsensusAtlasEncoder(eqx.Module):
    encoding: Mapping[str, Tensor]
    limits: Mapping[str, Tuple[int, int]]

    def __init__(
        self,
        consensus_atlas: BaseAtlas,
        *,
        key: Optional['jax.random.PRNGKey'] = None,
    ):
        self.encoding = {k: v.T for k, v in consensus_atlas.maps.items()}
        self.limits = {
            c: (o.slice_index, o.slice_size)
            for c, o in consensus_atlas.compartments.items()
        }


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


def configure_icosphere_encoder() -> IcosphereEncoder:
    coords_L, mask_L = get_coords_and_mask_fslr(hemi='L')
    coords_R, mask_R = get_coords_and_mask_fslr(hemi='R')
    return IcosphereEncoder(
        subdivisions=SUBDIVISIONS,
        scale=0.5,
        point_coords={'L': coords_L, 'R': coords_R},
        point_mask={'L': mask_L, 'R': mask_R},
        rotation_target=np.array((0., 0., 1.)),
        rotation_secondary=np.array((0., 1., 0.)),
    )


def configure_consensus_atlas_encoder() -> ConsensusAtlasEncoder:
    atlas = CortexSubcortexCIfTIAtlas(
        ref_pointer=(
            '/Users/rastkociric/Downloads/combined_clusters/'
            'abcd_template_matching_combined_clusters_thresh0.605.dlabel.nii'
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
    return ConsensusAtlasEncoder(atlas)


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
    atlas = IcosphereAtlas(
        name='Icosphere',
        n_subdivisions=5,
        rotation_target=np.array((0., 0., 1.)),
        rotation_secondary=np.array((0., 1., 0.)),
        disc_scale=0.5,
    )
    # map_L = atlas.maps['cortex_L']
    # map_R = atlas.maps['cortex_R']
    # visualise_surface_encoder(
    #     encoder_name='Icosphere',
    #     array_L=(map_L.argmax(0) + 1) * (map_L.max(0) > 0),
    #     array_R=(map_R.argmax(0) + 1) * (map_R.max(0) > 0),
    #     is_masked=True,
    # )

    model = AtlasLinear.from_atlas(atlas, encode=True, key=jax.random.PRNGKey(0))

    # data_L = nb.load(
    #     '/Users/rastkociric/Downloads/ds000224-fmriprep/sub-MSC01/ses-func01/'
    #     'func/sub-MSC01_ses-func01_task-rest_hemi-L_space-fsaverage5_bold.func.gii'
    # ).darrays[0].data
    # data_R = nb.load(
    #     '/Users/rastkociric/Downloads/ds000224-fmriprep/sub-MSC01/ses-func01/'
    #     'func/sub-MSC01_ses-func01_task-rest_hemi-R_space-fsaverage5_bold.func.gii'
    # ).darrays[0].data
    # data = np.concatenate((data_L, data_R), axis=-1)
    data_full = nb.load(
        '/Users/rastkociric/Downloads/ds000224-fmriprep/sub-MSC01/ses-func02/func/'
        'sub-MSC01_ses-func02_task-rest_space-fsLR_den-91k_bold.dtseries.nii'
    ).get_fdata(dtype=np.float32).T
    data = data_full[:atlas.mask.shape[0]][atlas.mask.mask_array]
    enc = model(data)

    junk_mask_L = np.isnan(enc['cortex_L'].sum(-1))
    junk_mask_R = np.isnan(enc['cortex_R'].sum(-1))
    plot_f = plotdef(
        surf_from_archive(),
        surf_scalars_from_array(
            'encoder',
            is_masked=True,
        ),
        #vertex_to_face('encoder', interpolation='mode'),
        plot_to_html(),
    )
    plot_f(
        template='fsLR',
        surf_projection='veryinflated',
        encoder_array_left=~junk_mask_L,
        encoder_array_right=~junk_mask_R,
        window_size=(600, 500),
        surf_scalars_cmap='bone',
        fname_spec=f'scalars-zerovariance',
        output_dir='/tmp',
        # theme=pv.themes.DarkTheme(),
        # hemisphere='left',
    )
    # plot_f = plotdef(
    #     surf_from_archive(),
    #     surf_scalars_from_array(
    #         'encoder',
    #         is_masked=False,
    #     ),
    #     #vertex_to_face('encoder', interpolation='mode'),
    #     plot_to_display(),
    # )
    # plot_f(
    #     template='fsLR',
    #     surf_projection='veryinflated',
    #     # encoder_array_left=~junk_mask_L,
    #     # encoder_array_right=~junk_mask_R,
    #     encoder_array=~np.isclose(data_full[:atlas.mask.shape[0]].sum(-1), 0),
    #     window_size=(600, 500),
    #     surf_scalars_cmap='bone',
    #     # theme=pv.themes.DarkTheme(),
    #     # hemisphere='left',
    # )
    assert 0

    # icosencoder = configure_icosphere_encoder()
    # disc_L = (
    #     icosencoder.encoding['L'].argmax(-1) + 1
    # ) * icosencoder.encoding['L'].max(-1)
    # disc_R = (
    #     icosencoder.encoding['R'].argmax(-1) + 1
    # ) * icosencoder.encoding['R'].max(-1)
    # visualise_surface_encoder(
    #     encoder_name='Icosphere',
    #     array_L=disc_L,
    #     array_R=disc_R,
    #     is_masked=False,
    # )
    atlasencoder = configure_consensus_atlas_encoder()
    atlas_L = atlasencoder.encoding['cortex_L'].argmax(-1)
    atlas_R = atlasencoder.encoding['cortex_R'].argmax(-1)
    visualise_surface_encoder(
        encoder_name='Consensus Atlas',
        array_L=atlas_L,
        array_R=atlas_R,
    )


if __name__ == '__main__':
    main()
