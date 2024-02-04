# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Two subjects
~~~~~~~~~~~~
Projection of parcellation fit to one subject onto a second subject's
functional space.
"""
from typing import Optional, Sequence, Tuple

import jax
import jax.numpy as jnp
import equinox as eqx
import matplotlib.pyplot as plt

from hypercoil.init import (
    CortexSubcortexGIfTIAtlas,
    VonMisesFisher,
)
from hypercoil.engine import Tensor
from hypercoil.nn.atlas import AtlasLinear
from hyve import (
    Cell,
    plotdef,
    plot_to_display,
    plot_to_image,
    save_figure,
    surf_from_archive,
    surf_scalars_from_array,
    text_element,
)

from hypercoil_examples.atlas.encoders import (
    visualise_surface_encoder,
    create_icosphere_encoder,
)
from hypercoil_examples.atlas.vmf import get_data, whiten_data


ATLAS_PATH = {
    'L': (
        '/Users/rastkociric/Downloads/sparque/src/sparque/'
        'DCBC/parcellations/Glasser_2016.32k.L.label.gii'
    ),
    'R': (
        '/Users/rastkociric/Downloads/sparque/src/sparque/'
        'DCBC/parcellations/Glasser_2016.32k.R.label.gii'
    ),
}


def _sign_times_magnitude(X: Tensor):
    return jnp.sign(X) * jnp.abs(X)

class SmoothThreshold(eqx.Module):
    inflection: Tensor
    scale: Tensor
    ingest: callable = jnp.abs
    argfn: callable = _sign_times_magnitude
    
    def __call__(self, X: Tensor):
        return self.argfn(X) * jax.nn.sigmoid(
            self.scale * (self.ingest(X) - self.inflection)
        )


def visualise(
    log_prob_L: jnp.ndarray,
    log_prob_R: jnp.ndarray,
    name: str,
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
            'surf_scalars',
            is_masked=True,
            allow_multihemisphere=False,
        ),
        #parcellate_colormap('encoder', 'network', template='fsLR'),
        #vertex_to_face('encoder', interpolation='mode'),
        #plot_to_display(),
        text_element(
            name='title',
            content=f'Glasser 360 {name}',
            bounding_box_height=128,
            font_size_multiplier=0.1,
            font_color='#cccccc',
            priority=-1,
        ),
        plot_to_image(),
        save_figure(
            layout_kernel=layout,
            padding=10,
            canvas_size=(900, 750),
            canvas_color=(0, 0, 0),
            fname_spec=f'scalars-{name}Glasser360',
            scalar_bar_action='collect',
        ),
    )
    plot_f(
        template='fsLR',
        surf_projection='veryinflated',
        surf_scalars_array_left=log_prob_L.argmax(-1),
        surf_scalars_array_right=log_prob_R.argmax(-1),
        surf_scalars_cmap='prism',
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
    atlas = CortexSubcortexGIfTIAtlas(
        data_L=ATLAS_PATH['L'],
        data_R=ATLAS_PATH['R'],
        name='Glasser360',
    )
    visualise(
        log_prob_L=atlas.maps['cortex_L'].T,
        log_prob_R=atlas.maps['cortex_R'].T,
        name='original',
    )
    encoder = create_icosphere_encoder()
    encoder = AtlasLinear.from_atlas(
        encoder,
        encode=True,
        key=jax.random.PRNGKey(0),
    )
    # visualise_surface_encoder(
    #     encoder_name='Glasser 360',
    #     array_L=(map_L.argmax(0) + 1) * (map_L.max(0) > 0),
    #     array_R=(map_R.argmax(0) + 1) * (map_R.max(0) > 0),
    #     is_masked=True,
    # )
    model = AtlasLinear.from_atlas(
        atlas,
        encode=True,
        forward_mode='project',
        key=jax.random.PRNGKey(0),
    )

    msc = get_data('MSC')
    enc = encoder(msc, encode=True, decode_labels=False)
    parcels = model(msc, encode=False, decode_labels=False)

    temporal_mu = encoder.enc(parcels, ref=msc, decode_labels=False)
    spatial_mu = model(
        atlas.coors, forward_mode='map', encode=False, decode_labels=False
    )

    threshold_locus = int(enc['cortex_L'].shape[-2] * 0.9)
    inflection = jnp.partition(
        enc['cortex_L'], threshold_locus, axis=-2
    )[None, threshold_locus, :]
    scale = jnp.array(3e8)
    enc['cortex_L'] = SmoothThreshold(
        inflection=inflection,
        scale=scale,
        ingest=lambda x: x,
        argfn=lambda x: 1,
    )(enc['cortex_L'])

    threshold_locus = int(enc['cortex_R'].shape[-2] * 0.9)
    inflection = jnp.partition(
        enc['cortex_R'], threshold_locus, axis=-2
    )[None, threshold_locus, :]
    enc['cortex_R'] = SmoothThreshold(
        inflection=inflection,
        scale=scale,
        ingest=lambda x: x,
        argfn=lambda x: 1,
    )(enc['cortex_R'])

    parcels_enc = model(
        jnp.concatenate((enc['cortex_L'], enc['cortex_R'])),
        encode=False,
        decode_labels=False,
        concatenate=False,
    )
    temporal_mu_L = parcels_enc['cortex_L']
    temporal_mu_R = parcels_enc['cortex_R']

    # enc['cortex_L'], temporal_mu_L = whiten_data(enc['cortex_L'], temporal_mu[:180])
    # enc['cortex_R'], temporal_mu_R = whiten_data(enc['cortex_R'], temporal_mu[180:])
    temporal_L = VonMisesFisher(mu=temporal_mu_L, kappa=10)
    temporal_R = VonMisesFisher(mu=temporal_mu_R, kappa=10)
    spatial_L = VonMisesFisher(mu=spatial_mu[:180], kappa=10)
    spatial_R = VonMisesFisher(mu=spatial_mu[180:], kappa=10)
    log_prob_L = (
        temporal_L.log_prob(enc['cortex_L']) +
        spatial_L.log_prob(atlas.coors[:enc['cortex_L'].shape[0]])
    )
    log_prob_R = (
        temporal_R.log_prob(enc['cortex_R']) +
        spatial_R.log_prob(atlas.coors[enc['cortex_L'].shape[0]:])
    )

    visualise(
        log_prob_L=log_prob_L,
        log_prob_R=log_prob_R,
        name='backprojection',
    )
    print(
        'Total parcels detected (left hemisphere):',
        len(jnp.unique(log_prob_L.argmax(-1)))
    )
    print(
        'Total parcels detected (right hemisphere):',
        len(jnp.unique(log_prob_R.argmax(-1)))
    )

    hcp = get_data('HCP')
    enc = encoder(hcp, encode=True, decode_labels=False)
    #parcels = model(hcp, encode=False, decode_labels=False)



    threshold_locus = int(enc['cortex_L'].shape[-2] * 0.9)
    inflection = jnp.partition(
        enc['cortex_L'], threshold_locus, axis=-2
    )[None, threshold_locus, :]
    scale = jnp.array(3e8)
    enc['cortex_L'] = SmoothThreshold(
        inflection=inflection,
        scale=scale,
        ingest=lambda x: x,
        argfn=lambda x: 1,
    )(enc['cortex_L'])

    threshold_locus = int(enc['cortex_R'].shape[-2] * 0.9)
    inflection = jnp.partition(
        enc['cortex_R'], threshold_locus, axis=-2
    )[None, threshold_locus, :]
    enc['cortex_R'] = SmoothThreshold(
        inflection=inflection,
        scale=scale,
        ingest=lambda x: x,
        argfn=lambda x: 1,
    )(enc['cortex_R'])

    parcels_enc = model(
        jnp.concatenate((enc['cortex_L'], enc['cortex_R'])),
        encode=False,
        decode_labels=False,
        concatenate=False,
    )
    temporal_mu_L = parcels_enc['cortex_L']
    temporal_mu_R = parcels_enc['cortex_R']


    # enc['cortex_L'], _ = whiten_data(enc['cortex_L'])
    # enc['cortex_R'], _ = whiten_data(enc['cortex_R'])
    log_prob_L = (
        temporal_L.log_prob(enc['cortex_L']) +
        spatial_L.log_prob(atlas.coors[:enc['cortex_L'].shape[0]])
    )
    log_prob_R = (
        temporal_R.log_prob(enc['cortex_R']) +
        spatial_R.log_prob(atlas.coors[enc['cortex_L'].shape[0]:])
    )

    visualise(
        log_prob_L=log_prob_L,
        log_prob_R=log_prob_R,
        name='crossprojection',
    )
    print(
        'Total parcels detected (left hemisphere):',
        len(jnp.unique(log_prob_L.argmax(-1)))
    )
    print(
        'Total parcels detected (right hemisphere):',
        len(jnp.unique(log_prob_R.argmax(-1)))
    )
    assert 0


if __name__ == "__main__":
    main()
