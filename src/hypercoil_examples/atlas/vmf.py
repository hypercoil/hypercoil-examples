# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
vMFs
~~~~
Visualise atlas vMF log probs using different encoders (e.g., icosphere,
consensus).
"""
from typing import Any, Optional, Sequence, Tuple

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import nibabel as nb
import numpy as np

from hypercoil.engine import Tensor, _to_jax_array
from hypercoil.functional import residualise
from hypercoil.init import (
    VonMisesFisher,
)
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

from hypercoil_examples.atlas.const import MSC_DATA_ROOT
from hypercoil_examples.atlas.encoders import (
    create_icosphere_encoder,
    create_consensus_encoder,
    create_7net_encoder,
    icosphere_encode,
    consensus_encode,
)

INDICES = {
    'icosphere': (0, 1, 2, 3, 4), # (0, 22, 37, 131, 207), #
    'consensus': (0, 1, 2, 3, 4),
    '7net': (0, 1, 2, 3, 4),
}

CIFTI = {
    'HCP': '/Users/rastkociric/Downloads/rfMRI_REST1_LR_Atlas_MSMAll.dtseries.nii',
    'MSC': (
        f'{MSC_DATA_ROOT}/'
        'sub-MSC01_ses-func02_task-rest_space-fsLR_den-91k_bold.dtseries.nii'
    ),
}

ENCODER_FACTORY = {
    'icosphere': create_icosphere_encoder,
    'consensus': create_consensus_encoder,
    '7net': create_7net_encoder,
}

ENCODE_SELF = {
    'icosphere': icosphere_encode,
    'consensus': consensus_encode,
    '7net': consensus_encode,
}

N_COMPONENTS = 64


from typing import Union, Optional
def generalised_whitening(
    data: Tensor,
    sphering: Union[float, Tensor] = 1.,
    register: Optional[Tensor] = None,
) -> Tensor:
    """
    Generalised whitening of data.

    Parameters
    ----------
    data : Tensor
        Data to whiten.
    sphering : Union[float, Tensor], optional
        Sphering of the data, by default 1. A value of 1 corresponds to
        true whitening, while a value of 0 corresponds to no whitening.
    register : Optional[Tensor], optional
        Register to use for whitening, by default None. If not provided, the
        register is set to the eigenvectors of the covariance matrix of the
        data (equivalent to ZCA whitening). Specify an identity matrix to
        perform PCA whitening.

    Returns
    -------
    Tensor
        Whitened data.
    """
    L, Q = jnp.linalg.eigh(jnp.cov(data.T))
    if register is None:
        register = Q
    coef = -jnp.flip(sphering) / 2
    LW = jnp.maximum(L, jnp.finfo(L.dtype).eps) ** coef
    # First, we must ensure that the scalar applied to each smaller eigenvalue
    # will be at least as large as the scalars applied to all the larger
    # eigenvalues. Otherwise, we would be increasing the eccentricity of the
    # distribution, which is not allowed.
    llim = jnp.flip(jax.lax.cummax(jnp.flip(LW), axis=0))
    LW = jnp.where(
        jnp.isclose(llim, LW), # Account for loss of precision
        LW,
        jnp.where(LW < llim, llim, LW),
    )
    # Now we need to make sure that the reconditioning matrix wouldn't switch
    # the order of any eigenvalues. We do this by replacing each eigenvalue
    # with the maximum of itself and all the eigenvalues that come after it.
    #TODO: switch to this when jax implements ufunc for cumulative max
    # cmax = jnp.maximum.accumulate(L) # Does not work in JAX!
    # cmax = (jnp.tril(jnp.ones((L.size, L.size))) * L).max(-1) # Slow
    # cmax = jax.lax.cummax(L, axis=0) # Using this for now
    llim = jnp.sqrt(jax.lax.cummax(LW ** 2 * L, axis=0) / L)
    # Note that this approach will also zero the derivatives wrt any
    # eigenvalues that would be switched. It's important that we only impose
    # the restriction where we are confident that there would be a switch --
    # i.e., we must be very mindful of false positives resulting from loss of
    # precision.
    LW = jnp.where(
        jnp.isclose(llim, LW), # Account for loss of precision
        LW,
        jnp.where(LW < llim, llim, LW),
    )
    return register @ jnp.diag(LW) @ Q.T


def visualise_surface_encoder(
    index: int,
    array: Tensor,
    is_masked: bool = True,
):
    layout = Cell() | Cell() << (1 / 2)
    layout = layout * layout * layout
    layout = Cell() / layout << (1 / 9)
    annotations = {
        0: dict(elements=['title']),
        1: dict(
            hemisphere='left',
            view='lateral',
        ),
        2: dict(
            hemisphere='left',
            view='medial',
        ),
        3: dict(
            hemisphere='right',
            view='medial',
        ),
        4: dict(
            hemisphere='right',
            view='lateral',
        ),
        5: dict(view='dorsal'),
        6: dict(view='anterior'),
        7: dict(view='posterior'),
        8: dict(view='ventral'),
    }
    layout = layout.annotate(annotations)
    plot_f = plotdef(
        surf_from_archive(),
        surf_scalars_from_array(
            'log prob vMF',
            is_masked=is_masked,
        ),
        #vertex_to_face('encoder', interpolation='mode'),
        #plot_to_display(),
        text_element(
            name='title',
            content=f'log prob vMF {index}',
            bounding_box_height=128,
            font_size_multiplier=0.2,
            font_color='#cccccc',
            priority=-1,
        ),
        plot_to_image(),
        save_figure(
            layout_kernel=layout,
            padding=10,
            canvas_size=(2400, 300),
            canvas_color=(0, 0, 0),
            fname_spec=f'scalars-logProbVMF{index}',
            scalar_bar_action='collect',
        ),
    )
    plot_f(
        template='fsLR',
        surf_projection='veryinflated',
        log_prob_vmf_array=array,
        window_size=(600, 500),
        surf_scalars_cmap='magma',
        # surf_scalars_clim='robust',
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


def _get_data(
    cifti: str,
    normalise: bool = True,
    gsr: bool = True,
    key: Optional['jax.random.PRNGKey'] = None,
):
    key = jax.random.PRNGKey(0) if key is None else key
    cifti = nb.load(cifti)
    data_full = cifti.get_fdata(dtype=np.float32).T
    data = data_full[~cifti.header.get_axis(1).volume_mask]
    if normalise:
        data = data - data.mean(-1, keepdims=True)
        data = data / data.std(-1, keepdims=True)
        data = jnp.where(jnp.isnan(data), 0, data)

    if gsr:
        gs = data.mean(0, keepdims=True)
        data = residualise(data, gs)
    # Plug zero-variance vertices with ramp (for no NaNs in log prob)
    data = jnp.where(
        jnp.isclose(data.std(-1), 0)[..., None],
        jax.random.normal(key, data.shape),
        data,
    )
    return data


def get_data(cifti: str):
    return _get_data(CIFTI[cifti])


def whiten_data(
    enc: Tensor,
    parcels_enc: Optional[Tensor] = None,
    num_components: int = N_COMPONENTS,
) -> Tuple[Tensor, Tensor]:
    # sphering = jnp.where(jnp.arange(enc.shape[-1]) < num_components, 1., 0.)
    attenuated = enc.shape[-1] - num_components
    sphering = jnp.exp(
        -jnp.where(
            jnp.arange(enc.shape[-1]) < num_components,
            0.,
            jnp.arange(-num_components, attenuated) / jnp.sqrt(attenuated),
        )
    )
    W = generalised_whitening(enc, sphering=sphering)
    # W = generalised_whitening(enc)
    # L, Q = jnp.linalg.eigh(jnp.cov(enc.T))
    # W = Q @ jnp.diag(jnp.maximum(L, jnp.finfo(L.dtype).eps) ** -0.5) @ Q.T
    enc = enc @ W.T
    if parcels_enc is not None:
        parcels_enc = parcels_enc @ W.T
    return enc, parcels_enc


def threshold_data(
    atlas: Any,
    model: AtlasLinear,
    enc: Tensor,
    threshold: Optional[float] = None,
    threshold_locus: Optional[float] = None,
    binarise: bool = False,
) -> Tensor:
    if threshold is not None:
        repl = 1. if binarise else enc
        if isinstance(threshold, int):
            enc = jnp.where(
                enc < jnp.partition(enc, threshold)[..., threshold, None],
                0.,
                repl,
            )
        else:
            enc = jnp.where(
                enc < threshold,
                0.,
                repl,
            )
    elif threshold_locus is not None:
        repl = 1. if binarise else enc
        if isinstance(threshold_locus, float):
            threshold_locus = int(enc.shape[-2] * threshold_locus)
        enc = jnp.where(
            enc < jnp.partition(enc, threshold_locus, axis=-2)[None, threshold_locus, :],
            0.,
            repl,
        )
        enc = jnp.where(enc.sum(-1, keepdims=True) == 0, -1 / enc.shape[-1], enc)
    parcels_enc = model(enc, encode=False)
    return enc, parcels_enc


def visualise(
    indices: Sequence[int],
    cifti: str,
    encoder: str,
    threshold: Optional[float] = None,
    threshold_locus: Optional[float] = None,
    binarise: bool = False,
    whiten: bool = False,
):
    atlas = ENCODER_FACTORY[encoder]()

    model = AtlasLinear.from_atlas(atlas, encode=True, key=jax.random.PRNGKey(0))
    cifti = nb.load(CIFTI[cifti])
    data_full = cifti.get_fdata(dtype=np.float32).T
    data = data_full[~cifti.header.get_axis(1).volume_mask] #[atlas.mask.mask_array]

    gs = data.mean(0, keepdims=True)
    data = residualise(data, gs)
    # Plug zero-variance vertices with ramp (for no NaNs in log prob)
    data = jnp.where(
        (data.sum(-1) == 0)[..., None],
        np.arange(data.shape[-1])[None, :],
        data,
    )

    coors, parcels_enc, atlas_coors = ENCODE_SELF[encoder](
        model=model, data=data, atlas=atlas
    )

    enc = model(data)
    enc = jnp.concatenate((enc['cortex_L'], enc['cortex_R']))
    cortical_dim = (
        model.weight['cortex_L'].shape[0] + model.weight['cortex_R'].shape[0]
    )
    # Drop subcortical references
    enc = enc[..., :cortical_dim]
    parcels_enc = parcels_enc[..., :cortical_dim]

    enc, parcels_enc = threshold_data(
        atlas=atlas,
        model=model,
        enc=enc,
        threshold=threshold,
        threshold_locus=threshold_locus,
        binarise=binarise,
    )
    if whiten:
        enc, parcels_enc = whiten_data(enc, parcels_enc)
    enc = enc / jnp.linalg.norm(enc, axis=-1, keepdims=True)

    for index in indices:
        spatial = VonMisesFisher(mu=coors[index], kappa=10)
        temporal = VonMisesFisher(mu=parcels_enc[index], kappa=10)
        # plot_f = plotdef(
        #     surf_from_archive(),
        #     surf_scalars_from_array(
        #         'log prob vMF',
        #         is_masked=True,
        #     ),
        #     plot_to_display(),
        # )
        # plot_f(
        #     template='fsLR',
        #     surf_projection='veryinflated',
        #     log_prob_vmf_array=(
        #         spatial.log_prob(atlas.coors).squeeze()
        #         +
        #         temporal.log_prob(enc).squeeze()
        #     ),
        #     window_size=(600, 500),
        #     surf_scalars_cmap='inferno',
        #     #surf_scalars_clim='robust',
        #     # theme=pv.themes.DarkTheme(),
        #     hemisphere='left',
        # )
        visualise_surface_encoder(
            index=index,
            array=(
                # enc[..., index]
                spatial.log_prob(atlas_coors).squeeze() +
                temporal.log_prob(enc).squeeze()
            ),
        )
        mu = _to_jax_array(temporal.mu)
        plt.figure(figsize=(10, 10))
        plt.hist(
            (enc / jnp.linalg.norm(enc, axis=-1, keepdims=True)) @ mu,
            bins=100,
        )
        plt.savefig(f'/tmp/distributionTemporalInner{index}.png')
    assert 0


def main():
    encoder = '7net'
    visualise(
        INDICES[encoder], 'MSC', encoder, whiten=True
    )


if __name__ == '__main__':
    main()
