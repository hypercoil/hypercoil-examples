# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Sphering to thresh+bin
~~~~~~~~~~~~~~~~~~~~~~
See if we can choose a sphering for the data that approximates the
thresh+bin approach.
"""
from typing import Optional, Sequence, Tuple

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import nibabel as nb
import numpy as np

from hypercoil.engine import Tensor, _to_jax_array
from hypercoil.functional import (
    residualise
)
from hypercoil.init import VonMisesFisher
from hypercoil.nn import AtlasLinear
from hypercoil_examples.atlas.sphering import (
    AdaptiveSpheringArctangent,
    AdaptiveSpheringHalfArctangent
)
from hypercoil_examples.atlas.vmf import (
    ENCODER_FACTORY, INDICES, CIFTI, ENCODE_SELF,
    threshold_data,
    visualise_surface_encoder,
    generalised_whitening,
)


SPHERING_COMPONENTS = (
    tuple(range(0, 8)) +
    tuple(range(8, 16, 2)) +
    tuple(range(16, 32, 4)) +
    tuple(range(32, 64, 8)) +
    tuple(range(64, 129, 16))
)


def whiten_data(
    enc: Tensor,
    parcels_enc: Optional[Tensor] = None,
    num_components: int = 1,
) -> Tuple[Tensor, Tensor]:
    sphering = jnp.where(jnp.arange(enc.shape[-1]) < num_components, 1., 0.)
    W = generalised_whitening(enc, sphering=sphering)
    enc = enc @ W.T
    if parcels_enc is not None:
        parcels_enc = parcels_enc @ W.T
    return enc, parcels_enc


def visualise(
    indices: Sequence[int],
    cifti: str,
    encoder: str,
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

    enc_thr, parcels_enc_thr = threshold_data(
        atlas=atlas,
        model=model,
        enc=enc,
        threshold=None,
        threshold_locus=0.9,
        binarise=True,
    )
    enc_thr = enc_thr / jnp.linalg.norm(enc_thr, axis=-1, keepdims=True)
    parcels_enc_thr = parcels_enc_thr / jnp.linalg.norm(
        parcels_enc_thr, axis=-1, keepdims=True
    )
    parcels_enc_thr = parcels_enc_thr.T[..., :cortical_dim].T
    target = parcels_enc_thr @ enc_thr.T

    corrs = []
    maxcorr = -1
    best_L = None
    best_enc = None
    best_parcels_enc = None
    total_n = enc.shape[-1]
    for num_components in SPHERING_COMPONENTS:
        # enc_sph, parcels_enc_sph = whiten_data(
        #     enc, parcels_enc, num_components=num_components
        # )
        W = AdaptiveSpheringArctangent(
            inflection=num_components / total_n,
            lim=1e6,
            floor=0.,
        )(enc)
        enc_sph = enc @ W.T
        parcels_enc_sph = parcels_enc @ W.T
        enc_sph = enc_sph / jnp.linalg.norm(enc_sph, axis=-1, keepdims=True)
        parcels_enc_sph = parcels_enc_sph / jnp.linalg.norm(
            parcels_enc_sph, axis=-1, keepdims=True
        )
        sphered = parcels_enc_sph @ enc_sph.T
        corrs += [jnp.corrcoef(target.ravel(), sphered.ravel())[0, 1]]
        if corrs[-1] > maxcorr:
            maxcorr = corrs[-1]
            best_L, _ = jnp.linalg.eigh(jnp.cov(enc_sph.T))
            best_enc = enc_sph
            best_parcels_enc = parcels_enc_sph
        print(num_components, corrs[-1])

    plt.figure(figsize=(10, 10))
    plt.stem(
        jnp.asarray(SPHERING_COMPONENTS),
        jnp.asarray(corrs),
        linefmt='grey',
    )
    plt.savefig('/tmp/sphthr_correlations.png')

    L, _ = jnp.linalg.eigh(jnp.cov(enc.T))
    plt.figure(figsize=(10, 10))
    plt.stem(
        jnp.arange(L.shape[0]),
        jnp.flip(L),
        linefmt='grey',
    )
    plt.savefig('/tmp/orig_eigenvalues.png')

    plt.figure(figsize=(10, 10))
    plt.stem(
        jnp.arange(best_L.shape[0]),
        jnp.flip(best_L),
        linefmt='grey',
    )
    plt.savefig('/tmp/sph_eigenvalues.png')

    L, _ = jnp.linalg.eigh(jnp.cov(enc_thr.T))
    plt.figure(figsize=(10, 10))
    plt.stem(
        jnp.arange(L.shape[0]),
        jnp.flip(L),
        linefmt='grey',
    )
    plt.savefig('/tmp/thr_eigenvalues.png')

    for index in indices:
        spatial = VonMisesFisher(mu=coors[index], kappa=10)
        temporal = VonMisesFisher(mu=best_parcels_enc[index], kappa=10)
        visualise_surface_encoder(
            index=index,
            array=(
                # enc[..., index]
                spatial.log_prob(atlas_coors).squeeze() +
                temporal.log_prob(best_enc).squeeze()
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
    encoder = 'icosphere'
    visualise(
        INDICES[encoder], 'MSC', encoder
    )


if __name__ == '__main__':
    main()
