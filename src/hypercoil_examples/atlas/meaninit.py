# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Mean atlas initialisation
~~~~~~~~~~~~~~~~~~~~~~~~~
Initialise an alignment atlas tensor as the mean of the input image encodings.
"""
from typing import Literal
import glob
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import nibabel as nb
import numpy as np
from hypercoil.functional import residualise
from hypercoil.nn import AtlasLinear
from hypercoil_examples.atlas.const import MSC_DATA_ROOT
from hypercoil_examples.atlas.encoders import (
    create_icosphere_encoder,
)
from hypercoil_examples.atlas.multiencoder import configure_multiencoder
from hypercoil_examples.atlas.vmf import ENCODE_SELF, whiten_data

IMGS = sorted(glob.glob(
    f'{MSC_DATA_ROOT}/'
    'sub-MSC0*_ses-func0*_task-rest_space-fsLR_den-91k_bold.dtseries.nii'
))
#IMGS = IMGS + ['/Users/rastkociric/Downloads/rfMRI_REST1_LR_Atlas_MSMAll.dtseries.nii']


def main(
    encoder_type: Literal['icosphere', 'multiencoder'] = 'icosphere',
    normalise: bool = True,
):
    match encoder_type:
        case 'icosphere':
            atlas = create_icosphere_encoder()
            encoder = AtlasLinear.from_atlas(
                atlas,
                encode=True,
                key=jax.random.PRNGKey(0),
            )
        case 'multiencoder':
            encoder = configure_multiencoder(use_7net=False)
    for i, img in enumerate(IMGS):
        print(f'Processing {img} ({i+1}/{len(IMGS)})')
        data = nb.load(img)
        data_full = jnp.asarray(data.get_fdata()).T
        data = data_full[~data.header.get_axis(1).volume_mask] #[atlas.mask.mask_array]

        if normalise:
            data = data - data.mean(-1, keepdims=True)
            data = data / data.std(-1, keepdims=True)
            data = jnp.where(jnp.isnan(data), 0, data)
        gs = data.mean(0, keepdims=True)
        data = residualise(data, gs)
        # Plug zero-variance vertices with ramp (for no NaNs in log prob)
        data = jnp.where(
            (data.sum(-1) == 0)[..., None],
            jnp.arange(data.shape[-1])[None, :],
            data,
        )
        enc = encoder(data)
        if encoder_type == 'icosphere':
            enc = jnp.concatenate((enc['cortex_L'], enc['cortex_R']))
        enc, _ = whiten_data(enc)
        enc = enc / jnp.linalg.norm(enc, axis=-1, keepdims=True)
        if jnp.any(jnp.isnan(enc)):
            print(f'NaN detected: {img}')
            print('Please verify the validity of the input. Skipping . . .')
            continue
        enc = jnp.where(
            jnp.isnan(enc), 0, enc
        )
        if i == 0:
            init = enc
        else:
            init += enc

    init = init / len(IMGS)
    plt.figure(figsize=(10, 10))
    plt.hist(jnp.linalg.norm(init, axis=-1), bins=200)
    plt.savefig('/tmp/mean_init.png')
    if encoder_type == 'multiencoder':
        init = encoder.rescale(init)
    np.save('/tmp/mean_init.npy', np.asarray(init), allow_pickle=False)
    assert 0


if __name__ == '__main__':
    main(encoder_type='multiencoder')
