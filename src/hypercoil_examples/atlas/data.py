# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Load and transform data
~~~~~~~~~~~~~~~~~~~~~~~
Load and transform CIfTI-formatted neuroimaging data.
"""
from typing import Optional

import jax
import jax.numpy as jnp
import nibabel as nb
import numpy as np
import pandas as pd
from scipy.signal import filtfilt, butter

from hypercoil.functional import residualise
from hypercoil_examples.atlas.const import (
    MSC_DATA_ROOT, HCP_DATA_ROOT
)


def get_msc_dataset(
    subject: str,
    session: str,
    task: str = 'rest',
    get_confounds: bool = False,
) -> str:
    cifti = (
        f'{MSC_DATA_ROOT}/sub-MSC{subject}_ses-func{session}_'
        f'task-{task}_space-fsLR_den-91k_bold.dtseries.nii'
    )
    if get_confounds:
        confounds = (
            f'{MSC_DATA_ROOT}/sub-MSC{subject}_ses-func{session}_'
            f'task-{task}_desc-confounds_timeseries.tsv'
        )
        data = (cifti, confounds)
    else:
        data = cifti
    return data


def get_hcp_dataset(
    subject: str,
    run: str = 'LR',
    task: str = 'REST1',
    get_confounds: bool = False,
):
    task = task.upper()
    scan_type = 'tfMRI'
    if task[:4] == 'REST':
        scan_type = 'rfMRI'
    cifti = (
        f'{HCP_DATA_ROOT}/{subject}/MNINonLinear/Results/'
        f'{scan_type}_{task}_{run}/{scan_type}_{task}_{run}_'
        'Atlas_MSMAll.dtseries.nii'
    )
    if get_confounds:
        confounds = (
            f'{HCP_DATA_ROOT}/{subject}/MNINonLinear/Results/'
            f'{scan_type}_{task}_{run}/Confound_Regressors.tsv'
        )
        data = (cifti, confounds)
    else:
        data = cifti
    return data


def _get_data(
    cifti: str,
    confounds: Optional[str] = None,
    normalise: bool = True,
    gsr: bool = True,
    filter_rps: bool = True,
    censor_thresh: float = 0.15,
    pad_to_size: Optional[int] = None,
    key: Optional['jax.random.PRNGKey'] = None,
):
    key = jax.random.PRNGKey(0) if key is None else key
    try:
        cifti = nb.load(cifti)
    except nb.filebasedimages.ImageFileError:
        # callers will be looking for FileNotFoundError
        raise FileNotFoundError
    data_full = cifti.get_fdata(dtype=np.float32).T
    data = data_full[~cifti.header.get_axis(1).volume_mask]
    if normalise:
        data = data - data.mean(-1, keepdims=True)
        data = data / data.std(-1, keepdims=True)
        data = jnp.where(jnp.isnan(data), 0, data)

    if gsr:
        gs = data.mean(0, keepdims=True)
        data = residualise(data, gs)
    if confounds:
        confounds = pd.read_csv(confounds, sep='\t')
        rp_cols = ['trans_x', 'trans_y', 'trans_z', 'rot_x', 'rot_y', 'rot_z']
        rp_cols = [f'{e}_derivative1' for e in rp_cols]
        rps = confounds[rp_cols].values
        rps[0] = rps[1]
        if filter_rps:
            t_rep = cifti.header.matrix._mims[0].series_step
            fs = 1 / t_rep
            fc = 0.1 # Cut-off frequency of the filter (6 bpm)
            filt = butter(N=1, Wn=fc, btype='low', fs=fs)
            rps = filtfilt(*filt, rps, axis=0)
        fd = jnp.abs(rps).sum(-1)
        tmask = fd <= censor_thresh
        if pad_to_size is not None:
            base_size = tmask.sum()
            if pad_to_size <= base_size:
                index = jax.random.choice(
                    key,
                    jnp.where(tmask)[0],
                    shape=(pad_to_size,),
                    replace=False,
                )
            else:
                index = jnp.concatenate((
                    jnp.where(tmask)[0],
                    jax.random.choice(
                        jax.random.PRNGKey(0),
                        jnp.where(tmask)[0],
                        shape=(pad_to_size - base_size,),
                        replace=True,
                    ),
                ))
        else:
            index = jnp.where(tmask)[0]
        data = data[..., index]
    # Plug zero-variance vertices with ramp (for no NaNs in log prob)
    data = jnp.where(
        jnp.isclose(data.std(-1), 0)[..., None],
        jax.random.normal(key, data.shape),
        data,
    )
    return data

