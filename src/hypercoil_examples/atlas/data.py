# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Load and transform data
~~~~~~~~~~~~~~~~~~~~~~~
Load and transform CIfTI-formatted neuroimaging data.
"""
from typing import Literal, Optional

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


def inject_noise_to_zero_variance(
    data: jnp.ndarray,
    sampler: callable = jax.random.normal,
    *,
    key: 'jax.random.PRNGKey',
) -> jnp.ndarray:
    return jnp.where(
        jnp.isclose(data.std(-1), 0)[..., None],
        sampler(key, data.shape),
        data,
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


PARAM9KEY = sum(
    [[f'trans_{d}', f'rot_{d}'] for d in ('x', 'y', 'z')], []
) + ['white_matter', 'csf', 'global_signal']


def _get_data(
    cifti: Optional[str] = None,
    confounds: Optional[str] = None,
    normalise: bool = True,
    mgtr: bool = True,
    gsr: bool = False,
    filter_rps: bool = True,
    censor_thresh: float = 0.15,
    pad_to_size: Optional[int] = None,
    censor_method: Literal['drop', 'zero'] = 'drop',
    t_rep: float = 0.72,
    filter_data: bool = True,
    param36: bool = False,
    *,
    data: Optional[jnp.ndarray] = None,
    key: Optional['jax.random.PRNGKey'] = None,
):
    key = jax.random.PRNGKey(0) if key is None else key
    if data is None:
        try:
            cifti = nb.load(cifti)
        except nb.filebasedimages.ImageFileError:
            # callers will be looking for FileNotFoundError
            raise FileNotFoundError
        data_full = cifti.get_fdata(dtype=np.float32).T
        data = data_full[~cifti.header.get_axis(1).volume_mask]
        t_rep = cifti.header.matrix._mims[0].series_step
    if t_rep > 100: # Convert to seconds
        t_rep /= 1000
    if normalise:
        data = data - data.mean(-1, keepdims=True)
        data = data / data.std(-1, keepdims=True)
        data = jnp.where(jnp.isnan(data), 0, data)

    if mgtr or (gsr and not confounds):
        # This is really MGTR, not GSR
        gs = data.mean(0, keepdims=True)
        gsd = jnp.concatenate((np.zeros((1, 1)), np.diff(gs)), -1)
        gs = jnp.concatenate((gs, gsd))
        data = residualise(data, gs)
    if confounds:
        confounds = pd.read_csv(confounds, sep='\t')
        if param36:
            param18key = PARAM9KEY + [f'{e}_derivative1' for e in PARAM9KEY]
            param36key = param18key + [f'{e}_power2' for e in param18key]
            param36model = confounds[param36key].values.T
            # So that nothing explodes when we invert
            breakpoint()
            param36model = param36model - param36model.mean(-1, keepdims=True)
            param36model = param36model / param36model.std(-1, keepdims=True)
            param36model = jnp.where(jnp.isnan(param36model), 0, param36model)
            data = residualise(data, param36model)
        elif gsr:
            if mgtr:
                raise ValueError('Combining GSR and MGTR')
            gs = confounds['global_signal'].values[None]
            gsd = jnp.concatenate((np.zeros((1, 1)), np.diff(gs)), -1)
            gs = jnp.concatenate((gs, gsd))
            data = residualise(data, gs)
        rp_cols = ['trans_x', 'trans_y', 'trans_z', 'rot_x', 'rot_y', 'rot_z']
        rp_cols = [f'{e}_derivative1' for e in rp_cols]
        rps = np.array(confounds[rp_cols].values)
        rps[0] = rps[1]
        fs = 1 / t_rep
        if filter_rps:
            fc = 0.1 # Cut-off frequency of the filter (6 bpm)
            filt = butter(N=1, Wn=fc, btype='low', fs=fs)
            rps = filtfilt(*filt, rps, axis=0)
        fd = jnp.abs(rps).sum(-1)
        tmask = fd <= censor_thresh
        surviving_frames = jnp.where(tmask)[0]
        if filter_data:
            fc = (0.01, 0.1)
            filt = butter(N=1, Wn=fc, btype='bandpass', fs=fs)
            if jnp.any(~tmask):
                breakpoint()
                jnp.interp
            data = filtfilt(*filt, data, axis=-1)
        if pad_to_size is not None:
            base_size = tmask.sum()
            if pad_to_size <= base_size:
                index = jax.random.choice(
                    key,
                    surviving_frames,
                    shape=(pad_to_size,),
                    replace=False,
                )
            else:
                index = jnp.concatenate((
                    surviving_frames,
                    jax.random.choice(
                        jax.random.PRNGKey(0),
                        surviving_frames,
                        shape=(pad_to_size - base_size,),
                        replace=True,
                    ),
                ))
        else:
            index = surviving_frames
        if censor_method == 'drop':
            data = data[..., index]
        else:
            data = jnp.where(tmask[None, ...], data, 0)
    # Plug zero-variance vertices with ramp (for no NaNs in log prob)
    data = inject_noise_to_zero_variance(data, key=key)
    return data
