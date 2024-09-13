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


def _normalise(data: jnp.ndarray) -> jnp.ndarray:
    data = data - data.mean(-1, keepdims=True)
    data = data / data.std(-1, keepdims=True)
    data = jnp.where(jnp.isnan(data), 0, data)
    return data


def _get_data(
    cifti: Optional[str] = None,
    confounds: Optional[str] = None,
    normalise: bool = True,
    denoising: Literal[
        'mgtr',
        'mgtr_deriv',
        'gsr',
        'gsr_deriv',
        'param18',
        'param36',
        'mgtr+18',
        'mgtr+36',
    ] | None = 'mgtr+18',
    filter_rps: bool = True,
    filter_data: bool = True,
    censor_thresh: float = 0.15,
    pad_to_size: Optional[int] = None,
    censor_method: Literal['drop', 'zero'] = 'drop',
    t_rep: float = 0.72,
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
        data = _normalise(data)

    confmodel = None
    if denoising in ('mgtr', 'mgtr_deriv', 'mgtr+18', 'mgtr+36') or (
        denoising in ('gsr', 'gsr_deriv') and not confounds
    ):
        # This is really MGTR, not GSR
        confmodel = data.mean(0, keepdims=True)
        if denoising in ('mgtr_deriv', 'gsr_deriv', 'mgtr+18', 'mgtr+36'):
            gsd = jnp.concatenate((np.zeros((1, 1)), np.diff(confmodel)), -1)
            confmodel = jnp.concatenate((confmodel, gsd))
            if denoising in ('mgtr+18', 'mgtr+36'):
                confmodel = jnp.concatenate(
                    (confmodel, confmodel ** 2), -1
                )
                mgtr = pd.DataFrame({
                    'global_signal': confmodel[0],
                    'global_signal_derivative1': confmodel[1],
                    'global_signal_power2': confmodel[2],
                    'global_signal_derivative1_power2': confmodel[3],
                })
    if confounds:
        confounds = pd.read_csv(confounds, sep='\t')
        rp_cols = ['trans_x', 'trans_y', 'trans_z', 'rot_x', 'rot_y', 'rot_z']
        rp_cols = [f'{e}_derivative1' for e in rp_cols]
        rps = np.array(confounds[rp_cols].values)
        rps[0] = rps[1]
        fs = 1 / t_rep
        if filter_rps:
            fc = 0.1 # Cut-off frequency of the filter (6 bpm)
            filt = butter(N=1, Wn=fc, btype='low', fs=fs)
            rps = filtfilt(*filt, rps, axis=0)
            confounds[rp_cols] = rps
        fd = jnp.abs(rps).sum(-1)
        tmask = fd <= censor_thresh
        if denoising in ('mgtr+18', 'mgtr+36'):
            confounds[mgtr.columns] = mgtr
            if denoising == 'mgtr+18': denoising = 'param18'
            if denoising == 'mgtr+36': denoising = 'param36'
        if denoising in ('param18', 'param36'):
            model_key = PARAM9KEY + [f'{e}_derivative1' for e in PARAM9KEY]
            if denoising == 'param36':
                model_key = model_key + [f'{e}_power2' for e in model_key]
            confmodel = confounds[model_key].values.T
            # So that nothing explodes when we invert
            confmodel = _normalise(confmodel)
        elif denoising in ('gsr', 'gsr_deriv'):
            confmodel = confounds['global_signal'].values[None]
            if denoising == 'gsr_deriv':
                gsd = jnp.concatenate((np.zeros((1, 1)), np.diff(confmodel)), -1)
                confmodel = jnp.concatenate((confmodel, gsd))
        if filter_data:
            fc = (0.01, 0.1)
            filt = butter(N=1, Wn=fc, btype='bandpass', fs=fs)
            if jnp.any(~tmask):
                f_interp = jax.vmap(
                    jnp.interp,
                    in_axes=(None, None, 0),
                    out_axes=0,
                )
                data_interp = f_interp(
                    jnp.where(~tmask)[0],
                    jnp.where(tmask)[0],
                    data[..., tmask],
                )
                data = data.at[..., ~tmask].set(data_interp)
                if confmodel is not None:
                    confmodel = confmodel.at[..., ~tmask].set(
                        f_interp(
                            jnp.where(~tmask)[0],
                            jnp.where(tmask)[0],
                            confmodel[..., tmask],
                        )
                    )
            data = filtfilt(*filt, data, axis=-1)
            if confmodel is not None:
                confmodel = filtfilt(*filt, confmodel, axis=-1)
        surviving_frames = jnp.where(tmask)[0]
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
            if confmodel is not None:
                confmodel = confmodel[..., index]
        else:
            data = jnp.where(tmask[None, ...], data, 0)
            if confmodel is not None:
                confmodel = jnp.where(tmask[None, ...], confmodel, 0)
    data = residualise(data, confmodel)
    # Plug zero-variance vertices with ramp (for no NaNs in log prob)
    data = inject_noise_to_zero_variance(data, key=key)
    return data


def main():
    import matplotlib.pyplot as plt
    import seaborn as sns
    from hypercoil.functional import sym2vec
    from hyve_examples import get_schaefer400_cifti
    cifti, confounds = get_msc_dataset('01', '01', task='rest', get_confounds=True)
    data = _get_data(cifti, confounds, denoising='mgtr+36', censor_thresh=0.1)

    rsts = pd.read_csv(
        (
            '/Users/rastkociric/Downloads/miniMSC/data/ts/'
            'sub-MSC01_ses-func01_task-rest_run-01_schaefer400_ts.1D'
        ),
        sep=' ',
        header=None,
    )
    ref = np.corrcoef(rsts.T)
    plt.imshow(ref, vmin=-0.25, vmax=0.25, cmap='RdYlBu_r'); plt.show()

    atlas = nb.load(get_schaefer400_cifti()).get_fdata().astype(int) - 1
    atlas_mat = np.where(
        (atlas < 0)[..., None],
        0,
        np.eye(atlas.max() + 1)[atlas],
    ).squeeze().T
    data_parc = jnp.linalg.lstsq(atlas_mat.T, data)[0]
    result = np.corrcoef(data_parc)
    plt.imshow(result, vmin=-0.25, vmax=0.25, cmap='RdYlBu_r'); plt.show()
    print(np.corrcoef(sym2vec(result), sym2vec(ref))[0, 1])
    sns.kdeplot(data=pd.DataFrame({'ref': sym2vec(ref), 'result': sym2vec(result)}))
    plt.show()
    breakpoint()


if __name__ == '__main__':
    main()
