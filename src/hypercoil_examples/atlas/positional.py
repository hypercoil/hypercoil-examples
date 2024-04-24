# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Positional encoding
~~~~~~~~~~~~~~~~~~~
Positional encoding module for the icospherical ELLGAT U-net model. Mostly
lifted from the analogous module from the SUGAR brain registration network,
which mostly follows the same architecture that we use here. The original:
https://github.com/IndiLab/SUGAR/blob/main/models/gatunet_model.py
SUGAR's module is itself based on the positional encoding module from the
neural radiance field model (NeRF).

It might be interesting to compare this positional encoding with the geometric
data from the Pang et al. paper.
"""
from typing import Any, Mapping, Optional, Sequence, Tuple, Union

import jax
import jax.numpy as jnp
import equinox as eqx
import nibabel as nb
import pandas as pd

from hypercoil.engine import Tensor

from hypercoil_examples.atlas.const import EIGENMODES_ROOT

EIGENMODES_PATH = {
    'cortex_L': (
        f'{EIGENMODES_ROOT}'
        'fsLR_32k_midthickness-lh_emode_200.txt'
    ),
    'cortex_R': (
        f'{EIGENMODES_ROOT}'
        'fsLR_32k_midthickness-rh_emode_200.txt'
    ),
}


class PositionalEncoding(eqx.Module):
    funcs: Tuple[callable, ...]
    freq_bands: Tensor

    def __init__(
        self,
        n_freq_bands: int = 4,
        funcs: Optional[Tuple[callable]] = None,
        log_scale: bool = True,
        *,
        key: Optional['jax.random.PRNGKey'] = None,
    ):
        self.funcs = funcs or (
            jnp.sin,
            jnp.cos,
        )

        if log_scale:
            self.freq_bands = 2 ** (
                jnp.linspace(0.0, (n_freq_bands - 1), n_freq_bands)
            ) * jnp.pi
        else:
            self.freq_bands = jnp.linspace(
                1.0, 2 ** (n_freq_bands - 1), n_freq_bands
            ) * jnp.pi

    def __call__(self, X: Tensor) -> Tensor:
        Y = X[..., None] * self.freq_bands
        X = jnp.concatenate(
            [X[..., None]] + [f(Y) for f in self.funcs],
            axis=-1,
        )
        return X.reshape(X.shape[:-2] + (-1,))


class GeometricEncoding(eqx.Module):
    default_eigenmodes: Tuple[int, ...]
    eigenmodes: Mapping[str, Tensor]
    default_encoding_dim: int

    def __init__(
        self,
        eigenmodes_path: Union[str, Mapping[str, str]],
        mask_path: Union[str, Mapping[str, str]],
        default_eigenmodes: Union[Sequence[int], Mapping[str, Sequence[int]]],
        *,
        key: Optional['jax.random.PRNGKey'] = None,
    ):
        if isinstance(eigenmodes_path, str):
            eigenmodes_path = {'all': eigenmodes_path}
        if isinstance(mask_path, str):
            mask_path = {'all': mask_path}
        if not isinstance(default_eigenmodes, Mapping):
            self.default_eigenmodes = {'all': tuple(default_eigenmodes)}
        else:
            self.default_eigenmodes = {
                k: tuple(v) for k, v in default_eigenmodes.items()
            }
        mask = {
            k: nb.load(v).darrays[0].data.astype(bool)
            for k, v in mask_path.items()
        }
        self.eigenmodes = {
            k: pd.read_csv(v, sep=' ', header=None).values[mask[k]]
            for k, v in eigenmodes_path.items()
        }
        self.default_encoding_dim = (
            3 + len(next(iter(self.default_eigenmodes.values())))
        )

    def __call__(
        self,
        X: Tensor,
        modes: Optional[Sequence[int]] = None,
        geom: str = 'all',
        *,
        key: Optional['jax.random.PRNGKey'] = None,
    ) -> Tensor:
        if modes is None:
            modes = self.default_eigenmodes[geom]
        return jnp.concatenate(
            [X] + [self.eigenmodes[geom][:, modes]],
            axis=-1,
        )


def configure_geometric_encoder():
    import templateflow.api as tflow
    mask_path = {
        k: str(
            tflow.get('fsLR', density='32k', hemi=k[-1], desc='nomedialwall')
        )
        for k in ('cortex_L', 'cortex_R')
    }
    ge = GeometricEncoding(
        eigenmodes_path=EIGENMODES_PATH,
        mask_path=mask_path,
        default_eigenmodes={
            k: range(1, 25) for k in ('cortex_L', 'cortex_R')
        },
    )
    return ge


def get_coors():
    import templateflow.api as tflow
    coor_L_path = tflow.get(
        'fsLR', density='32k', hemi='L', space=None, suffix='sphere'
    )
    coor_R_path = tflow.get(
        'fsLR', density='32k', hemi='R', space=None, suffix='sphere'
    )
    coor_L = nb.load(coor_L_path).darrays[0].data / 100
    coor_R = nb.load(coor_R_path).darrays[0].data / 100
    mask_path = {
        k: str(tflow.get('fsLR', density='32k', hemi=k, desc='nomedialwall'))
        for k in ('L', 'R')
    }
    coor_L = coor_L[nb.load(mask_path['L']).darrays[0].data.astype(bool)]
    coor_R = coor_R[nb.load(mask_path['R']).darrays[0].data.astype(bool)]
    return coor_L, coor_R


def main():
    import templateflow.api as tflow
    from hyve import (
        plotdef,
        surf_from_archive,
        surf_scalars_from_array,
        plot_to_image,
        save_grid,
    )
    coor_L_path = tflow.get(
        'fsLR', density='32k', hemi='L', space=None, suffix='sphere'
    )
    coor_R_path = tflow.get(
        'fsLR', density='32k', hemi='R', space=None, suffix='sphere'
    )
    coor_L = nb.load(coor_L_path).darrays[0].data / 100
    coor_R = nb.load(coor_R_path).darrays[0].data / 100
    mask_path = {
        k: str(tflow.get('fsLR', density='32k', hemi=k, desc='nomedialwall'))
        for k in ('L', 'R')
    }
    coor_L = coor_L[nb.load(mask_path['L']).darrays[0].data.astype(bool)]
    coor_R = coor_R[nb.load(mask_path['R']).darrays[0].data.astype(bool)]
    pe = PositionalEncoding()
    pe_L = pe(coor_L)
    pe_R = pe(coor_R)
    pe_all = jnp.concatenate((pe_L, pe_R), axis=0)

    ge = configure_geometric_encoder()
    ge_L = ge(coor_L, geom='cortex_L')
    ge_R = ge(coor_R, geom='cortex_R')
    ge_all = jnp.concatenate((ge_L, ge_R), axis=0)

    plot_f = plotdef(
        surf_from_archive(),
        #surf_scalars_from_array('projection', is_masked=False),
        surf_scalars_from_array('projection', is_masked=True),
        plot_to_image(),
        save_grid(
            n_cols=8,
            n_rows=pe_L.shape[-1],
            padding=10,
            canvas_size=(3200, 300 * pe_L.shape[-1]),
            canvas_color=(0, 0, 0),
            fname_spec='scalars-encoding',
            sort_by=['surfscalars'],
            scalar_bar_action='collect',
        ),
    )
    for k, v in {'pe': (pe_L, pe_R), 'ge': (ge_L, ge_R)}.items():
        plot_f(
            template='fsLR',
            load_mask=True,
            # TODO: This shouldn't cause an error! Fix it in hyve.
            # projection_array=result,
            projection_array_left=v[0].T,
            projection_array_right=v[1].T,
            surf_scalars_cmap='RdYlBu_r',
            surf_projection=('veryinflated',),
            hemisphere=['left', 'right', None],
            views={
                'left': ('medial', 'lateral'),
                'right': ('medial', 'lateral'),
                'both': ('dorsal', 'ventral', 'anterior', 'posterior'),
            },
            output_dir='/tmp',
            fname_spec=f'scalars-{k}',
            window_size=(800, 600),
        )
    assert 0


if __name__ == "__main__":
    main()
