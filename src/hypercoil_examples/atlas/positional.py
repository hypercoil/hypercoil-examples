# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Positional encoding

Positional encoding module for the icospherical ELLGAT U-net model. Mostly
lifted from the analogous module from the SUGAR brain registration network,
which mostly follows the same architecture that we use here. The original:
https://github.com/IndiLab/SUGAR/blob/main/models/gatunet_model.py
SUGAR's module is itself based on the positional encoding module from the
neural radiance field model (NeRF).

It might be interesting to compare this positional encoding with the geometric
data from the Pang et al. paper.
"""
from typing import Any, Mapping, Optional, List, Union, Tuple

import jax
import jax.numpy as jnp
import equinox as eqx

from hypercoil.engine import Tensor


class PositionalEncoding(eqx.Module):
    funcs: Tuple[callable]
    freq_bands: Tensor

    def __init__(
        self,
        n_freq_bands: int = 4,
        funcs: Optional[Tuple[callable]] = None,
        log_scale: bool = True,
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


def main():
    import templateflow.api as tflow
    import nibabel as nb
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
    pe = PositionalEncoding()
    result_L = pe(coor_L)
    result_R = pe(coor_R)
    result = jnp.concatenate((result_L, result_R), axis=0)

    plot_f = plotdef(
        surf_from_archive(),
        surf_scalars_from_array('projection', is_masked=False),
        plot_to_image(),
        save_grid(
            n_cols=8,
            n_rows=result.shape[-1],
            padding=10,
            canvas_size=(3200, 300 * result.shape[-1]),
            canvas_color=(0, 0, 0),
            fname_spec='scalars-positionalencoding',
            sort_by=['surfscalars'],
            scalar_bar_action='collect',
        ),
    )
    plot_f(
        template='fsLR',
        load_mask=True,
        # TODO: This shouldn't cause an error! Fix it in hyve.
        # projection_array=result,
        projection_array_left=result_L,
        projection_array_right=result_R,
        surf_scalars_cmap='RdYlBu_r',
        surf_projection=('veryinflated',),
        hemisphere=['left', 'right', None],
        views={
            'left': ('medial', 'lateral'),
            'right': ('medial', 'lateral'),
            'both': ('dorsal', 'ventral', 'anterior', 'posterior'),
        },
        output_dir='/tmp',
        fname_spec='scalars-positionalencoding',
        window_size=(800, 600),
    )
    assert 0


if __name__ == "__main__":
    main()
