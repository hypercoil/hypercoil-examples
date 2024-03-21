# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Annular decomposition: identifiability
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Identifiability of subjects based on annular decompositions of their
selectivity profiles.
"""
import pickle
from typing import List, Tuple

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from hypercoil.engine import Tensor
from hypercoil_examples.atlas.aligngrouplevel import (
    discr_impl, within_between_distance
)

NUM_MAPS = 32


def compute_distance(projections: Tensor, index: int) -> Tuple[
    Tensor, List[Tuple[str, str]]
]:
    subjects = list(projections.keys())
    sessions = list(projections[subjects[0]].keys())
    ident = [(sub, ses) for sub in subjects for ses in sessions]
    P = jnp.stack([
        projections[sub][ses][index]
        for sub in subjects for ses in sessions
    ])
    return jnp.arccos(jnp.abs(jnp.corrcoef(P))), ident


def plot_distance(distance: Tensor, index: int) -> None:
    plt.imshow(distance, cmap='inferno_r', vmin=0.9, vmax=1.1)
    #plt.xticks(jnp.arange(len(distance)), labels=ident, rotation=90)
    plt.xticks([])
    plt.yticks([])
    plt.colorbar()
    plt.savefig(f'/tmp/identifiability-annular{index}.png')
    plt.close('all')


def main():
    with open('/tmp/subjectspecificprojections.pkl', 'rb') as f:
        projections = pickle.load(f)
    discriminability = {}
    wbdist = {}
    for i in range(NUM_MAPS):
        print(f'Processing map {i + 1}/{NUM_MAPS}')
        distance, ident = compute_distance(projections=projections, index=i)
        plot_distance(distance, index=i)
        discriminability[i] = discr_impl(
            np.asarray(distance),
            np.asarray([e[0] for e in ident]),
        )
        n_sessions = len(projections[list(projections.keys())[0]])
        index = (
            (n_sessions * np.floor(np.arange(distance.shape[0]) / n_sessions))[:, None] ==
            (n_sessions * np.floor(np.arange(distance.shape[0]) / n_sessions))
        )
        bdist = distance[~index].mean()
        wdist = distance[index].mean()
        wbdist[i] = (bdist - wdist) / distance.mean()
    print(discriminability)
    print(discriminability.mean())
    print(wbdist)
    assert 0


if __name__ == '__main__':
    main()
