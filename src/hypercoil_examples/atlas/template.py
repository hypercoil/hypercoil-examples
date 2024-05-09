# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Training loop
~~~~~~~~~~~~~
Training loop for the parcellation model
"""
import pickle
from itertools import product
from typing import Optional

import equinox as eqx
import jax.numpy as jnp
import numpy as np

from hypercoil.engine import _to_jax_array
from hypercoil_examples.atlas.aligned_dccc import (
    get_msc_dataset, _get_data
)
from hypercoil_examples.atlas.model import init_encoder_model
from hypercoil_examples.atlas.positional import get_coors

MAX_EPOCH = 2000
CHECKPOINT_INTERVAL = 100
SUBJECTS = ('01', '02', '03', '04', '05', '06', '07', '08', '09', '10',)
SESSIONS = ('01', '02', '03', '04', '05', '06', '07', '08', '09', '10',)


def null_spatial_encoder(*pparams, **params):
    return None


def main():
    data_entities = tuple(product(SESSIONS, SUBJECTS))
    num_entities = len(data_entities)
    coor_L, coor_R = get_coors()
    model, template = init_encoder_model(coor_L=coor_L)
    encode = eqx.filter_jit(model)
    #encode = model
    epoch_history = []
    new_template = None
    update_weight = 0
    for i in range(0, MAX_EPOCH):
        session, subject = data_entities[i % num_entities]
        print(f'Epoch {i} (sub-{subject} ses-{session})')
        try:
            T = _get_data(
                get_msc_dataset(subject, session),
                normalise=False,
                gsr=False,
            )
        except FileNotFoundError:
            print(
                f'Data entity sub-{subject} ses-{session} is absent. '
                'Skipping'
            )
        if jnp.any(jnp.isnan(T)):
            print(
                f'Invalid data for entity sub-{subject} ses-{session}. '
                'Skipping'
            )
            continue
        _, (template, new_new_template, new_update_weight, _, _) = encode(
            T=T,
            coor_L=coor_L,
            coor_R=coor_R,
            M=template,
            new_M=new_template,
            update_weight=update_weight,
        )
        if (
            jnp.any(jnp.isnan(new_new_template['cortex_L'])) or
            jnp.any(jnp.isnan(new_new_template['cortex_R']))
        ):
            print(
                f'Invalid data for entity sub-{subject} ses-{session}. '
                'Skipping'
            )
            continue
        new_template = new_new_template
        update_weight = new_update_weight
        if ((i % num_entities == 0) and (i != 0)):
            delta_norm_L = jnp.linalg.norm(
                new_template['cortex_L'] - template['cortex_L']
            )
            delta_norm_R = jnp.linalg.norm(
                new_template['cortex_R'] - template['cortex_R']
            )
            delta_norm = delta_norm_L + delta_norm_R
            epoch_history += [(delta_norm,)]
            print(f'Epoch {i} delta norm: {delta_norm}')
            template = {
                compartment: model.temporal.rescale(new_template[compartment])
                for compartment in ('cortex_L', 'cortex_R')
            }
            update_weight = 0
        if i % CHECKPOINT_INTERVAL == 0:
            print('Serialising template for checkpoint')
            template_concat = np.asarray(jnp.concatenate(
                (template['cortex_L'], template['cortex_R'])
            ))
            np.save(
                '/tmp/template.npy', template_concat, allow_pickle=False
            )
    import matplotlib.pyplot as plt
    plt.plot([e[0] for e in epoch_history]) # training loss
    plt.savefig('/tmp/losses.png')
    assert 0


if __name__ == '__main__':
    main()
